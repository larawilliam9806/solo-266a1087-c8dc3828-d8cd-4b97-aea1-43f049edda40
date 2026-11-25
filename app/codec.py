"""固定内存的校验片（XOR 补零）计算、缺片恢复与流式拼接。

语义：把每片视为尾部补零到「最长片长度」的字节串，逐字节异或。

恢复缺片时只处理缺片长度以内的前缀：
  - 上传的校验片不落盘，直接按固定大小分块读取并参与异或（请求体缓存由
    框架负责，不计入补片写出的数据）；
  - 每片已验收片最多只读取「缺片长度」字节的前缀，较短者按补零规则处理；
  - 累加器就是本次补片的目标文件本身，最终原子替换到目标路径，因此一次
    补片新建的数据量不超过缺片的字节数；
  - 同一时刻打开的已验收片不超过 MAX_OPEN_FILES 个；超出时以目标文件为
    累加器分批折叠，内存与文件句柄数均不随片数或片大小增长。
"""

from __future__ import annotations

import hashlib
import os
import tempfile

from .storage import READ_SIZE, StorageError, fsync_dir, replace_file

# 同一批次最多同时打开的已验收片数。分批折叠时复用同一上限。
MAX_OPEN_FILES = 32


class CodecError(StorageError):
    """校验片计算或恢复过程中的数据问题。"""


class CheckshardLengthError(CodecError):
    """校验片的实际字节数与其应在长度（清单最长片长度）不一致。"""

    def __init__(self, expected: int, actual: int | None):
        self.expected = expected
        self.actual = actual
        if actual is None:
            detail = "实收超过该长度"
        elif actual > expected:
            detail = f"实收 {actual} 字节（超出）"
        else:
            detail = f"实收 {actual} 字节"
        super().__init__(f"校验片长度应为 {expected} 字节（最长片长度），{detail}")


def _xor_fold(
    sources: list[tuple[str, int]],
    length: int,
    chunk_size: int,
    directory: str,
) -> str:
    """把 sources 尾部补零到 length 后逐字节异或，写入临时累加器文件。

    返回累加器路径（长度恰为 length）。调用方负责替换或删除。
    """
    os.makedirs(directory, exist_ok=True)
    fd, acc_path = tempfile.mkstemp(prefix=".xor-", dir=directory)
    os.close(fd)
    try:
        with open(acc_path, "w+b") as acc:
            acc.truncate(length)  # 稀疏文件，内容为零
            _xor_prefix_into(acc, sources, length, chunk_size)
        return acc_path
    except BaseException:
        try:
            os.unlink(acc_path)
        except OSError:
            pass
        raise


def _digest_file(path: str, length: int, chunk_size: int) -> str:
    digest = hashlib.sha256()
    remaining = length
    with open(path, "rb") as handle:
        while remaining > 0:
            block = handle.read(min(chunk_size, remaining))
            if not block:
                raise CodecError("文件长度不足")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def build_checkshard(
    sources: list[tuple[str, int]],
    length: int,
    dest: str,
    chunk_size: int = READ_SIZE,
) -> tuple[int, str]:
    """按「尾部补零至 length 后逐字节异或」生成校验片，原子落到 dest。

    sources 为 (文件路径, 原始字节数) 列表，通常就是全部原片。
    返回 (字节数, sha256)。
    """
    directory = os.path.dirname(dest) or "."
    acc_path = _xor_fold(sources, length, chunk_size, directory)
    try:
        digest = _digest_file(acc_path, length, chunk_size)
        replace_file(acc_path, dest)
        acc_path = None
        fsync_dir(directory)
        return length, digest
    finally:
        if acc_path is not None:
            try:
                os.unlink(acc_path)
            except OSError:
                pass


def _xor_prefix_into(
    acc,
    sources: list[tuple[str, int]],
    length: int,
    chunk_size: int,
) -> None:
    """把各片不超过 length 字节的前缀逐块异或进已打开的累加器 acc。

    每片最多读取 length 字节；短片在尾部按补零规则处理（不再读取）。
    分批打开源文件，同一时刻打开的已验收片不超过 MAX_OPEN_FILES 个。
    累加器跨批次复用，因此内存与打开的文件句柄数都不随片数增长。
    """
    for start in range(0, len(sources), MAX_OPEN_FILES):
        batch = sources[start : start + MAX_OPEN_FILES]
        handles = []
        try:
            for path, _size in batch:
                handles.append(open(path, "rb"))
            position = 0
            while position < length:
                want = min(chunk_size, length - position)
                acc.seek(position)
                acc_block = acc.read(want)
                if len(acc_block) != want:
                    raise CodecError("累加器长度异常")
                acc_value = int.from_bytes(acc_block, "little")
                for handle, (path, size) in zip(handles, batch):
                    remaining = size - position
                    if remaining <= 0:
                        continue  # 该片已到尾部，余下按零补齐
                    take = min(want, remaining)
                    block = handle.read(take)
                    if len(block) != take:
                        raise CodecError(
                            f"{os.path.basename(path)} 长度不足，与清单声明不符"
                        )
                    acc_value ^= int.from_bytes(block, "little")
                acc.seek(position)
                acc.write(acc_value.to_bytes(want, "little"))
                position += want
        except OSError as exc:
            raise StorageError(f"读写分片失败：{exc.strerror or exc}") from exc
        finally:
            for handle in handles:
                try:
                    handle.close()
                except OSError:
                    pass


def recover_part(
    checkshard,
    checkshard_length: int,
    sources: list[tuple[str, int]],
    target_length: int,
    dest: str,
    chunk_size: int = READ_SIZE,
) -> tuple[int, str]:
    """校验片 ⊕ 其余片 = 缺片；只处理缺片长度以内的前缀并写到 dest。

    checkshard 可以是带 read() 的上传流（推荐，无需落盘），也可以是校验片
    文件路径。其实际字节数必须恰为 checkshard_length（清单最长片长度），
    否则抛出 CheckshardLengthError。sources 为已验收片的 (路径, 字节数)，
    每片最多读取 target_length 字节。dest 是本次补片唯一新建的文件（长度
    恰为 target_length），调用方核验 SHA-256 后再原子替换到正式位置，
    失败则删除。

    返回 (字节数, sha256)；调用方必须再与清单哈希核对后才计入交付。
    """
    if target_length > checkshard_length:
        raise CodecError("缺片长度大于校验片长度，无法恢复")
    directory = os.path.dirname(dest) or "."
    close_source = False
    if isinstance(checkshard, (str, bytes, os.PathLike)):
        try:
            checkshard = open(checkshard, "rb")
        except OSError as exc:
            raise StorageError(f"读取校验片失败：{exc.strerror or exc}") from exc
        close_source = True
    try:
        return _recover_part_stream(
            checkshard,
            checkshard_length,
            sources,
            target_length,
            dest,
            directory,
            chunk_size,
        )
    finally:
        if close_source:
            try:
                checkshard.close()
            except OSError:
                pass


def _recover_part_stream(
    checkshard_stream,
    checkshard_length: int,
    sources: list[tuple[str, int]],
    target_length: int,
    dest: str,
    directory: str,
    chunk_size: int,
) -> tuple[int, str]:
    try:
        os.makedirs(directory, exist_ok=True)
        acc = open(dest, "w+b")
    except OSError as exc:
        raise StorageError(f"写入分片失败：{exc.strerror or exc}") from exc

    try:
        with acc:
            acc.truncate(target_length)  # 先占位；随后按需覆盖前缀
            # 1. 校验片前缀直接写入累加器；其余字节读到并计数，用于核对长度。
            position = 0
            total = 0
            while True:
                block = checkshard_stream.read(chunk_size)
                if not block:
                    break
                total += len(block)
                if total > checkshard_length:
                    raise CheckshardLengthError(checkshard_length, None)
                if position < target_length:
                    take = min(len(block), target_length - position)
                    acc.seek(position)
                    acc.write(block[:take])
                    position += take
            if total != checkshard_length:
                raise CheckshardLengthError(checkshard_length, total)
            acc.flush()
            os.fsync(acc.fileno())

            # 2. 已验收片按前缀异或进来；短片余下部分保持零。
            _xor_prefix_into(acc, sources, target_length, chunk_size)
            acc.flush()
            os.fsync(acc.fileno())

        digest = _digest_file(dest, target_length, chunk_size)
        fsync_dir(directory)
        return target_length, digest
    except BaseException:
        try:
            os.unlink(dest)
        except OSError:
            pass
        raise


def iter_part(path: str, chunk_size: int = READ_SIZE):
    """按固定块大小产出单片字节，用于按片号升序流式拼接下载。"""
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            yield block


def checkshard_bytes(parts: list[bytes], length: int) -> bytes:
    """测试与工具用：纯内存版本，语义与 build_checkshard 相同。"""
    acc = bytearray(length)
    for part in parts:
        if len(part) > length:
            raise ValueError("part longer than checkshard length")
        for i, value in enumerate(part):
            acc[i] ^= value
    return bytes(acc)
