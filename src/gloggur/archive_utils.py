from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveFileSource:
    """Mapping from an on-disk file to a deterministic archive path."""

    source_path: Path
    archive_path: str


def sha256_file(path: str | Path) -> str:
    """Return SHA256 digest for a file using chunked reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return SHA256 digest for bytes payloads."""
    return hashlib.sha256(payload).hexdigest()


def create_deterministic_tar_gz(
    archive_path: str | Path,
    *,
    file_sources: Sequence[ArchiveFileSource],
    extra_files: Sequence[tuple[str, bytes]] = (),
) -> None:
    """Create a deterministic tar.gz archive with stable ordering and metadata."""
    archive_target = Path(archive_path)
    sorted_sources = sorted(file_sources, key=lambda item: item.archive_path)
    sorted_extras = sorted(extra_files, key=lambda item: item[0])

    with archive_target.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for file_source in sorted_sources:
                    tar_info = tar.gettarinfo(
                        str(file_source.source_path),
                        arcname=file_source.archive_path,
                    )
                    tar_info.uid = 0
                    tar_info.gid = 0
                    tar_info.uname = ""
                    tar_info.gname = ""
                    tar_info.mtime = 0
                    with file_source.source_path.open("rb") as source_handle:
                        tar.addfile(tar_info, source_handle)

                for archive_name, payload in sorted_extras:
                    tar_info = tarfile.TarInfo(name=archive_name)
                    tar_info.size = len(payload)
                    tar_info.mode = 0o644
                    tar_info.uid = 0
                    tar_info.gid = 0
                    tar_info.uname = ""
                    tar_info.gname = ""
                    tar_info.mtime = 0
                    tar.addfile(tar_info, io.BytesIO(payload))
