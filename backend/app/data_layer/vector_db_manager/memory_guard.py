"""What the vector index may allocate before it degrades instead."""

import shutil
from pathlib import Path
from typing import Optional

from config import Config

MiB = 1 << 20
GiB = 1 << 30

# Margins over the raw sizes, from measurement; README.md has the numbers.
LOADED_FACTOR = 1.1
RECENT_SLOT_FACTOR = 1.75
BUILD_FACTOR = 2.0
BUILD_BASE = 128 * MiB
SMALLEST_DISK_BUILD = 256 * MiB
LARGEST_DISK_BUILD = 4 * GiB

MEMINFO = Path("/proc/meminfo")
OWN_CGROUP = Path("/proc/self/cgroup")
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _meminfo_available() -> Optional[int]:
    try:
        with open(MEMINFO) as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_headroom() -> Optional[int]:
    """Room left under the tightest cgroup v2 memory limit above this process."""
    try:
        with open(OWN_CGROUP) as groups:
            entry = next((line for line in groups if line.startswith("0::")), None)
    except OSError:
        return None
    if entry is None:
        return None
    root = CGROUP_ROOT
    group = root / entry.strip()[3:].lstrip("/")
    headroom = None
    while True:
        try:
            limit = (group / "memory.max").read_text().strip()
            if limit != "max":
                used = int((group / "memory.current").read_text())
                room = max(int(limit) - used, 0)
                headroom = room if headroom is None else min(headroom, room)
        except (OSError, ValueError):
            pass
        if group == root or group.parent == group:
            return headroom
        group = group.parent


def available_memory() -> Optional[int]:
    """Bytes the system and this process's cgroup can still give, or None if unknown."""
    known = [v for v in (_meminfo_available(), _cgroup_headroom()) if v is not None]
    return min(known) if known else None


def spare_memory() -> Optional[int]:
    """What may be allocated and still leave the configured headroom; None if unknown."""
    available = available_memory()
    if available is None:
        return None
    return max(available - Config.MEMORY_HEADROOM_MB * MiB, 0)


def can_hold(size: int) -> bool:
    """Whether `size` more bytes fit. Unknown memory admits it, on the budget alone."""
    spare = spare_memory()
    return spare is None or size <= spare


def free_disk(path: Path) -> int:
    """Free bytes on the filesystem `path` is, or would be, created on."""
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    return shutil.disk_usage(path).free


def budget() -> int:
    return Config.VECTOR_INDEX_RAM_MB * MiB


def vector_bytes(dimensions: int, degree: int) -> int:
    """One vector and its neighbour list."""
    return dimensions * 4 + degree * 4


def recent_index_bytes(capacity: int, dimensions: int, degree: int) -> int:
    """The recent index allocates every slot up front, used or not."""
    return int(capacity * vector_bytes(dimensions, degree) * RECENT_SLOT_FACTOR)


def memory_index_bytes(count: int, dimensions: int, degree: int) -> int:
    return int(count * vector_bytes(dimensions, degree) * LOADED_FACTOR)


def cache_node_bytes(dimensions: int, degree: int) -> int:
    return int(vector_bytes(dimensions, degree + 1) * LOADED_FACTOR)


def memory_build_bytes(count: int, dimensions: int, degree: int) -> int:
    return BUILD_BASE + int(count * vector_bytes(dimensions, degree) * BUILD_FACTOR)


def disk_build_budget() -> Optional[int]:
    """Bytes a disk build may use, or None if not even the smallest one fits now."""
    spare = spare_memory()
    allowed = LARGEST_DISK_BUILD if spare is None else min(LARGEST_DISK_BUILD, spare - BUILD_BASE)
    return allowed if allowed >= SMALLEST_DISK_BUILD else None


def index_disk_bytes(count: int, dimensions: int, degree: int) -> int:
    """Disk a build needs at its peak: the vectors file beside the finished index."""
    vectors = count * dimensions * 4
    return 3 * vectors + 2 * count * (degree + 1) * 4 + 64 * MiB
