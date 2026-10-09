"""The one definition of a chunk's vector id."""

import hashlib

from config import Config


def vector_id_for(chunk_id: str) -> int:
    """Derive the vector id from the chunk id, never from the chunk text."""
    hash_bytes = hashlib.md5(chunk_id.encode("utf-8")).digest()
    uint64_id = int.from_bytes(hash_bytes[:8], byteorder="little", signed=False)
    # Without the mask ~49% of ids overflow a signed 64-bit column.
    return uint64_id & Config.VECTOR_ID_MASK
