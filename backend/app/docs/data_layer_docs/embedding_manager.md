# Embedding Manager Module (`embedding/EmbeddingManager.py`)

## Overview & Purpose
The `EmbeddingManager` module wraps HuggingFace's `sentence-transformers` library (`SentenceTransformer`) to convert text chunks (`HChunk` or `RChunk`) into high-dimensional floating-point vector representations (`List[float]`). It attaches provenance metadata (`EmbeddedChunkMetaData`) to every generated embedding to track model lineage.

---

## Classes & Public APIs

### `class EmbeddingManager`
Manages model loading and exposes unified encoding endpoints capable of handling single chunks or batched lists of chunks.

#### Constructor: `__init__(self, model_name: str = Config.EMBEDDING_MODEL, embeddding_dimension: int = Config.EMBEDDING_DIMENSIONS) -> None`
Initializes the sentence transformer model in memory.

##### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `Config.EMBEDDING_MODEL` (`"sentence-transformers/all-MiniLM-L6-v2"`) | HuggingFace repository identifier or local filesystem path of the target embedding model. |
| `embeddding_dimension` | `int` | `Config.EMBEDDING_DIMENSIONS` (`128`) | Truncation width passed to `model.encode(..., truncate_dim=...)`. |

---

## Vector Identifiers

`__generate_vector_id(chunk_id)` derives the id stored in DiskANN and pgvector:

```
md5(chunk_id)[:8]  ->  little-endian uint64  ->  & Config.VECTOR_ID_MASK
```

Two properties matter, and each fixes a distinct failure:

- **Derived from the chunk id, never the chunk text.** Chunk text repeats — a licence header, a boilerplate paragraph, a `"yes"` conversation turn — so hashing it assigned one vector id to several distinct chunks and all but one became unreachable in the index. Ingesting this repository lost **385 of 4033 chunks (9.5%)** this way. The ids the chunkers emit already bind position as well as content, so deriving from them is collision-free by construction.
- **Masked into the non-negative signed 64-bit range.** Vector ids land in SQLite `INTEGER` and Postgres `bigint`, both signed. An unmasked MD5-derived id overflows both for roughly half of all inputs.

`embed_text()` derives `chunk_id` from a SHA-256 of the text when the caller omits it, so a raw string with no id of its own still gets a deterministic vector id.

---

#### Methods

##### `embed(self, arg: Union[HChunk, RChunk, List[Union[HChunk, RChunk]]]) -> Union[EmbeddedChunk, List[EmbeddedChunk]]`
The polymorphic public endpoint that generates dense vector embeddings for input chunks.

###### Parameters
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `arg` | `Union[HChunk, RChunk, List[Union[HChunk, RChunk]]]` | A single hierarchical or recursive chunk object (`HChunk` / `RChunk`), or a list containing multiple chunk objects. |

###### Return Value
- **Type:** `Union[EmbeddedChunk, List[EmbeddedChunk]]`
- **Description:** If `arg` is a single chunk object, returns a single `EmbeddedChunk` (`vector: np.ndarray[float32]`, `vector_id: int`, `meta_data: EmbeddedChunkMetaData`). If `arg` is a list, returns a list of `EmbeddedChunk` instances preserving the original order.

###### How It Works
1. Inspects the type of `arg`:
   - **Single Chunk (`isinstance(arg, HChunk | RChunk)`)**: Delegates to `__embed_chunk(arg)`. Encodes `chunk.chunk`, casts the vector to `np.float32` (DiskANN requires a NumPy array, not a Python list), and derives `vector_id` from `chunk.chunk_id`.
   - **Chunk List (`isinstance(arg, list)`)**: Delegates to `__embed_chunks(arg)`. Verifies every element is an `HChunk` or `RChunk` (`raise InvalidEmbeddingArgument` on mismatch), then encodes the texts in batches of 64 and zips each vector back to its source chunk.
   - **Invalid Type**: Raises `InvalidEmbeddingArgument` if `arg` is neither a valid chunk object nor a list.

---

##### `embed_text(self, text: str, chunk_id: str | None = None) -> EmbeddedChunk`
Embeds a raw string that never passed through the `Chunker`. The memory layer needs this for conversation summaries and turns, which are not `HChunk`/`RChunk`.

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `text` | `str` | Non-empty string to embed. Anything else raises `InvalidEmbeddingArgument`. |
| `chunk_id` | `str \| None` | Identifier to bind the vector id to. Derived from a SHA-256 of `text` when omitted. |

---

##### `embed_texts(self, texts: List[str], chunk_ids: List[str] | None = None) -> List[EmbeddedChunk]`
Batch form of `embed_text`. `chunk_ids`, when given, must align with `texts` or `InvalidEmbeddingArgument` is raised.
