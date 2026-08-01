# Embedding Manager Module (`embedding/EmbeddingManager.py`)

## Overview & Purpose
The `EmbeddingManager` module wraps HuggingFace's `sentence-transformers` library (`SentenceTransformer`) to convert text chunks (`HChunk` or `RChunk`) into high-dimensional floating-point vector representations (`List[float]`). It attaches provenance metadata (`EmbeddedChunkMetaData`) to every generated embedding to track model lineage.

---

## Classes & Public APIs

### `class EmbeddingManager`
Manages model loading and exposes unified encoding endpoints capable of handling single chunks or batched lists of chunks.

#### Constructor: `__init__(self, model_name: str = Config.EMBEDDING_MODEL) -> None`
Initializes the sentence transformer model in memory.

##### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | `Config.EMBEDDING_MODEL` (`"sentence-transformers/all-MiniLM-L6-v2"`) | HuggingFace repository identifier or local filesystem path of the target embedding model. |

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
- **Description:** If `arg` is a single chunk object, returns a single `EmbeddedChunk` (`vector: List[float], meta_data: EmbeddedChunkMetaData`). If `arg` is a list, returns a list of `EmbeddedChunk` instances preserving the original order.

###### How It Works
1. Inspects the type of `arg`:
   - **Single Chunk (`isinstance(arg, HChunk | RChunk)`)**: Delegates to `__embed_chunk(arg)`. Extracts `chunk.chunk` text and runs `self.model.encode(chunk)`. Converts the resulting NumPy array to a Python float list (`embedded_chunk.tolist()`) and wraps it in `EmbeddedChunk(vector, meta_data)`.
   - **Chunk List (`isinstance(arg, list)`)**: Delegates to `__embed_chunks(arg)`. Verifies that all elements in the list are valid `HChunk` or `RChunk` instances (`raise InvalidEmbeddingArgument` on mismatch). Collects all chunk string texts into `texts = [chunkObj.chunk for chunkObj in chunks]` and performs batch encoding via `self.model.encode(texts)`. Zips each resulting vector with its source chunk object and returns a list of `EmbeddedChunk` objects.
   - **Invalid Type**: Raises `InvalidEmbeddingArgument` if `arg` is neither a valid chunk object nor a list.
