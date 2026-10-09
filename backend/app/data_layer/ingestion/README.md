# Ingestion (`ingestion/`)

## What this module does

The pipeline that turns a folder of files into vectors.

## Data flow

```
folder
  -> FileLoader          Dict[extension, List[Path]]
  -> TextExtractor       Dict[path, text]
  -> TextNormalizer      NormalizedContent (+ SectionSpan offsets)
  -> Chunker             HChunk / RChunk  (+ rows in SQLite)
  -> EmbeddingManager    EmbeddedChunk (vector + vector_id)
  -> pgvector            the vector, under its vector id
  -> vector_meta_data    a DiskANN label for it
  -> IndexGenerations    a new index generation, every INDEX_REBUILD_AT vectors
```

`ingestion_pipeline.py` is a facade over those stages, and deliberately adds no
logic of its own — it is an interface, not a stage. The last three are
`vector_db_manager/`'s; why the vector is stored before its label, and why the
index is built in generations, is in its README. It wraps the five expensive
stages in `log_timing`, which is the only place the cost of a whole run is
visible in one log.

## Why the stages are ordered this way

**Normalization happens before chunking, and finds the sections.** The chunkers
used to re-detect headings themselves, on text the normalizer had already
reshaped, so the two disagreed about whether a document had sections at all.
The normalizer is now the only heading detector and hands down `SectionSpan`
offsets.

**Chunking happens before embedding** so that a chunk id — which binds position
as well as content — exists before a vector id is derived from it. Deriving the
vector id from chunk *text* collapsed every repeated paragraph onto one id and
made all but one copy unreachable.

**Ids bind position, not just content.** Content repeats: a licence header, a
boilerplate paragraph, two `NOTES` headings in one document. Any id acting as a
primary key therefore includes an ordinal or an offset.

**Writes are `on conflict do nothing`**, so re-ingesting an unchanged folder is
a no-op rather than a primary-key failure.

## Sub-modules

| Directory | Does |
| :--- | :--- |
| `TextFileProcessor/` | Find candidate files; decode any of them to text |
| `normalizer/` | Clean text line by line; emit section offsets |
| `Chunker/` | Split into chunks; persist the hierarchy to SQLite |
| `embedding/` | Text to vectors, and vector ids |
| `nodes/`, `metadata/` | The dataclasses passed between stages |
