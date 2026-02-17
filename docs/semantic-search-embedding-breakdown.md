# Semantic Search Embedding Breakdown

Example file: `gloggur/embeddings/base.py`

## What is embedded

Gloggur embeds one chunk per extracted symbol (class/function/interface), not fixed-size token windows.

For each symbol, embedding text is built as:

1. `signature` (first source line of the symbol)
2. `docstring` (Python docstring or nearby comment)
3. `snippet` (3-line window starting at `start_line`)

These non-empty parts are joined with newlines:

```text
chunk_text = "\n".join([signature, docstring, snippet])
```

This behavior comes from:

- `gloggur/parsers/treesitter_parser.py` (symbol extraction + metadata)
- `gloggur/indexer/indexer.py` (`_symbol_text` and `_apply_embeddings`)

## Runtime embedding provider in this workspace

- `embedding_profile`: `local:microsoft/codebert-base`
- provider class: `LocalEmbeddingProvider`
- runtime mode: deterministic fallback (`_use_fallback=True`)
- fallback marker: `.gloggur-cache/.local_embedding_fallback` (present)
- embedding dimension produced here: `256`

Fallback embeddings in `LocalEmbeddingProvider` are produced by:

1. tokenizing lowercase text with regex `[A-Za-z0-9_]+`
2. hashing each token into a deterministic pseudo-random 256-dim vector
3. summing token vectors and L2-normalizing the result

## Example chunking + annotations

Source file has 4 extracted symbols, so 4 embedding chunks.

### Chunk 1: `EmbeddingProvider` (class)

Parser/index annotations:

- `id`: `gloggur/embeddings/base.py:6:EmbeddingProvider`
- `name`: `EmbeddingProvider`
- `kind`: `class`
- `file_path`: `gloggur/embeddings/base.py`
- `start_line`: `7`
- `end_line`: `22`
- `signature`: `class EmbeddingProvider(ABC):`
- `docstring`: `Abstract interface for embedding providers.`
- `body_hash`: `a5977db3e562572e342ee55001dfdc5b125b1928fd931a97bbc248a6bd04bf16`
- `language`: `python`
- `embedding_dimension`: `256`
- `embedding_preview_first_8`: `[0.018195, -0.129857, 0.000566, -0.070583, -0.039889, 0.083351, 0.142366, 0.051082]`

Embedded `chunk_text`:

```python
class EmbeddingProvider(ABC):
Abstract interface for embedding providers.
class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""
    @abstractmethod
```

### Chunk 2: `embed_text` (function)

Parser/index annotations:

- `id`: `gloggur/embeddings/base.py:9:embed_text`
- `name`: `embed_text`
- `kind`: `function`
- `file_path`: `gloggur/embeddings/base.py`
- `start_line`: `10`
- `end_line`: `12`
- `signature`: `def embed_text(self, text: str) -> list[float]:`
- `docstring`: `Embed a single text string.`
- `body_hash`: `e23a0133bf258b78cf6f3eea36a8813ff79e27e0a3054487139561678267891d`
- `language`: `python`
- `embedding_dimension`: `256`
- `embedding_preview_first_8`: `[-0.008171, -0.00785, 0.004567, 0.032761, 0.031255, -0.054258, 0.11596, -0.086632]`

Embedded `chunk_text`:

```python
def embed_text(self, text: str) -> list[float]:
Embed a single text string.
def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        raise NotImplementedError
```

### Chunk 3: `embed_batch` (function)

Parser/index annotations:

- `id`: `gloggur/embeddings/base.py:14:embed_batch`
- `name`: `embed_batch`
- `kind`: `function`
- `file_path`: `gloggur/embeddings/base.py`
- `start_line`: `15`
- `end_line`: `17`
- `signature`: `def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:`
- `docstring`: `Embed a batch of text strings.`
- `body_hash`: `9ded1a09567ab465bd19ec0defe5b62d36658b1755f3fe26c06b568458cbd4be`
- `language`: `python`
- `embedding_dimension`: `256`
- `embedding_preview_first_8`: `[0.002918, -0.023552, 0.017378, 0.016461, 0.036518, -0.074892, -0.015074, -0.117004]`

Embedded `chunk_text`:

```python
def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
Embed a batch of text strings.
def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        raise NotImplementedError
```

### Chunk 4: `get_dimension` (function)

Parser/index annotations:

- `id`: `gloggur/embeddings/base.py:19:get_dimension`
- `name`: `get_dimension`
- `kind`: `function`
- `file_path`: `gloggur/embeddings/base.py`
- `start_line`: `20`
- `end_line`: `22`
- `signature`: `def get_dimension(self) -> int:`
- `docstring`: `Return the vector dimensionality for embeddings.`
- `body_hash`: `b4a091cff3366d1452d826cdbba29df6b218eaa01ab347730538cd783715432f`
- `language`: `python`
- `embedding_dimension`: `256`
- `embedding_preview_first_8`: `[0.044086, -0.009116, -0.046496, -0.1009, -0.000501, 0.056298, -0.040683, -0.018104]`

Embedded `chunk_text`:

```python
def get_dimension(self) -> int:
Return the vector dimensionality for embeddings.
def get_dimension(self) -> int:
        """Return the vector dimensionality for embeddings."""
        raise NotImplementedError
```

## Where annotations are stored

- Symbol metadata (including `signature`, `docstring`, `body_hash`, `embedding_vector`) is stored in `.gloggur-cache/index.db` (`symbols` table).
- Vector index IDs are stored in `.gloggur-cache/vectors.json`.
- Vector data is stored in `.gloggur-cache/vectors.index` (FAISS) or `.gloggur-cache/vectors.npy` (fallback).
