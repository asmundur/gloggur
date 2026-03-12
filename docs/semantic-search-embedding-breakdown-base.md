# Semantic Search Embedding Breakdown

Generated: `2026-03-12T13:18:36.593545+00:00`
Source file: `src/gloggur/embeddings/base.py`

## Embedding Setup

- `embedding_profile`: `test:microsoft/codebert-base|embed_graph_edges=0`
- `provider_class`: `DeterministicTestEmbeddingProvider`
- `provider_module`: `gloggur.embeddings.test_provider`
- `embedding_dimension`: `256`

## Chunking Rule

The report uses the same chunk builder as the live indexer. A symbol may produce one or more chunks when its body exceeds the configured chunk-size budget.

```text
chunk_text = join_non_empty([
  fqname/kind/file/line header,
  signature block,
  imports-in-scope block (when present),
  bounded body slice
])
```

## Chunks (4)

### Chunk 1: `EmbeddingProvider` (class)

- Annotations:
  - `chunk_id`: `41ce766bc424c2a1e53529e662d7a781f48614d21d0db73ca18f90f7111d3005`
  - `symbol_id`: `b7d588b966aa6ad12e6b037ae8910598666ba0b96492287057e552d02d061600`
  - `name`: `EmbeddingProvider`
  - `kind`: `class`
  - `fqname`: `EmbeddingProvider`
  - `file_path`: `src/gloggur/embeddings/base.py`
  - `start_line`: `7`
  - `end_line`: `23`
  - `start_byte`: `111`
  - `end_byte`: `668`
  - `chunk_part_index`: `1`
  - `chunk_part_total`: `1`
  - `signature`: `class EmbeddingProvider(ABC):`
  - `docstring`: `Abstract interface for embedding providers.`
  - `body_hash`: `5c19c0c21d9bcee1543704e45c512bb7c08aacbf1304fbe02927c3df4b3e4977`
  - `language`: `python`
  - `embedding_dimension`: `256`
  - `embedding_preview_first_8`: `[0.025422, -0.006287, 0.02799, -0.006596, 0.015515, -0.017825, 0.011964, -0.020011]`

- `chunk_text`:

```python
FQNAME: EmbeddingProvider
KIND: class
FILE: src/gloggur/embeddings/base.py
LINES: 7-23

SIGNATURE:
RAW: class EmbeddingProvider(ABC):
PARAMS: ABC
RETURNS: <unknown>
MODIFIERS: <none>

IMPORTS_IN_SCOPE:
from abc import ABC, abstractmethod
from collections.abc import Iterable

BODY:

class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        raise NotImplementedError

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimensionality for embeddings."""
        raise NotImplementedError
```

### Chunk 2: `embed_text` (function)

- Annotations:
  - `chunk_id`: `7ac359d7390230888d940da2c0a75483d001fb3a6450f658118d9b150c06d8fc`
  - `symbol_id`: `a5c31575e44b94e8172ed3a9fb0571487dacab93a2c69c0f65b6530438cbc251`
  - `name`: `embed_text`
  - `kind`: `function`
  - `fqname`: `EmbeddingProvider.embed_text`
  - `file_path`: `src/gloggur/embeddings/base.py`
  - `start_line`: `10`
  - `end_line`: `13`
  - `start_byte`: `196`
  - `end_byte`: `344`
  - `chunk_part_index`: `1`
  - `chunk_part_total`: `1`
  - `signature`: `def embed_text(self, text: str) -> list[float]:`
  - `docstring`: `Embed a single text string.`
  - `body_hash`: `afa046facbd9c13e233282202b05eaa7a7f80e91d1231224f53c7b803cf30cc0`
  - `language`: `python`
  - `embedding_dimension`: `256`
  - `embedding_preview_first_8`: `[-0.033608, -0.025162, -0.014561, 0.023483, 0.084552, -0.046878, 0.035469, -0.067268]`

- `chunk_text`:

```python
FQNAME: EmbeddingProvider.embed_text
KIND: function
FILE: src/gloggur/embeddings/base.py
LINES: 10-13

SIGNATURE:
RAW: def embed_text(self, text: str) -> list[float]:
PARAMS: self, text: str
RETURNS: list[float]
MODIFIERS: <none>

IMPORTS_IN_SCOPE:
from abc import ABC, abstractmethod

BODY:

@abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        raise NotImplementedError
```

### Chunk 3: `embed_batch` (function)

- Annotations:
  - `chunk_id`: `121ff26a228f9428ed490a4eba241156ae2565c07a33b52e426660ac1fc2045f`
  - `symbol_id`: `61ec0ec828ac7dddc53a20a3ced5c6919aadf7158b8b00206d89c4abb2880c3b`
  - `name`: `embed_batch`
  - `kind`: `function`
  - `fqname`: `EmbeddingProvider.embed_batch`
  - `file_path`: `src/gloggur/embeddings/base.py`
  - `start_line`: `15`
  - `end_line`: `18`
  - `start_byte`: `345`
  - `end_byte`: `514`
  - `chunk_part_index`: `1`
  - `chunk_part_total`: `1`
  - `signature`: `def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:`
  - `docstring`: `Embed a batch of text strings.`
  - `body_hash`: `6f2e958095945aad1a04b540a85a64d66a7fe306ca0197e2d8a7be6fee9a422f`
  - `language`: `python`
  - `embedding_dimension`: `256`
  - `embedding_preview_first_8`: `[-0.016689, -0.018459, 0.017123, 0.020331, 0.08447, -0.09396, -0.067613, -0.056348]`

- `chunk_text`:

```python
FQNAME: EmbeddingProvider.embed_batch
KIND: function
FILE: src/gloggur/embeddings/base.py
LINES: 15-18

SIGNATURE:
RAW: def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
PARAMS: self, texts: Iterable[str]
RETURNS: list[list[float]]
MODIFIERS: <none>

IMPORTS_IN_SCOPE:
from abc import ABC, abstractmethod
from collections.abc import Iterable

BODY:

@abstractmethod
    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Embed a batch of text strings."""
        raise NotImplementedError
```

### Chunk 4: `get_dimension` (function)

- Annotations:
  - `chunk_id`: `1f0ce0a2d434f3d3fd1a9b48f8df472e07cadf2520da248f3d1994dd92d7d56c`
  - `symbol_id`: `b6c1827651f8394bf39bca585e77f17f463e8dc5d14d6151ecb6d87b156d50bb`
  - `name`: `get_dimension`
  - `kind`: `function`
  - `fqname`: `EmbeddingProvider.get_dimension`
  - `file_path`: `src/gloggur/embeddings/base.py`
  - `start_line`: `20`
  - `end_line`: `23`
  - `start_byte`: `515`
  - `end_byte`: `668`
  - `chunk_part_index`: `1`
  - `chunk_part_total`: `1`
  - `signature`: `def get_dimension(self) -> int:`
  - `docstring`: `Return the vector dimensionality for embeddings.`
  - `body_hash`: `fcb2a1421dae74fd74f0657c1152f347ca12cf7ac6cea60a2ecba910702fe40c`
  - `language`: `python`
  - `embedding_dimension`: `256`
  - `embedding_preview_first_8`: `[0.003115, 0.034014, 0.007503, -0.054413, 0.027291, 0.072198, -0.083954, -0.014615]`

- `chunk_text`:

```python
FQNAME: EmbeddingProvider.get_dimension
KIND: function
FILE: src/gloggur/embeddings/base.py
LINES: 20-23

SIGNATURE:
RAW: def get_dimension(self) -> int:
PARAMS: self
RETURNS: int
MODIFIERS: <none>

IMPORTS_IN_SCOPE:
from abc import ABC, abstractmethod

BODY:

@abstractmethod
    def get_dimension(self) -> int:
        """Return the vector dimensionality for embeddings."""
        raise NotImplementedError
```
