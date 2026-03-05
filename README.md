# Glöggur

**Glöggur** is a local semantic search tool designed for coding agents and developers.  
It augments traditional text search with embeddings so agents can locate relevant code and context even when exact keywords are missing.

The goal is simple: **make codebases navigable for LLM agents without requiring external services or complex infrastructure.**

---

## Core Idea

Agents already know how to use tools like `grep`.  
Glöggur keeps that workflow but adds **semantic retrieval** so agents can search by meaning, not just text.

Instead of:

```

grep "authentication middleware"

```

Agents can run:

```

gloggur search --query "authentication middleware that validates jwt tokens"

```

Glöggur then returns the most semantically relevant code fragments.

---

## Key Principles

- **Local-first**  
  Runs on your machine. No server required.

- **Agent-compatible interface**  
  CLI designed to mirror patterns agents already use (`grep`, `rg`, etc.).

- **Semantic + lexical retrieval**  
  Combines embeddings with traditional text search.

- **Minimal infrastructure**  
  Embeddings are the only optional external dependency.

---

## Features

### 1. Semantic Code Search

Find code based on intent rather than keywords.

```

gloggur search "where does the app send password reset emails"

```

Returns ranked code snippets and file locations.

---

### 2. Agent-Friendly Grep Interface

Agents frequently default to regex search.  
Glöggur provides a compatible interface while enriching results with semantic context.

```

gloggur grep "jwt|token"

```

Agents can combine lexical and semantic queries.

---

### 3. Repository Indexing

Before searching, a repository is indexed into an embedding store.

```

gloggur index .

```

Indexing extracts code chunks and metadata needed for retrieval.

---

### 4. Hybrid Retrieval

Queries combine:

- embedding similarity
- keyword matches
- file metadata

This improves precision when agents explore unfamiliar repositories.

---

## Example Workflow

```

gloggur index .

gloggur search "database connection initialization"

gloggur grep "postgres|db.connect"

```

Agents can alternate between semantic exploration and precise filtering.

---

## Design Philosophy

Glöggur is **not an agent framework**.

It is a **tool for agents**.

The focus is narrow:

- make repositories searchable
- improve agent code navigation
- stay lightweight and composable

Agents remain responsible for reasoning and decision making.

---

## Intended Users

- coding agents
- developers exploring large repositories
- automated code analysis tools

---

## Status

Active development.

Current priorities:

- improve indexing reliability
- optimize retrieval ranking
- strengthen agent CLI compatibility


