# Awareness Studio

Minimal local app: **Book Generator CLI** + **Guidance Chatbot CLI** backed by Notion-exported knowledge pages.

## Setup

```bash
pip install -e apps/awareness_studio
cp apps/awareness_studio/.env.example apps/awareness_studio/.env
# Edit .env — add your ANTHROPIC_API_KEY
```

## Drop in your Notion exports

Place exported `.md` files in:
```
apps/awareness_studio/inputs/notion_export/
```

Then rebuild the index:
```bash
python -m awareness_studio.index_build
```

Sample files for all 7 canonical pages are already included.

## Chat CLI

```bash
python -m awareness_studio.chat_cli --mode EXPLAIN --question "What is not-self via controllability?"
python -m awareness_studio.chat_cli --mode TEACH --question "What is consciousness?"
python -m awareness_studio.chat_cli --mode MATRIX --question "Explain tanha as gain and upadana as latch"
python -m awareness_studio.chat_cli --mode ELABORATE --question "What is the second arrow?"
python -m awareness_studio.chat_cli --mode CARD --question "Vedana and the cut-point"
```

Available modes: `TEACH` `EXPLAIN` `ELABORATE` `MATRIX` `CARD` `CANONICAL`

## Book Generator CLI

```bash
python -m awareness_studio.book_generator --quadrant q1 --chapter "Soltar es aflojar" --words 1200
python -m awareness_studio.book_generator --quadrant q2 --chapter "Vedana precision" --words 1400
python -m awareness_studio.book_generator --quadrant q3 --chapter "Samsara as loops" --words 900
python -m awareness_studio.book_generator --quadrant q4 --chapter "Gain control and liberation" --words 1400
```

Quadrant voices:
| Flag | Voice |
|------|-------|
| `q1` | Warm, story-first, autoayuda práctica |
| `q2` | Pali/EBT dense, Theravāda avanzado |
| `q3` | Skeptical science-pop, info-theory analogies |
| `q4` | PhD rigor, formal claims, falsifiers |

## File naming for source kind inference

| File pattern | `source_kind` |
|---|---|
| `book_system*.md` | `book_system` |
| `*q1*.md`, `*autoayuda*.md` | `book_seed_q1` |
| `*q2*.md`, `*therav*.md` | `book_seed_q2` |
| `*q3*.md`, `*esceptic*.md` | `book_seed_q3` |
| `*q4*.md`, `*liberation*.md` | `book_seed_q4` |
| `*answer_template*.md`, `*monk*.md` | `answer_templates` |
| `*rag_plan*.md`, `*dev*plan*.md` | `rag_plan` |
| anything else | `other` |

## Run tests

```bash
cd apps/awareness_studio
pip install -e ".[dev]"
pytest
```

## Architecture

```
inputs/notion_export/*.md
        ↓ io_markdown.py (load + infer source_kind)
        ↓ chunking.py (heading-aware, stable chunk IDs)
        ↓ index_build.py (persist JSON + BM25 index)
        ↓ retrieval.py (pure-Python BM25)
        ↓ prompts.py + answer_modes.py (Monk+Scientist templates)
        ↓ llm_client.py (anthropic, swappable)
   chat_cli.py | book_generator.py
```

All retrieval is offline BM25 (no embeddings required). LLM calls require `ANTHROPIC_API_KEY`.
