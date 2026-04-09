# pdf-translator

Converts PDFs into translated HTML pages. For each page it produces:

- the original page image
- a translated version with each text block replaced in-place

Both images are stacked in a single HTML file. Translated page images are also saved to a flat folder for easy upload (e.g. Facebook).

## How it works

1. Render each PDF page to an image (PyMuPDF)
2. Extract text blocks with bounding boxes — native PDF text layer first, Tesseract OCR fallback for scanned pages
3. Split blocks into paragraphs using vertical gaps and sentence-ending short lines
4. Translate all blocks in batches using [Meta NLLB-200 1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) running locally on Apple MPS (GPU)
5. White out each original text region and redraw with the translated text
6. Write one HTML per page with both images stacked

## Output layout

```
extracts/
  flat/
    page_001.jpg          # all translated pages, sequential across all PDFs
    page_002.jpg
    ...
  <source>/
    <name>/
      index.html
      page_01.html        # original + translated images
      page_02.html
      ...
```

## Benchmarks

**Machine:** MacBook Pro 14" (Mac14,10) · Apple M2 Pro · 12 CPU cores (8P + 4E) · 19 GPU cores · 32 GB unified memory · macOS 26

| Backend | PDF | Pages | DPI | Wall time | Sec/page | Notes |
|---------|-----|-------|-----|-----------|----------|-------|
| NLLB-200 1.3B (MPS) | Globe and Mail — Mar 28 2026 | 80 | 200 | 39m 10s | ~29.4s | Native text |
| NLLB-200 1.3B (MPS) | Globe and Mail — Mar 30 2026 | 32 | 200 | 16m 29s | ~30.9s | Native text |
| NLLB-200 1.3B (MPS) | NYT — Mar 28 2026 | 36 | 200 | 28m 35s | ~47.6s | All OCR · 141 tok_in/s |
| NLLB-200 1.3B (MPS) | NYT — Mar 29 2026 | 102 | 200 | 71m 03s | ~41.8s | All OCR · 163 tok_in/s |
| Apple Translation | Globe and Mail — Mar 30 2026 | 32 | 200 | **3m 19s** | **~6.2s** | Native text · **4.9× faster** |

**Apple Translation backend** (`--backend apple`) uses Apple's on-device Translation framework (macOS 15+). Requires the target language pack to be installed via **System Settings → General → Language & Region → Translation Languages**. Vietnamese is supported.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Tesseract (`brew install tesseract`)
- macOS with Apple Silicon recommended (MPS acceleration)

## Setup

```bash
uv sync
```

The NLLB model (~2.5 GB) is downloaded from Hugging Face on first run.

## Usage

```bash
# Process all PDFs under pdfs/
uv run python extract.py

# Process a single PDF
uv run python extract.py pdfs/folder/file.pdf

# Translate to French instead of Vietnamese
uv run python extract.py --lang fr

# First 5 pages only, at 300 DPI
uv run python extract.py --pages 5 --dpi 300

# Use Apple Translation (macOS 15+, ~5x faster, requires language pack installed)
uv run python extract.py --backend apple pdfs/folder/file.pdf
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `vi` | Target language (see below) |
| `--dpi` | `200` | Page render resolution |
| `--pages` | all | Limit to first N pages |
| `--backend` | `nllb` | `nllb` (default) or `apple` (Translation.framework, macOS 15+) |

### Supported languages

| Code | Language |
|------|----------|
| `vi` | Vietnamese |
| `fr` | French |
| `de` | German |
| `es` | Spanish |
| `zh` | Simplified Chinese |
| `ja` | Japanese |
| `ko` | Korean |
| `ar` | Arabic |
| `hi` | Hindi |
| `pt` | Portuguese |

Any [NLLB language token](https://huggingface.co/facebook/nllb-200-distilled-1.3B) (e.g. `vie_Latn`) can also be passed directly.

## Input layout

Place PDFs under `pdfs/<source-name>/`:

```
pdfs/
  folder_a/
    file.pdf
  folder_b/
    file.pdf
```
