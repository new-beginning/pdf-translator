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
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `vi` | Target language (see below) |
| `--dpi` | `200` | Page render resolution |
| `--pages` | all | Limit to first N pages |

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
