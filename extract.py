#!/usr/bin/env python3
"""
PDF → per-page HTML with original + translated text overlay.

For each page:
  1. Render the page to an image.
  2. Extract text blocks with bounding boxes (native PDF text, or Tesseract OCR).
  3. Translate each block individually to Vietnamese (Meta NLLB-200 1.3B, via MPS).
  4. Produce two images:
       • Original page scan (unchanged)
       • Translated page: each text block region is whited out and redrawn
         with the Vietnamese translation
  5. Write one HTML per page with both images stacked.

Output layout:
  extracts/<source>/<name>/
    index.html
    page_01.html
    page_02.html
    ...

Usage:
    uv run python extract.py                        # all PDFs under pdfs/
    uv run python extract.py pdfs/folder/file.pdf   # one PDF
    uv run python extract.py --dpi 200              # render resolution
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import multiprocessing as mp
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import pytesseract
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEWS_DIR        = Path("pdfs")
EXTRACTS_DIR    = Path("extracts")
DEFAULT_DPI     = 200
MIN_NATIVE_TEXT = 80      # chars — below this, treat page as image-only
MODEL_NAME      = "facebook/nllb-200-distilled-1.3B"
MAX_TOKENS      = 480

# NLLB language token for the target language (set at startup via --lang).
# Common codes: vie_Latn (Vietnamese), fra_Latn (French), zho_Hans (Simplified Chinese),
#               deu_Latn (German), spa_Latn (Spanish), jpn_Jpan (Japanese)
_target_lang_token: str = "vie_Latn"
_target_lang_name:  str = "Vietnamese"   # human-readable (display only)
_target_lang_code:  str = "vi"           # BCP-47 short code, used by Apple backend
_backend:           str = "nllb"         # "nllb" | "apple"

# Friendly short-code → NLLB token mapping for --lang convenience
_LANG_MAP: dict[str, str] = {
    "vi": "vie_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "pt": "por_Latn",
}

# Short-code → human-readable name (used by Apple Foundation Models backend)
_LANG_NAMES: dict[str, str] = {
    "vi": "Vietnamese",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "zh": "Simplified Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "pt": "Portuguese",
}

# Font for drawing Vietnamese text on the translated image.
# Arial covers the full Vietnamese Unicode range.
_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",   # Linux
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
]
_FONT_PATH: str | None = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


# ---------------------------------------------------------------------------
# Translation — Meta NLLB-200 1.3B loaded once in main(), reused for every page
# Uses Apple MPS (GPU) when available, falls back to CPU.
# ---------------------------------------------------------------------------

_worker_tokenizer = None   # NllbTokenizerFast, src_lang="eng_Latn", loaded in main()
_worker_model     = None   # AutoModelForSeq2SeqLM (NLLB-200 distilled 1.3B), loaded in main()

# Token throughput tracking (updated by _run_batch, printed per page)
_total_tokens_in  = 0
_total_tokens_out = 0
_total_infer_sec  = 0.0

# ---------------------------------------------------------------------------
# Apple Foundation Models backend
# Requires macOS 26+. apple_translate.swift compiled to .apple_translate on
# first use and kept as a long-lived subprocess for the duration of the run.
# ---------------------------------------------------------------------------

_APPLE_TRANSLATE_SRC = Path(__file__).parent / "apple_translate.swift"
_APPLE_TRANSLATE_BIN = Path(__file__).parent / ".apple_translate"


def _ensure_apple_binary() -> None:
    src_mtime = _APPLE_TRANSLATE_SRC.stat().st_mtime if _APPLE_TRANSLATE_SRC.exists() else 0
    bin_mtime = _APPLE_TRANSLATE_BIN.stat().st_mtime if _APPLE_TRANSLATE_BIN.exists() else 0
    if _APPLE_TRANSLATE_BIN.exists() and bin_mtime >= src_mtime:
        return
    if not _APPLE_TRANSLATE_SRC.exists():
        sys.exit(f"Error: {_APPLE_TRANSLATE_SRC} not found")
    print("Compiling apple_translate.swift …", flush=True)
    r = subprocess.run(
        ["swiftc", "-O", str(_APPLE_TRANSLATE_SRC), "-o", str(_APPLE_TRANSLATE_BIN)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        sys.exit(f"swiftc failed:\n{r.stderr}")
    print("Compiled.", flush=True)


class AppleTranslator:
    """Spawns a fresh one-shot subprocess per page to avoid main-thread deadlocks
    in the Translation framework. Startup overhead is ~0.1s per page."""

    def __init__(self) -> None:
        _ensure_apple_binary()

    def translate(self, texts: list[str]) -> list[str]:
        if not texts:
            return []
        payload = json.dumps({"texts": texts, "source": "en", "target": _target_lang_code})
        result = subprocess.run(
            [str(_APPLE_TRANSLATE_BIN)],
            input=payload, capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"apple_translate failed: {result.stderr}")
        return json.loads(result.stdout)["translations"]

    def close(self) -> None:
        pass


_apple_translator: AppleTranslator | None = None


def _chunk_text(text: str, tokenizer=None) -> list[str]:
    """Split text into chunks that fit within MAX_TOKENS (word-level split)."""
    tok   = tokenizer or _worker_tokenizer
    words = text.split()
    chunks, current, length = [], [], 0
    for word in words:
        wlen = len(tok.encode(word, add_special_tokens=False)) + 1
        if length + wlen > MAX_TOKENS and current:
            chunks.append(" ".join(current))
            current, length = [], 0
        current.append(word)
        length += wlen
    if current:
        chunks.append(" ".join(current))
    return chunks or [""]


BATCH_SIZE = 16   # mini-batch size; balances parallelism vs padding overhead


def _run_batch(batch: list[str], tokenizer=None, model=None) -> list[str]:
    """Run one mini-batch through the model. Uses worker globals if not passed."""
    global _total_tokens_in, _total_tokens_out, _total_infer_sec
    tok    = tokenizer or _worker_tokenizer
    mdl    = model     or _worker_model
    device = next(mdl.parameters()).device
    inputs  = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_TOKENS)
    inputs  = {k: v.to(device) for k, v in inputs.items()}
    vi_id   = tok.convert_tokens_to_ids(_target_lang_token)
    t0      = time.perf_counter()
    outputs = mdl.generate(**inputs, forced_bos_token_id=vi_id)
    _total_infer_sec  += time.perf_counter() - t0
    _total_tokens_in  += int(inputs["input_ids"].numel())
    _total_tokens_out += int(outputs.numel())
    return [tok.decode(o, skip_special_tokens=True) for o in outputs]


def translate_blocks_batch(texts: list[str], tokenizer=None, model=None) -> list[str]:
    """
    Translate a list of text blocks efficiently:
    1. Chunk any block that exceeds MAX_TOKENS.
    2. Sort chunks by token length (minimises padding waste within each mini-batch).
    3. Run mini-batches of BATCH_SIZE through the model.
    4. Reassemble chunks → original block order.
    """
    if not texts:
        return []

    # Step 1: expand into chunks, remember which original block each came from
    expanded: list[tuple[int, str]] = []   # (orig_idx, chunk_text)
    for idx, text in enumerate(texts):
        text = text.strip()
        if not text:
            expanded.append((idx, ""))
        else:
            for chunk in _chunk_text(text, tokenizer):
                expanded.append((idx, chunk))

    # Step 2: sort by approximate token length to reduce padding waste
    expanded.sort(key=lambda x: len(x[1]))

    # Step 3: mini-batch inference
    translated_sorted: list[str] = []
    for i in range(0, len(expanded), BATCH_SIZE):
        mini = [t for _, t in expanded[i:i + BATCH_SIZE]]
        translated_sorted.extend(_run_batch(mini, tokenizer, model))

    # Step 4: reassemble — collect chunk translations per original block
    result: list[list[str]] = [[] for _ in texts]
    for (orig_idx, _), vi_chunk in zip(expanded, translated_sorted):
        result[orig_idx].append(vi_chunk)

    return [" ".join(parts) for parts in result]


# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

def render_page(page: fitz.Page, dpi: int) -> Image.Image:
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


# ---------------------------------------------------------------------------
# Text block extraction — returns list of dicts with text + pixel bbox
# ---------------------------------------------------------------------------

def native_blocks(page: fitz.Page, dpi: int) -> list[dict]:
    """
    Extract native text blocks, splitting each into paragraphs where a
    vertical gap between lines exceeds 0.5× the block's average line height.
    Each paragraph becomes its own block with its own pixel bbox.
    """
    scale  = dpi / 72
    result = []

    for block in page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", []):
        if block["type"] != 0:
            continue
        lines = block.get("lines", [])
        if not lines:
            continue

        # Column width for short-line detection
        block_w = block["bbox"][2] - block["bbox"][0]

        # Average line height for vertical-gap detection
        heights    = [l["bbox"][3] - l["bbox"][1] for l in lines]
        avg_line_h = sum(heights) / len(heights) if heights else 12

        def _line_text(line) -> str:
            return " ".join(
                s["text"].strip() for s in line.get("spans", []) if s["text"].strip()
            )

        def _is_para_end(line) -> bool:
            """True if this line looks like the end of a paragraph."""
            txt      = _line_text(line).rstrip()
            if not txt:
                return False
            # Short line: doesn't reach 70% of column width
            line_w   = line["bbox"][2] - line["bbox"][0]
            is_short = line_w < block_w * 0.70
            # Ends with sentence-closing punctuation
            ends_sent = txt[-1] in ".!?:\"'"
            return is_short and ends_sent

        # Group lines into paragraphs
        paragraphs: list[list] = []
        current:    list       = []
        prev_bottom: float | None = None

        for i, line in enumerate(lines):
            top = line["bbox"][1]
            # Split on large vertical gap
            if prev_bottom is not None and (top - prev_bottom) > avg_line_h * 0.5:
                if current:
                    paragraphs.append(current)
                    current = []
            current.append(line)
            prev_bottom = line["bbox"][3]
            # Split after a short sentence-ending line
            if _is_para_end(line):
                paragraphs.append(current)
                current = []
                prev_bottom = None
        if current:
            paragraphs.append(current)

        # Each paragraph → one block entry
        for para in paragraphs:
            words, max_size = [], 0.0
            for line in para:
                for span in line.get("spans", []):
                    t = span["text"].strip()
                    if t:
                        words.append(t)
                    max_size = max(max_size, span["size"])
            text = " ".join(words).strip()
            if not text:
                continue
            px0 = min(l["bbox"][0] for l in para) * scale
            py0 = para[0]["bbox"][1]               * scale
            px1 = max(l["bbox"][2] for l in para) * scale
            py1 = para[-1]["bbox"][3]              * scale
            result.append({
                "text":      text,
                "px_bbox":   (px0, py0, px1, py1),
                "font_size": max_size * scale,
            })

    return result


def ocr_blocks(img: Image.Image, dpi: int) -> list[dict]:
    """Tesseract block-level OCR with pixel bboxes."""
    data   = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    groups: dict[tuple, dict] = {}
    for i, word in enumerate(data["text"]):
        if data["conf"][i] < 20 or not word.strip():
            continue
        key = (data["block_num"][i], data["par_num"][i])
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if key not in groups:
            groups[key] = {"words": [], "x0": x, "y0": y, "x1": x+w, "y1": y+h, "max_h": h}
        g = groups[key]
        g["words"].append(word)
        g["x0"] = min(g["x0"], x);   g["y0"] = min(g["y0"], y)
        g["x1"] = max(g["x1"], x+w); g["y1"] = max(g["y1"], y+h)
        g["max_h"] = max(g["max_h"], h)

    result = []
    for g in groups.values():
        text = " ".join(g["words"]).strip()
        if not text:
            continue
        result.append({
            "text":      text,
            "px_bbox":   (float(g["x0"]), float(g["y0"]),
                          float(g["x1"]), float(g["y1"])),
            "font_size": float(g["max_h"]),
        })
    return result


# ---------------------------------------------------------------------------
# Text rendering helpers
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if _FONT_PATH:
        try:
            return ImageFont.truetype(_FONT_PATH, max(6, size))
        except Exception:
            pass
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str,
               font, max_width: float) -> list[str]:
    """Wrap text to fit within max_width pixels."""
    words  = text.split()
    lines, current = [], []
    for word in words:
        test = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] > max_width and current:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def _draw_text_in_box(draw: ImageDraw.ImageDraw, text: str,
                      px_bbox: tuple, font_size: int) -> None:
    """
    White out px_bbox then draw text inside it, shrinking font until it fits.
    """
    x0, y0, x1, y1 = px_bbox
    if y1 < y0:
        y0, y1 = y1, y0
    box_w = x1 - x0
    box_h = y1 - y0

    # White out original text region
    draw.rectangle([x0, y0, x1, y1], fill="white")

    if not text.strip():
        return

    # Try decreasing font sizes until the wrapped text fits vertically
    for size in range(max(6, int(font_size)), 5, -1):
        font    = _get_font(size)
        lines   = _wrap_text(draw, text, font, box_w - 2)
        line_h  = size + 2
        if len(lines) * line_h <= box_h:
            y = y0 + 1
            for line in lines:
                draw.text((x0 + 1, y), line, fill="black", font=font)
                y += line_h
            return

    # Last resort: tiny font, single line truncated
    font = _get_font(6)
    draw.text((x0 + 1, y0 + 1), text[:80], fill="black", font=font)


# ---------------------------------------------------------------------------
# Build translated page image
# ---------------------------------------------------------------------------

def build_translated_image(page_img: Image.Image,
                            blocks: list[dict]) -> Image.Image:
    """
    Return a copy of page_img with each text block whited out and
    replaced by its Vietnamese translation.
    """
    translated = page_img.copy()
    draw       = ImageDraw.Draw(translated)
    for block in blocks:
        if not block.get("vi_text", "").strip():
            continue
        _draw_text_in_box(draw, block["vi_text"],
                          block["px_bbox"], int(block["font_size"]))
    return translated


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def img_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buf, format=fmt)
    b64  = base64.b64encode(buf.getvalue()).decode()
    mime = "image/jpeg" if fmt == "JPEG" else f"image/{fmt.lower()}"
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: sans-serif;
      max-width: 980px;
      margin: 40px auto;
      padding: 0 24px;
      background: #f5f5f3;
      color: #1a1a1a;
    }}
    h1  {{ font-size: 1.2em; border-bottom: 2px solid #555; padding-bottom: 8px; }}
    .meta {{ font-size: 0.8em; color: #777; margin-bottom: 12px; }}
    .nav  {{ font-size: 0.85em; margin-bottom: 28px; }}
    .nav a {{ margin-right: 14px; color: #336; }}
    .label {{
      font-size: 0.7em; font-weight: bold; letter-spacing: 0.12em;
      text-transform: uppercase; color: #999;
      margin: 28px 0 8px;
    }}
    img.page {{
      width: 100%;
      border: 1px solid #ccc;
      border-radius: 4px;
      display: block;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
  <div class="nav">
    <a href="index.html">↑ Issue index</a>
    {prev_link}
    {next_link}
  </div>

  <div class="label">Original (English)</div>
  <img class="page" src="{original_src}" alt="Original page">

  <div class="label">Translated (Tiếng Việt)</div>
  <img class="page" src="{translated_src}" alt="Translated page">
</body>
</html>
"""


def build_page_html(original_img: Image.Image, translated_img: Image.Image,
                    paper: str, issue: str, page_num: int, page_count: int) -> str:
    n     = page_num + 1
    title = f"{paper.replace('_', ' ')} — Page {n}"
    meta  = (f"{paper.replace('_', ' ')} &mdash; "
             f"{issue.replace('_', ' ')} &mdash; Page {n} of {page_count}")
    prev_link = (f'<a href="page_{n-1:02d}.html">&larr; Page {n-1}</a>'
                 if page_num > 0 else "")
    next_link = (f'<a href="page_{n+1:02d}.html">Page {n+1} &rarr;</a>'
                 if page_num < page_count - 1 else "")
    return _PAGE_TEMPLATE.format(
        title=title, meta=meta,
        prev_link=prev_link, next_link=next_link,
        original_src=img_to_data_url(original_img),
        translated_src=img_to_data_url(translated_img),
    )


# ---------------------------------------------------------------------------
# Issue index
# ---------------------------------------------------------------------------

def build_index(issue: str, paper: str, page_count: int) -> str:
    rows = "\n".join(
        f'    <li><a href="page_{n:02d}.html">Page {n}</a></li>'
        for n in range(1, page_count + 1)
    )
    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{issue.replace("_", " ")} — {paper.replace("_", " ")}</title>
  <style>
    body {{ font-family: sans-serif; max-width: 700px; margin: 40px auto; padding: 0 20px; }}
    h1 {{ font-size: 1.4em; }}
    ul {{ line-height: 2.2; column-count: 3; }}
  </style>
</head>
<body>
  <h1>{paper.replace("_", " ")}</h1>
  <p>{issue.replace("_", " ")}</p>
  <ul>
{rows}
  </ul>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _process_page_worker(args: tuple) -> tuple[int, bytes, bytes]:
    """
    Worker function — runs in a child process that already has the model loaded.
    Returns (page_num, original_jpeg_bytes, translated_jpeg_bytes).
    Images are serialised as JPEG bytes so they can cross the process boundary.
    """
    page_num, pdf_path_str, page_count, dpi = args
    doc      = fitz.open(pdf_path_str)
    page     = doc[page_num]
    page_img = render_page(page, dpi)
    all_text = page.get_text("text")
    if len(all_text.strip()) < MIN_NATIVE_TEXT:
        blocks = ocr_blocks(page_img, dpi)
        method = "OCR"
    else:
        blocks = native_blocks(page, dpi)
        method = f"{len(blocks)}blk"
    doc.close()

    vi_texts = translate_blocks_batch([b["text"] for b in blocks])
    for block, vi in zip(blocks, vi_texts):
        block["vi_text"] = vi

    translated_img = build_translated_image(page_img, blocks)
    print(f"  page_{page_num+1:02d}/{page_count} [{method}] ✓", flush=True)

    def to_bytes(img):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return buf.getvalue()

    return page_num, to_bytes(page_img), to_bytes(translated_img)


def process_pdf(pdf_path: Path, output_dir: Path, dpi: int,
                max_pages: int | None = None,
                flat_dir: Path | None = None) -> None:
    paper = pdf_path.parent.name
    issue = pdf_path.stem
    print(f"\n[{paper}] {issue}")

    output_dir.mkdir(parents=True, exist_ok=True)
    doc        = fitz.open(str(pdf_path))
    page_count = min(len(doc), max_pages) if max_pages else len(doc)

    for page_num in range(page_count):
        page     = doc[page_num]
        page_img = render_page(page, dpi)
        all_text = page.get_text("text")

        if len(all_text.strip()) < MIN_NATIVE_TEXT:
            blocks = ocr_blocks(page_img, dpi)
            method = "OCR"
        else:
            blocks = native_blocks(page, dpi)
            method = f"{len(blocks)}blk"

        t_page_start = time.perf_counter()
        print(f"  page_{page_num+1:02d}/{page_count} [{method}]", end="", flush=True)
        if _backend == "apple":
            vi_texts = _apple_translator.translate([b["text"] for b in blocks])
        else:
            vi_texts = translate_blocks_batch([b["text"] for b in blocks])
        for block, vi in zip(blocks, vi_texts):
            block["vi_text"] = vi

        translated_img = build_translated_image(page_img, blocks)
        label = f"page_{page_num + 1:02d}"
        html  = build_page_html(page_img, translated_img, paper, issue, page_num, page_count)
        (output_dir / f"{label}.html").write_text(html, encoding="utf-8")

        if flat_dir is not None:
            flat_issue_dir = flat_dir / paper / issue
            flat_issue_dir.mkdir(parents=True, exist_ok=True)
            translated_img.save(flat_issue_dir / f"{label}.jpg", format="JPEG", quality=85)

        page_sec = time.perf_counter() - t_page_start
        if _backend == "nllb":
            tok_in_s  = _total_tokens_in  / _total_infer_sec if _total_infer_sec else 0
            tok_out_s = _total_tokens_out / _total_infer_sec if _total_infer_sec else 0
            print(f" ✓  {page_sec:.1f}s  [{tok_in_s:.0f} tok_in/s  {tok_out_s:.0f} tok_out/s]")
        else:
            print(f" ✓  {page_sec:.1f}s")

    doc.close()
    (output_dir / "index.html").write_text(
        build_index(issue, paper, page_count), encoding="utf-8"
    )
    print(f"  → {page_count} pages → {output_dir}")
    if _backend == "nllb":
        tok_in_s  = _total_tokens_in  / _total_infer_sec if _total_infer_sec else 0
        tok_out_s = _total_tokens_out / _total_infer_sec if _total_infer_sec else 0
        print(f"  tokens — in: {_total_tokens_in:,}  out: {_total_tokens_out:,}  "
              f"infer: {_total_infer_sec:.1f}s  "
              f"[{tok_in_s:.0f} tok_in/s  {tok_out_s:.0f} tok_out/s]")


def process_all(dpi: int, max_pages: int | None = None) -> None:
    if not NEWS_DIR.exists():
        sys.exit(f"Error: '{NEWS_DIR}' directory not found")
    EXTRACTS_DIR.mkdir(exist_ok=True)
    flat_dir = EXTRACTS_DIR / "flat"
    flat_dir.mkdir(exist_ok=True)
    pdfs = sorted(NEWS_DIR.glob("**/*.pdf"))
    if not pdfs:
        sys.exit(f"No PDF files found under '{NEWS_DIR}'")
    for pdf_path in pdfs:
        issue_dir = EXTRACTS_DIR / pdf_path.parent.name / pdf_path.stem
        process_pdf(pdf_path, issue_dir, dpi, max_pages=max_pages, flat_dir=flat_dir)
    print(f"\nFlat folder → {flat_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Translate PDF pages with in-place text overlay"
    )
    parser.add_argument("pdfs", nargs="*",
                        help="PDF file(s) to process (default: all under pdfs/)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"Render resolution (default: {DEFAULT_DPI})")
    parser.add_argument("--pages", type=int, default=None,
                        help="Only process the first N pages (default: all)")
    parser.add_argument("--lang", default="vi",
                        help="Target language: short code (vi, fr, de, es, zh, ja, ko, ar, "
                             "hi, pt) or full NLLB token e.g. vie_Latn (default: vi)")
    parser.add_argument("--backend", choices=["nllb", "apple"], default="nllb",
                        help="Translation backend: nllb (default, NLLB-200 1.3B via MPS) or "
                             "apple (Foundation Models, requires macOS 26+)")
    args = parser.parse_args()

    global _worker_tokenizer, _worker_model, _target_lang_token, \
           _target_lang_name, _target_lang_code, _backend, _apple_translator
    _target_lang_token = _LANG_MAP.get(args.lang, args.lang)
    _target_lang_name  = _LANG_NAMES.get(args.lang, args.lang)
    _target_lang_code  = args.lang
    _backend           = args.backend

    print(f"Target language : {args.lang} ({_target_lang_name})", flush=True)
    print(f"Backend         : {_backend}", flush=True)

    if _backend == "nllb":
        # Load model — use MPS (Apple GPU) if available for fast inference
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading {MODEL_NAME} on {device.upper()} …", flush=True)
        _worker_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="eng_Latn")
        _worker_model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        _worker_model.eval()
        print("Model ready.", flush=True)
    else:
        _apple_translator = AppleTranslator()
        print("Apple translator ready.", flush=True)

    try:
        if args.pdfs:
            EXTRACTS_DIR.mkdir(exist_ok=True)
            flat_dir = EXTRACTS_DIR / "flat"
            flat_dir.mkdir(exist_ok=True)
            for path_str in args.pdfs:
                pdf_path = Path(path_str)
                if not pdf_path.exists():
                    print(f"Warning: {pdf_path} not found, skipping")
                    continue
                issue_dir = EXTRACTS_DIR / pdf_path.parent.name / pdf_path.stem
                process_pdf(pdf_path, issue_dir, args.dpi, max_pages=args.pages,
                            flat_dir=flat_dir)
            print(f"\nFlat folder → {flat_dir}")
        else:
            process_all(args.dpi, args.pages)
    finally:
        if _apple_translator is not None:
            _apple_translator.close()


if __name__ == "__main__":
    main()
