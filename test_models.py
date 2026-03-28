#!/usr/bin/env python3
"""
Quick test: try different model/device combinations for EN→VI translation.
Run each as a separate background task to find what works best.
"""
import time
import torch

SAMPLE = (
    "The Bank of Canada held its key interest rate steady on Wednesday, "
    "citing uncertainty over U.S. tariff policy as a key reason for pausing "
    "after a string of cuts. Governor Tiff Macklem said the central bank "
    "needed more time to assess the economic impact of potential trade disruptions."
)

def test(label, load_fn, translate_fn):
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"{'='*60}")
    try:
        t0 = time.time()
        tok, model = load_fn()
        print(f"  Load time:  {time.time()-t0:.1f}s")
        t1 = time.time()
        result = translate_fn(tok, model, SAMPLE)
        print(f"  Infer time: {time.time()-t1:.1f}s")
        print(f"  Output: {result[:200]}")
        print(f"  PASS ✓")
    except Exception as e:
        print(f"  FAIL ✗ — {e}")

# ── 1. MarianMT on MPS ────────────────────────────────────────────────────
def load_marian_mps():
    from transformers import MarianMTModel, MarianTokenizer
    tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-vi")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-vi").to("mps")
    model.eval()
    return tok, model

def translate_marian(tok, model, text):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512).to("mps")
    out = model.generate(**inputs)
    return tok.decode(out[0], skip_special_tokens=True)

# ── 2. VinAI v2 on MPS with float16 ──────────────────────────────────────
def load_vinai_mps_f16():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tok = MBart50TokenizerFast.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained(
        "vinai/vinai-translate-en2vi-v2", torch_dtype=torch.float16
    ).to("mps")
    model.eval()
    return tok, model

def translate_vinai(tok, model, text):
    device = next(model.parameters()).device
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, forced_bos_token_id=tok.lang_code_to_id["vi_VN"])
    return tok.decode(out[0], skip_special_tokens=True)

# ── 3. VinAI v2 on CPU with torch.compile ────────────────────────────────
def load_vinai_compile():
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    tok = MBart50TokenizerFast.from_pretrained("vinai/vinai-translate-en2vi-v2", src_lang="en_XX")
    model = MBartForConditionalGeneration.from_pretrained("vinai/vinai-translate-en2vi-v2")
    model.eval()
    model = torch.compile(model)
    return tok, model

def translate_vinai_cpu(tok, model, text):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**inputs, forced_bos_token_id=tok.lang_code_to_id["vi_VN"])
    return tok.decode(out[0], skip_special_tokens=True)

# ── 4. NLLB-200 distilled 600M on MPS (Meta, supports 200 languages) ─────
def load_nllb_mps():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                        src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to("mps")
    model.eval()
    return tok, model

def translate_nllb(tok, model, text):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512).to("mps")
    vi_id = tok.lang_code_to_id["vie_Latn"]
    out = model.generate(**inputs, forced_bos_token_id=vi_id)
    return tok.decode(out[0], skip_special_tokens=True)

# ── 5. NLLB-200 distilled 600M on CPU ────────────────────────────────────
def load_nllb_cpu():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M",
                                        src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model.eval()
    return tok, model

def translate_nllb_cpu(tok, model, text):
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=512)
    vi_id = tok.lang_code_to_id["vie_Latn"]
    out = model.generate(**inputs, forced_bos_token_id=vi_id)
    return tok.decode(out[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(f"PyTorch {torch.__version__} | MPS: {torch.backends.mps.is_available()}")

    test("1. MarianMT (Helsinki) on MPS",      load_marian_mps,    translate_marian)
    test("2. VinAI v2 on MPS float16",         load_vinai_mps_f16, translate_vinai)
    test("3. VinAI v2 on CPU + torch.compile", load_vinai_compile, translate_vinai_cpu)
    test("4. NLLB-600M on MPS",                load_nllb_mps,      translate_nllb)
    test("5. NLLB-600M on CPU",                load_nllb_cpu,      translate_nllb_cpu)

    print("\nDone.")
