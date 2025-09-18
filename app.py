import io
import os
import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, List

import streamlit as st
import pandas as pd
from openai import OpenAI

import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import traceback

# =========================
# Fixed settings (no sidebar)
# =========================
USE_GPT: bool = True                    # set False to force heuristic only
MAX_CHARS_PER_DOC: int = 12_000
OCR_DPI: int = 300
OUT_CSV_NAME: str = "invoices_out.csv"

# Allow paths via env, but not via UI
POPPLER_PATH = os.environ.get("POPPLER_PATH") or None
TESSERACT_PATH =  st.secrets.get("TESSERACT_PATH") or None

if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# OpenAI client (auto-disable GPT if key missing)
API_KEY = st.secrets.get("OPENAI_API_KEY")
client: Optional[OpenAI] = None
if USE_GPT:
    if not API_KEY:
        USE_GPT = False  # fall back to heuristic
    else:
        client = OpenAI(api_key=API_KEY)

# ========== UI CONFIG ==========
st.set_page_config(page_title="Invoice Extractor", layout="wide")
st.title("📄 AI Invoice Extractor (PDF → OCR → GPT-5 → CSV)")
st.caption(f"Settings — USE_GPT={USE_GPT}, MAX_CHARS_PER_DOC={MAX_CHARS_PER_DOC}, OCR_DPI={OCR_DPI}")

if USE_GPT is False and (st.secrets.get("OPENAI_API_KEY") or  st.secrets.get("TESSERACT_PATH")) is None:
    st.info("No OpenAI API key found. or tesseact path Running heuristic only.")

# ========= Helpers =========
_money_re = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def _clean(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def _to_number(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x)
    m = _money_re.search(s.replace("£", "").replace("$", "").replace("€", ""))
    if not m:
        return None
    return float(m.group(0).replace(",", ""))

def extract_text_native_bytes(pdf_bytes: bytes) -> str:
    # Try pdfplumber
    try:
        out = []
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        text = "\n".join(out)
        if text.strip():
            return _clean(text)
    except Exception:
        pass

    # Try pypdf
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return _clean("\n".join(pages))
    except Exception:
        return ""

def ocr_pdf_bytes(pdf_bytes: bytes, dpi: int = 300, lang: str = "eng") -> str:
    images: List[Image.Image] = convert_from_bytes(
        pdf_bytes, dpi=dpi, poppler_path=POPPLER_PATH or None
    )
    texts = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang=lang)
        texts.append(txt or "")
    return _clean("\n".join(texts))

def extract_text_with_ocr_fallback_bytes(pdf_bytes: bytes, dpi: int = 300) -> str:
    native = extract_text_native_bytes(pdf_bytes)
    if len(native.strip()) < 50:  # likely scanned
        return ocr_pdf_bytes(pdf_bytes, dpi=dpi)
    return native

def prompt_gpt_extract_fields(text: str, max_chars: int) -> Dict:
    assert client is not None, "OpenAI client not initialised"
    text = text[:max_chars]
    prompt = f"""
You are a strict JSON extractor for invoices.
From the text below, extract ONLY these fields and return VALID JSON (no extra text):

{{
  "invoice_no": string | null,
  "invoice_date": string | null,
  "company": string | null,
  "subtotal": number | null,
  "vat_percentage": number,        // if VAT missing or not stated, use 0
  "vat_amount": number | null,
  
}}

Text:
\"\"\"{text}\"\"\""""
    resp = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )
    raw = resp.choices[0].message.content.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].lstrip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "invoice_no": None,
            "invoice_date": None,
            "company": None,
            "subtotal": None,
            "vat_percentage": 0,
            "vat_amount": None,
            "total_cost": None,
            "_raw": raw
        }

def normalize_record(rec: Dict) -> Dict:
    out = {
        "invoice_no": (rec.get("invoice_no") or None),
        "invoice_date": (rec.get("invoice_date") or None),
        "company": (rec.get("company") or None),
        "subtotal": _to_number(rec.get("subtotal")),
        "vat_percentage": float(rec.get("vat_percentage", 0) or 0),
        "vat_amount": _to_number(rec.get("vat_amount")),
        "total_cost": _to_number(rec.get("total_cost")),
    }
    if out["vat_percentage"] is None:
        out["vat_percentage"] = 0.0
    # Derive missing numbers if possible
    if out["subtotal"] is not None and out["vat_amount"] is None:
        out["vat_amount"] = round(out["subtotal"] * out["vat_percentage"] / 100.0, 2)
    if out["subtotal"] is not None and out["vat_amount"] is not None and out["total_cost"] is None:
        out["total_cost"] = round(out["subtotal"] + out["vat_amount"], 2)
    return out

def heuristic_extract(text: str) -> Dict:
    inv = re.search(r"(Invoice\s*(No|#|Number)[:\s]*)([A-Za-z0-9\-\/]+)", text, re.I)
    date = re.search(r"(Date|Invoice Date)[:\s]*([0-9]{1,2}[/\-][0-9]{1,2}[/\-][0-9]{2,4})", text, re.I)
    comp = re.search(r"(Company|Supplier|Vendor)[:\s]*(.+)", text, re.I)
    subtotal_m = re.search(r"(Subtotal|Sub total|Net)[:\s]*([£$\€]?\s*[-+]?\d[\d,]*\.?\d*)", text, re.I)
    vat_pct_m = re.search(r"VAT\s*([0-9]{1,2}(\.\d+)?)\s*%", text, re.I)
    vat_amt_m = re.search(r"(VAT|Tax)[:\s]*([£$\€]?\s*[-+]?\d[\d,]*\.?\d*)", text, re.I)
    total_m   = re.search(r"(Total|Amount Due)[:\s]*([£$\€]?\s*[-+]?\d[\d,]*\.?\d*)", text, re.I)

    subtotal = _to_number(subtotal_m.group(2)) if subtotal_m else None
    vat_pct  = float(vat_pct_m.group(1)) if vat_pct_m else 0.0
    vat_amt  = _to_number(vat_amt_m.group(2)) if vat_amt_m else None
    total    = _to_number(total_m.group(2)) if total_m else None

    # compute missing pieces if possible
    if subtotal is not None and vat_amt is None:
        vat_amt = round(subtotal * vat_pct / 100.0, 2)
    if subtotal is not None and vat_amt is not None and total is None:
        total = round(subtotal + vat_amt, 2)

    return {
        "invoice_no": inv.group(3) if inv else None,
        "invoice_date": date.group(2) if date else None,
        "company": comp.group(2).strip() if comp else None,
        "subtotal": subtotal,
        "vat_percentage": vat_pct if vat_pct is not None else 0.0,
        "vat_amount": vat_amt,
        "total_cost": total,
    }

def process_uploaded_file(file_like, *, use_gpt: bool, max_chars: int, ocr_dpi: int) -> Dict:
    pdf_bytes = file_like.read()
    text = extract_text_with_ocr_fallback_bytes(pdf_bytes, dpi=ocr_dpi)
    if not text:
        data = {
            "invoice_no": None,
            "invoice_date": None,
            "company": None,
            "subtotal": None,
            "vat_percentage": 0,
            "vat_amount": None,
            "total_cost": None,
        }
    else:
        if use_gpt and client:
            data = prompt_gpt_extract_fields(text, max_chars=max_chars)
            data = normalize_record(data)
        else:
            data = heuristic_extract(text)
            data = normalize_record(data)
    return data

# ========= Main UI =========
uploaded_files = st.file_uploader(
    "Upload one or more PDF invoices", type=["pdf"], accept_multiple_files=True
)

if st.button("Process Invoices"):
    if not uploaded_files:
        st.error("Please upload at least one PDF.")
        st.stop()

    progress = st.progress(0, text="Starting…")
    status = st.empty()
    t0 = time.perf_counter()
    rows = []
    errors: List[str] = []

    total = len(uploaded_files)
    for i, uf in enumerate(uploaded_files, start=1):
        f_start = time.perf_counter()
        try:
            # make a fresh buffer so we can re-read safely
            buf = io.BytesIO(uf.getbuffer())
            row = process_uploaded_file(
                file_like=io.BytesIO(buf.getvalue()),
                use_gpt=USE_GPT,
                max_chars=MAX_CHARS_PER_DOC,
                ocr_dpi=OCR_DPI
            )
            row["file_name"] = uf.name
        except Exception as e:
            row = {
                "file_name": uf.name,
                "invoice_no": None,
                "invoice_date": None,
                "company": None,
                "subtotal": None,
                "vat_percentage": 0,
                "vat_amount": None,
                "total_cost": None,
            }
            errors.append(f"{uf.name}: {e}\n{traceback.format_exc()}")

        rows.append(row)

        elapsed = time.perf_counter() - t0
        avg = elapsed / i
        remaining = max(avg * (total - i), 0)
        progress.progress(i / total, text=f"Processing {i}/{total}…  Elapsed: {elapsed:0.1f}s  ETA: {remaining:0.1f}s")
        status.write(f"Processed: **{uf.name}** in {time.perf_counter() - f_start:0.2f}s")

    progress.progress(1.0, text="Done ✔")
    total_time = time.perf_counter() - t0
    st.success(f"Processed {total} file(s) in {total_time:0.2f}s.")

    if errors:
        with st.expander("Errors (click to expand)"):
            for e in errors:
                st.code(e)

    df = pd.DataFrame(rows, columns=[
        "file_name", "invoice_no", "invoice_date", "company",
        "subtotal", "vat_percentage", "vat_amount", "total_cost"
    ])
    st.subheader("Results")
    st.dataframe(df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=OUT_CSV_NAME,
        mime="text/csv"
    )
