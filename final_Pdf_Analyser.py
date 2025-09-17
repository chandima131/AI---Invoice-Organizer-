# %%
##########################################################
# Invoices → Text (native or OCR) → GPT-5-nano → DataFrame → CSV
##########################################################
import os
import re
import json
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
from openai import OpenAI
import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# =========================
# --- Configuration ---
# =========================

INPUT_PATH = Path("./src/invoices")
OUT_CSV    = Path("./invoices_out.csv")
MAX_CHARS_PER_DOC = 12000
USE_GPT = True

POPPLER_PATH = os.environ.get("POPPLER_PATH") or None
TESSERACT_PATH = os.environ.get("TESSERACT_PATH")
if TESSERACT_PATH:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

API_KEY = os.environ.get("OPEN_AI_API_KEY")
client = OpenAI(api_key= API_KEY)

# =========================
# --- Helpers ---
# =========================
def _clean(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\x00", " ")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def extract_text_native(pdf_path: Path) -> str:
    # Try pdfplumber, then pypdf
    try:
        out = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                out.append(page.extract_text() or "")
        text = "\n".join(out)
        if text.strip():
            return _clean(text)
    except Exception:
        pass

    try:
        reader = PdfReader(str(pdf_path))
        pages = [(p.extract_text() or "") for p in reader.pages]
        return _clean("\n".join(pages))
    except Exception:
        return ""

def ocr_pdf(pdf_path: Path) -> str:
    images: List[Image.Image] = convert_from_path(
        str(pdf_path), dpi=300, poppler_path=POPPLER_PATH
    )
    texts = []
    for img in images:
        txt = pytesseract.image_to_string(img, lang="eng")
        texts.append(txt or "")
    return _clean("\n".join(texts))

def extract_text_with_ocr_fallback(pdf_path: Path) -> str:
    native = extract_text_native(pdf_path)
    if len(native) < 50:  # likely scanned
        return ocr_pdf(pdf_path)
    return native

def prompt_gpt_extract_fields(text: str) -> Dict:
    text = text[:MAX_CHARS_PER_DOC]
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
  "total_cost": number | null
}}

Text:
\"\"\"{text}\"\"\"
"""
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

_money_re = re.compile(r"[-+]?\d[\d,]*\.?\d*")

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
    return out

def heuristic_extract(text: str) -> Dict:
    inv = re.search(r"(Invoice\s*(No|#|Number)[:\s]*)([A-Za-z0-9\-\/]+)", text, re.I)
    date = re.search(r"(Date|Invoice Date)[:\s]*([0-9]{1,2}[/\-][0-9]{1,2}[/\-][0-9]{2,4})", text, re.I)
    comp = re.search(r"(Company|Supplier|Vendor)[:\s]*(.+)", text, re.I)
    subtotal = re.search(r"(Subtotal|Sub total|Net)[:\s]*([£$\€]?\s*[-+]?\d[\d,]*\.?\d*)", text, re.I)
    vat_pct = re.search(r"VAT\s*([0-9]{1,2}(\.\d+)?)\s*%", text, re.I)
    vat_amt = re.search(r"(VAT|Tax)[:\s]*([£$\€]?\s*[-+]?\d[\d,]*\.?\d*)", text, re.I)
    total = subtotal + vat_amt 

    return normalize_record({
        "invoice_no": inv.group(3) if inv else None,
        "invoice_date": date.group(2) if date else None,
        "company": comp and comp.group(2).strip(),
        "subtotal": subtotal and subtotal.group(2),
        "vat_percentage": float(vat_pct.group(1)) if vat_pct else 0,
        "vat_amount": vat_amt and vat_amt.group(2),
        "total_cost": total and total.group(2),
    })

def iter_pdfs(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        yield input_path
    elif input_path.is_dir():
        yield from sorted(input_path.glob("*.pdf"))
    else:
        return

# =========================
# --- Main runner ---
# =========================
def process_all_pdfs(input_path: Path) -> pd.DataFrame:
    rows = []
    for pdf_path in iter_pdfs(input_path):
        try:
            text = extract_text_with_ocr_fallback(pdf_path)
            if not text:
                print(f"[WARN] No text extracted: {pdf_path.name}")
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
                if USE_GPT:
                    data = prompt_gpt_extract_fields(text)
                    data = normalize_record(data)
                else:
                    data = heuristic_extract(text)

            # filename first
            row = {
                "file_name": pdf_path.name,  # <-- first column
                **data
            }
            rows.append(row)
            print(f"[OK] Processed: {pdf_path.name}")
        except Exception as e:
            print(f"[ERROR] {pdf_path.name}: {e}")
            rows.append({
                "file_name": pdf_path.name,
                "invoice_no": None,
                "invoice_date": None,
                "company": None,
                "subtotal": None,
                "vat_percentage": 0,
                "vat_amount": None,
                "total_cost": None,
            })

    df = pd.DataFrame(rows, columns=[
        "file_name",              # first
        "invoice_no", "invoice_date", "company",
        "subtotal", "vat_percentage", "vat_amount", "total_cost"
    ])
    return df

if __name__ == "__main__":
    df = process_all_pdfs(INPUT_PATH)
    print(df)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV.resolve()}")

# %%
