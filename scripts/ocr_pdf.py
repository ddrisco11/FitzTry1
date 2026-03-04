"""OCR a scanned PDF to plain text using pdf2image + pytesseract."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from pdf2image import convert_from_path
import pytesseract


def ocr_pdf(pdf_path: str, out_path: str, dpi: int = 300) -> None:
    pdf = Path(pdf_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Converting PDF pages to images at {dpi} DPI...")
    pages = convert_from_path(str(pdf), dpi=dpi)
    n = len(pages)
    print(f"Total pages: {n}")

    texts = []
    t0 = time.time()
    for i, img in enumerate(pages, 1):
        text = pytesseract.image_to_string(img, lang="eng")
        texts.append(text)
        elapsed = time.time() - t0
        rate = i / elapsed
        eta = (n - i) / rate if rate > 0 else 0
        print(f"  Page {i:3d}/{n}  ETA {eta:.0f}s", end="\r", flush=True)

    print()
    combined = "\n\n".join(texts)
    out.write_text(combined, encoding="utf-8")
    print(f"Written {len(combined):,} chars to {out}")


if __name__ == "__main__":
    pdf_in = sys.argv[1] if len(sys.argv) > 1 else "corpus/raw/service-gdc-gdclccn-02-02-99-50-02029950-02029950.pdf"
    txt_out = sys.argv[2] if len(sys.argv) > 2 else "corpus/raw/amazon_madeira_rivers.txt"
    ocr_pdf(pdf_in, txt_out)
