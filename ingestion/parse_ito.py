import pdfplumber
import json
import re
from pathlib import Path

PDF_PATH = Path("data/raw") / "IncomeTaxOrdinance2001.pdf"
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 150

SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:SECTION\s+)?(\d+[A-Z]{0,2})\.\s+([A-Z][^\n]{3,80})', 
    re.IGNORECASE
)

def clean_text(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r' \n', '\n', text)
    text = re.sub(r'\n ', '\n', text)
    text = re.sub(r'INCOME TAX ORDINANCE, 2001\n?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+\n?', '', text)
    return text.strip()

def extract_text_from_pdf(pdf_path: Path) -> str:
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Total pages: {total_pages}")
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                full_text.append(text)
            if i % 100 == 0 or i == total_pages:
                print(f"Processed {i}/{total_pages} pages...")
    combined = "\n\n".join(full_text)
    return clean_text(combined)

def split_into_clause_chunks(text: str) -> list[dict]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    raw_chunks = []
    current_chunk = ""
    current_section = "General / Preamble"

    for para in paragraphs:
        sec_match = SECTION_PATTERN.search(para)
        if sec_match:
            sec_num = sec_match.group(1)
            sec_title = sec_match.group(2).strip()
            
            if int(sec_num) if sec_num.isdigit() else 0 < 300:
                current_section = f"Section {sec_num}: {sec_title}"

        if len(current_chunk) + len(para) + 2 <= CHUNK_SIZE:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                raw_chunks.append((current_chunk, current_section))
            
            if len(para) > CHUNK_SIZE:
                start = 0
                while start < len(para):
                    end = start + CHUNK_SIZE
                    raw_chunks.append((para[start:end].strip(), current_section))
                    start = end - CHUNK_OVERLAP
                current_chunk = ""
            else:
                current_chunk = para

    if current_chunk:
        raw_chunks.append((current_chunk, current_section))

    structured_chunks = []
    for i, (chunk_text, sec) in enumerate(raw_chunks):
        structured_chunks.append({
            "chunk_id": f"ito_{i:04d}",
            "text": chunk_text,
            "source": "ITO_2001",
            "section": sec,
            "effective_from": "2001-07-01",
            "effective_to": None,
            "char_count": len(chunk_text)
        })

    return structured_chunks

def main():
    print("Extracting text from Income Tax Ordinance 2001...")
    text = extract_text_from_pdf(PDF_PATH)
    print(f"\nTotal characters after cleaning: {len(text):,}")
    
    print("Creating clause-aware chunks...")
    chunks = split_into_clause_chunks(text)
    print(f"Total chunks created: {len(chunks)}")

    output_file = OUTPUT_DIR / "ito_2001_chunks.json"
    data = {
        "source": "Income Tax Ordinance, 2001 (as amended)",
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunks": chunks
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to: {output_file}")
    
    found_sections = set(c["section"] for c in chunks if c["section"])
    print(f"Total unique sections extracted: {len(found_sections)}")
    print("\nSample extracted sections:")
    for s in list(found_sections)[:5]:
        print(f" - {s}")

if __name__ == "__main__":
    main()