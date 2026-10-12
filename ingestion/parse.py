import hashlib
import json
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
MAX_CHARS = 1800

HEADING = re.compile(r"^(?:\d{1,2}\[)?(\d{1,3}[A-Z]{0,3})\.\s+([A-Z“\"][^—–─]{2,220}?)\s*(?:[.:]?\s*[—–─]|\.\s?-)")
DIVISION = re.compile(r"^(?:\d{0,2}\[\s*(?:\]\s*\d{0,2}\[)?)?(?i:division)\s+([IVXLC]+[A-Z]*)\b(?!.*\b(?:inserted|added|substituted|omitted|of Part)\b)[^:]{0,50}$")
MARKER = re.compile(r"(\d{1,2})\[")
YEAR = re.compile(r"(?:Act|Ordinance),?\s*\(?[^\d]{0,30}?(\d{4})")
OLD_TEXT = re.compile(r"read as (?:follows|under)\s*[:\-–—]*\s*(.+)", re.S)


def load_sources():
    return json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))


def scrub(line):
    line = re.sub(r"(?<=[a-z])\.(?=[a-z])|(?<=[A-Z])(?:\.|\s\.\s)(?=[A-Z])", "", line)
    return re.sub(r"(?:\s+\.)+$", "", line).strip()


def split_page(text):
    lines = [scrub(l) for l in text.split("\n")]
    header = re.sub(r"_+.*", "", lines[0]).strip()
    lines = [l for l in lines[1:] if l and l != "."]
    if lines and lines[-1].isdigit():
        lines.pop()
    body = "\n".join(lines)
    for i, line in enumerate(lines):
        rest = " ".join(lines[i + 1:])
        if re.fullmatch(r"\d{1,2}", line) and f"{line}[" in body and re.search(r"\b(?:Act|Ordinance)\b", rest):
            return header, lines[:i], parse_notes(lines[i:])
    return header, lines, {}


def parse_notes(lines):
    notes, current = {}, None
    for line in lines:
        if re.fullmatch(r"\d{1,2}", line):
            current = line
            notes[current] = ""
        elif current:
            notes[current] = f"{notes[current]} {line}".strip()
    return notes


def summary(note):
    return re.split(r"\.\s+The\s[^.]{0,60}?read as", note)[0].rstrip(".") + "."


def amended_on(note):
    year = YEAR.search(note)
    return f"{year.group(1)}-07-01" if year else None


def joiner(previous, line):
    if previous.endswith((":", ";", "—", "–")) or re.match(r"^(?:\[|\d{1,2}\[)?(?:\(|Provided|Explanation|Illustration|\d+\.)", line):
        return "\n"
    return " "


def sections(source):
    with pdfplumber.open(ROOT / source["file"]) as pdf:
        pages = [p.extract_text() or "" for p in pdf.pages]
    current, last_number, chapter, division = None, 0, "", ""
    for page_no, text in enumerate(pages[source["body_starts_at_page"] - 1:], source["body_starts_at_page"]):
        header, lines, notes = split_page(text)
        in_schedule = "Schedule" in header
        for i, line in enumerate(lines):
            label = None
            heading = None if in_schedule else HEADING.match(" ".join(lines[i:i + 3]))
            if heading and last_number <= int(re.match(r"\d+", heading.group(1)).group()) <= last_number + 20:
                last_number = int(re.match(r"\d+", heading.group(1)).group())
                title = " ".join(re.sub(r"\d{1,2}\[|\]", "", heading.group(2)).split()).strip(" .")
                label = f"Section {heading.group(1)}: {title}"
            elif in_schedule and (header != chapter or DIVISION.match(line)):
                division = f"Division {DIVISION.match(line).group(1)}" if DIVISION.match(line) else ""
                label = ", ".join(filter(None, [header, division]))
            if label or current is None:
                if current:
                    yield current
                current = {"section": label or header, "chapter": header, "page": page_no, "text": "", "notes": []}
            chapter = header
            for number in MARKER.findall(line):
                if number in notes:
                    current["notes"].append((len(current["text"]), notes[number]))
            clean = MARKER.sub("[", line)
            current["text"] += (joiner(current["text"], clean) if current["text"] else "") + clean
    if current:
        yield current


def pieces(text):
    chunk = ""
    for line in text.split("\n"):
        if chunk and len(chunk) + len(line) > MAX_CHARS:
            yield chunk
            chunk = ""
        while len(line) > MAX_CHARS:
            cut = line.rfind(" ", 0, MAX_CHARS) + 1 or MAX_CHARS
            yield line[:cut].strip()
            line = line[cut:].strip()
        chunk = f"{chunk}\n{line}" if chunk else line
    if chunk:
        yield chunk


def chunk_id(*parts):
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]


def record(source, section, text, notes, effective_to=None):
    return {
        "id": chunk_id(source["id"], section["section"], text, str(effective_to)),
        "source": source["id"],
        "title": source["title"],
        "section": section["section"],
        "page": section["page"],
        "content": text,
        "amendments": list(dict.fromkeys(summary(n) for n in notes)),
        "effective_from": source["effective_from"],
        "effective_to": effective_to,
    }


def chunks(source):
    for section in sections(source):
        offset = 0
        for text in pieces(section["text"]):
            start = section["text"].find(text[:80], offset)
            end = start + len(text)
            offset = max(offset, start)
            notes = [n for at, n in section["notes"] if start <= at <= end]
            if len(re.findall(r"[A-Za-z]", text)) > 40:
                yield record(source, section, text, notes)
        for _, note in section["notes"]:
            old, until = OLD_TEXT.search(note), amended_on(note)
            if old and until and len(old.group(1)) > 60:
                for text in pieces(MARKER.sub("[", old.group(1)).strip(" “”\"-–:.")):
                    yield record(source, {**section, "section": f"{section['section']} (superseded)"}, text, [note], until)


def main(ids):
    OUT.mkdir(parents=True, exist_ok=True)
    for source in load_sources():
        if ids and source["id"] not in ids:
            continue
        rows = list({c["id"]: c for c in chunks(source)}.values())
        (OUT / f"{source['id']}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{source['id']}: {len(rows)} chunks, {len({r['section'] for r in rows})} sections, "
              f"{sum(1 for r in rows if r['effective_to'])} superseded")


if __name__ == "__main__":
    main(sys.argv[1:])
