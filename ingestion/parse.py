import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "data" / "processed"
TRANSCRIPTS = ROOT / "data" / "transcripts"
MAX_CHARS = 1800

HEADING = re.compile(r"^(?:\d{1,4}\[\s?|\[)?(\d{1,3}[A-Z]{0,4})\.\s*(?:\d{1,4}\[\*+\]\s*|\d{1,4}\[)?([A-Za-z“\"][^—–─―]{2,220}?)\s*(?:[.:]?\s*[—–─―]|\.\s?-|\.\s+(?:\d{1,4}\[)?\(1\))")
DIVISION = re.compile(r"^(?:\d{0,2}\[\s*(?:\]\s*\d{0,2}\[)?)?(?i:division)\s+([IVXLC]+[A-Z]*)\b(?!.*\b(?:inserted|added|substituted|omitted|of Part)\b)[^:]{0,50}$")
MARKER = re.compile(r"(\d{1,2})\[")
LONG_MARKER = re.compile(r"(\d{1,4})\[")
SUPERSCRIPT = re.compile(r"(?<=[.,;:)”\"’])\s?(\d{1,2})(?=\s|$)")
SCHEDULE = re.compile(r"^(?:\d{1,4}\[)?(?:THE\s+)?((?:FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|ELEVENTH|TWELFTH)\s+SCHEDULE)\b")
NOTE = re.compile(r"^(\d{1,4})\s*(?=[A-Za-z“\"‘])(.*)")
NOTE_WORDS = re.compile(r"\b(?:Act|Ordinance|S\.?R\.?O|Circular|Letter|Ibid|Notification|substituted|inserted|omitted|added|renumbered)\b", re.I)
NOTE_START = re.compile(r"^\d{1,4}\s*(?:Substituted|Inserted|Omitted|Added)\b")
CIRCULAR = re.compile(r"\b(?:Circular|Ibid|Letter|Notification|Para)\b", re.I)
PARA = re.compile(r"^(\d{1,2}[A-Z]?)\.?\s+(\S.*)")
ANNEX = re.compile(r"^(?:Appendix|Annexure)[\s-]+\S+$")
SRO_DATE = re.compile(r"dated:?\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+),?\s+(\d{4})")
YEAR = re.compile(r"(?:Act|Ordinance),?\s*\(?[^\d]{0,30}?(\d{4})")
OLD_TEXT = re.compile(r"read as (?:follows|under)\s*[:\-–—]*\s*(.+)", re.S)


def load_sources():
    return json.loads((ROOT / "sources.json").read_text(encoding="utf-8"))


def scrub(line):
    line = re.sub(r"(?<=[a-z])\.(?=[a-z])|(?<=[A-Z])(?:\.|\s\.\s)(?=[A-Z])", "", line)
    return re.sub(r"(?:\s+\.)+$", "", line).strip()


def real_table(table, width):
    rows = table.extract()
    filled = sum(1 for row in rows if sum(1 for cell in row if cell and cell.strip()) >= 2)
    return filled >= 2 and table.bbox[2] - table.bbox[0] >= 0.4 * width


def highlight(obj):
    if obj["object_type"] != "rect" or obj.get("stroke") or not obj.get("fill"):
        return False
    color = obj.get("non_stroking_color")
    return all(v == 1 for v in (color if isinstance(color, (list, tuple)) else [color]))


def table_rows(table):
    rows = [[" ".join((cell or "").split()) for cell in row] for row in table.extract()]
    used = [i for i in range(len(rows[0])) if any(row[i] for row in rows)]
    return ["| " + " | ".join(row[i] for i in used) + " |" for row in rows if any(row)]


def page_text(page):
    tables = [t for t in page.filter(lambda o: not highlight(o)).find_tables() if real_table(t, page.width)]
    if not tables:
        return page.extract_text() or ""
    parts, top = [], page.bbox[1]
    for table in sorted(tables, key=lambda t: t.bbox[1]):
        if table.bbox[1] > top:
            parts.append(page.crop((page.bbox[0], top, page.bbox[2], table.bbox[1]), strict=False).extract_text() or "")
        parts.extend(table_rows(table))
        top = max(top, table.bbox[3])
    if top < page.bbox[3]:
        parts.append(page.crop((page.bbox[0], top, page.bbox[2], page.bbox[3]), strict=False).extract_text() or "")
    return "\n".join(parts)


def clean_transcript(text):
    return "\n".join(l.strip() for l in text.split("\n") if l.strip() and not re.fullmatch(r"[|\s:\-]+", l))


def transcripts(source):
    # Scanned pages were transcribed once and saved; nothing is transcribed at parse time.
    # A new PDF's image-only pages are skipped unless a transcript file is added for it.
    path = TRANSCRIPTS / f"{source['id']}.json"
    return {int(n): text for n, text in json.loads(path.read_text(encoding="utf-8")).items()} if path.exists() else {}


def read_pages(source):
    saved = transcripts(source)
    with pdfplumber.open(ROOT / source["file"]) as pdf:
        if source.get("text_only"):
            return [p.extract_text() or "" for p in pdf.pages]
        start = source["body_starts_at_page"] - 1
        included = pdf.pages[:source.get("body_ends_at_page")]
        pages = [p.extract_text() or "" for p in included]
        saved = {n: text for n, text in saved.items() if start < n <= len(pages)}
        for i, page in enumerate(included):
            if i >= start and i + 1 not in saved:
                pages[i] = page_text(page)
    for number, text in saved.items():
        layer = "" if source.get("vision") else pages[number - 1]
        pages[number - 1] = f"{layer}\n{clean_transcript(text)}".strip()
    return pages


def split_page(text, source):
    legacy = source.get("text_only")
    lines = [scrub(l) if legacy else l.strip() for l in text.split("\n")]
    if source.get("drop"):
        lines = [l for l in lines if not re.search(source["drop"], l)]
    header = ""
    if source.get("running_header", True) and lines:
        header = re.sub(r"_+.*", "", lines[0]).strip()
        lines = lines[1:]
    lines = [l for l in lines if l and l != "."]
    if lines and lines[-1].isdigit():
        lines.pop()
    if not legacy:
        return (header, *split_inline_notes(lines, source.get("layout") == "manual"))
    body = "\n".join(lines)
    for i, line in enumerate(lines):
        rest = " ".join(lines[i + 1:])
        if re.fullmatch(r"\d{1,2}", line) and f"{line}[" in body and re.search(r"\b(?:Act|Ordinance)\b", rest):
            return header, lines[:i], parse_notes(lines[i:])
    return header, lines, {}


def split_inline_notes(lines, circulars):
    markers = set(LONG_MARKER.findall("\n".join(lines)))
    for i, line in enumerate(lines):
        note = NOTE.match(line)
        if not note:
            continue
        marked = CIRCULAR.search(line) if circulars else NOTE_START.match(line) or note.group(1) in markers and NOTE_WORDS.search(" ".join(lines[i:i + 2]))
        if marked:
            return lines[:i], parse_inline_notes(lines[i:])
    return lines, {}


def parse_inline_notes(lines):
    notes, current = {}, None
    for line in lines:
        note = NOTE.match(line)
        if note and (current is None or int(current) < int(note.group(1)) <= int(current) + 3):
            current = note.group(1)
            notes[current] = note.group(2)
        elif current:
            notes[current] = f"{notes[current]} {line}".strip()
    return notes


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


def amended_on(note, legacy=True):
    sro = None if legacy or not re.search(r"S\.?\s?R\.?\s?O", note) else SRO_DATE.search(note)
    if sro:
        try:
            return datetime.strptime(" ".join(sro.groups()), "%d %B %Y").strftime("%Y-%m-%d")
        except ValueError:
            pass
    year = YEAR.search(note)
    return f"{year.group(1)}-07-01" if year else None


def joiner(previous, line):
    if line.startswith("|") or previous.endswith((":", ";", "—", "–", "|")) or re.match(r"^(?:\[|\d{1,2}\[)?(?:\(|Provided|Explanation|Illustration|\d+\.)", line):
        return "\n"
    return " "


def contents(pages):
    titles, last = {}, None
    for text in pages:
        for line in text.split("\n"):
            entry = PARA.match(line.strip())
            if entry:
                last = entry.group(1)
                titles[last] = entry.group(2).rstrip(". ")
            elif last:
                titles[last] = f"{titles[last]} {line.strip()}".rstrip(". ")
    return titles


def card_sections(source, pages):
    head, current = None, None
    for page_no, text in enumerate(pages, 1):
        for line in text.split("\n"):
            if not line.startswith("|"):
                continue
            first = line.strip().strip("|").split("|")[0].strip()
            if first == "Section":
                head = line
                continue
            label = f"Section {first}" if first else current and current["section"]
            if not current or label != current["section"]:
                if current:
                    yield current
                current = {"section": label, "chapter": "", "page": page_no, "text": head, "notes": []}
            current["text"] += "\n" + line
    yield current


def sections(source):
    pages = read_pages(source)
    layout = source.get("layout", "statute")
    if layout == "card":
        yield from card_sections(source, pages)
        return
    legacy = source.get("text_only")
    unit = source.get("unit", "Section")
    start = source["body_starts_at_page"]
    titles = contents(pages[:start - 1]) if layout == "manual" else {}
    marker = SUPERSCRIPT if layout == "manual" else MARKER if legacy else LONG_MARKER
    current, last_number, chapter, division, schedule = None, 0, "", "", False
    seen = set()
    for page_no, text in enumerate(pages[start - 1:], start):
        header, lines, notes = split_page(text, source)
        for i, line in enumerate(lines):
            label = None
            named = None if legacy else SCHEDULE.match(line)
            schedule = schedule or bool(named)
            in_schedule = schedule or ("Schedule" in header if legacy else "schedule" in header.lower())
            para = PARA.match(line) if layout == "manual" else None
            heading = None if in_schedule or layout != "statute" else HEADING.match(" ".join(lines[i:i + 3]))
            if layout == "slides":
                label = f"Slide {page_no}" if i == 0 else None
            elif layout == "manual":
                title = para and re.sub(r"^\d{1,2}(?=[A-Z])", "", para.group(2))
                if para and para.group(1) in titles and para.group(1) not in seen and title.lower()[:12] == titles[para.group(1)].lower()[:12]:
                    seen.add(para.group(1))
                    label = f"{unit} {para.group(1)}: {titles[para.group(1)]}"
                elif para and para.group(1) in titles and para.group(1) not in seen and title[0].isupper() and title.endswith(".") and len(title) < 140:
                    seen.add(para.group(1))
                    label = f"{unit} {para.group(1)}: {title.rstrip('. ')}"
                elif ANNEX.match(line):
                    label = line
            elif named:
                label = named.group(1).title()
            elif heading and last_number <= int(re.match(r"\d+", heading.group(1)).group()) <= last_number + source.get("max_section_jump", 20):
                last_number = int(re.match(r"\d+", heading.group(1)).group())
                title = " ".join(re.sub(r"\d{1,4}\[|\]", "", heading.group(2)).split()).strip(" .")
                label = f"{unit} {heading.group(1)}: {title}"
            elif in_schedule and header and (header != chapter or DIVISION.match(line)):
                division = f"Division {DIVISION.match(line).group(1)}" if DIVISION.match(line) else ""
                label = ", ".join(filter(None, [header, division]))
            if label or current is None:
                if current:
                    yield current
                current = {"section": label or header or "Preamble", "chapter": header, "page": page_no, "text": "", "notes": []}
            chapter = header
            for number in marker.findall(line):
                if number in notes:
                    current["notes"].append((len(current["text"]), notes[number]))
            clean = marker.sub(lambda m: "" if m.group(1) in notes else m.group(0), line) if layout == "manual" else marker.sub("[", line)
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


def table_head(before):
    head = None
    for line in reversed(before.split("\n")):
        if not line.startswith("|"):
            break
        head = line
    return head


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
            head = text.startswith("|") and table_head(section["text"][:start].rstrip("\n"))
            if len(re.findall(r"[A-Za-z]", text)) > 40:
                yield record(source, section, f"{head}\n{text}" if head else text, notes)
        for _, note in section["notes"]:
            old, until = OLD_TEXT.search(note), amended_on(note, source.get("text_only"))
            if old and until and until > source["effective_from"] and len(old.group(1)) > 60:
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
