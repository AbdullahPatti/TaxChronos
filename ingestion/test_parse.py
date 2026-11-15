import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ingestion.parse import HEADING, amended_on, card_sections, clean_transcript, highlight, joiner, split_inline_notes, table_head


def test_wrapped_body_line_is_not_a_footnote():
    lines = ["232[of section", "2 of this Act and the benchmarks", "7. Determination of tax.– (1)", "232 Words inserted by Finance Act, 2008."]
    body, notes = split_inline_notes(lines, False)
    assert body == lines[:3]
    assert notes == {"232": "Words inserted by Finance Act, 2008."}


def test_note_without_space_or_marker():
    body, notes = split_inline_notes(["Body text.", "12Inserted by the Punjab Finance Act 2015."], False)
    assert body == ["Body text."] and "12" in notes


def test_heading_variants():
    for line, number, title in [
        ("7. Determination of tax liability. – (1) Subject", "7", "Determination of tax liability"),
        ("403[26. 404[***] Return.– (1) Every", "26", "Return"),
        ("336[14AB.Discontinuance of connections.– (1)", "14AB", "Discontinuance of connections"),
        ("470[32A. 471[Audit by Panels]. 472[(1) The Board", "32A", "Audit by Panels]"),
        ("669[52A. e-intermediaries to be appointed.– (1)", "52A", "e-intermediaries to be appointed"),
        ("155. 12[Rent of property].— (1)", "155", "Rent of property]"),
    ]:
        match = HEADING.match(line)
        assert match and match.group(1) == number and match.group(2) == title, line


def test_sro_amendment_uses_exact_date():
    note = "The word substituted by the SRO 1145(I)/2016 dated 7th December, 2016."
    assert amended_on(note, False) == "2016-12-07"
    assert amended_on("Substituted by Finance Act, 2008.", False) == "2008-07-01"


def test_white_box_is_not_table_ruling():
    box = {"object_type": "rect", "fill": True, "stroke": False}
    assert highlight({**box, "non_stroking_color": (1, 1, 1)})
    assert not highlight({**box, "non_stroking_color": (0,)})


def test_table_rows_stay_on_their_own_lines():
    assert joiner("| a | b |", "| c | d |") == "\n"
    assert joiner("intro text", "| a | b |") == "\n"
    assert table_head("text\n| head |\n| row |") == "| head |"
    assert clean_transcript("| a |\n| :--- |\n\n| b |") == "| a |\n| b |"


def test_card_rows_inherit_merged_section():
    page = "| Section | A | Rate |\n| 148 Imports | x | 1% |\n| | y | 2% |\n| 149 Salary | z | 3% |"
    found = list(card_sections({}, [page]))
    assert [s["section"] for s in found] == ["Section 148 Imports", "Section 149 Salary"]
    assert found[0]["text"] == "| Section | A | Rate |\n| 148 Imports | x | 1% |\n| | y | 2% |"


if __name__ == "__main__":
    for name, check in list(globals().items()):
        if name.startswith("test_"):
            check()
            print("ok", name)
