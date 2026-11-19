"""Tests for trace_numbers.py. Small fixture text only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import trace_numbers as tn  # noqa: E402


def nums(text):
    return [(h["number"], h["unit"]) for h in tn.find_numbers(text)]


def test_integers_decimals_units_percent():
    text = r"We use 5000 requests, 3.5\,km, 4\,h, 28\,km/h, 600\,s and 12.9\% of trips."
    assert nums(text) == [
        ("5000", ""), ("3.5", "km"), ("4", "h"), ("28", "km/h"), ("600", "s"), ("12.9", "%"),
    ]


def test_thousand_separator_is_one_number():
    assert nums(r"one with 10\,000 requests") == [("10000", "")]


def test_ranges_give_two_numbers():
    assert [n for n, _ in nums(r"spread 20--120\,min") ] == ["20", "120"]


def test_clock_time_is_one_number():
    assert [n for n, _ in nums("from 10:00 to 14:00")] == ["10:00", "14:00"]


def test_ignored_things():
    text = "\n".join([
        r"Section~\ref{sec:a1} and Table~\ref{tab:x2}\label{l3}",
        r"as in \citep{queiroz2025evaluation} and \citet{a2020b}",
        r"\includegraphics[width=0.48\textwidth]{figures/ISA_b90_i001.png}",
        r"\begin{tabular}{@{}p{3.6cm}p{1.4cm}@{}}",
        r"decision D29 and file t04 and \texttt{design_v2.json}",
        r"% a comment with 777",
        r"50\% is real, % but 888 is a comment",
        r"\TODO{check 1282 edges} \michellF{checked 2026-11-15, 31 numbers}",
        r"the 2024 report, from 2025 to 2035",
    ])
    assert nums(text) == [("50", "%")]


def test_years_that_are_counts_stay():
    assert [n for n, _ in nums("2000 requests and 1500 pairs")] == ["2000", "1500"]


def test_single_digits_in_math_skipped_but_not_in_text():
    text = r"$c = 1$ and $[0.1, 1.5]$ and $25 \times 25$ and 4 seats and $L = 3$\,km"
    assert nums(text) == [("0.1", ""), ("1.5", ""), ("25", ""), ("25", ""), ("4", ""), ("3", "km")]


def test_pending_body_is_kept():
    assert nums(r"\pending{69\%} of trips") == [("69", "%")]


def test_line_numbers_survive_multiline_notes():
    text = "a 11\n\\TODO{one\ntwo 99\nthree}\nb 22\n"
    hits = tn.find_numbers(text)
    assert [(h["number"], h["line"]) for h in hits] == [("11", 1), ("22", 5)]


def test_resolve_files_follows_input(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "tables").mkdir()
    (tmp_path / "sections" / "04-x.tex").write_text("\\input{tables/t04_a}\n% \\input{tables/t04_b}\n")
    (tmp_path / "tables" / "t04_a.tex").write_text("5\n")
    (tmp_path / "tables" / "t04_b.tex").write_text("6\n")
    files = tn.resolve_files(tmp_path, ["04"])
    assert [f.name for f in files] == ["04-x.tex", "t04_a.tex"]  # the commented input is skipped


REGISTRY = [
    {"section_file": "a.tex", "number": "5000", "source_file": "design.json", "command": "python -c 1", "note": ""},
    {"section_file": "a.tex", "number": "3600s", "source_file": "f", "command": "UNSURE: not traced", "note": ""},
    {"section_file": "a.tex:3", "number": "4", "source_file": "f", "command": "ok", "note": ""},
    {"section_file": "a.tex", "number": "777", "source_file": "f", "command": "ok", "note": ""},
]


def test_classify():
    text = "5000 requests\n3600\\,s bound\nonly here: 4 now\n4 on a later line\nand 42\n"
    hits = {"sections/a.tex": tn.find_numbers(text)}
    mapped, untraced, unmapped, stale = tn.classify(hits, REGISTRY)
    assert [h["number"] for h in mapped] == ["5000", "4"]  # line-qualified row matches line 3 only
    assert [h["number"] for h in untraced] == ["3600"]
    assert [(h["number"], h["line"]) for h in unmapped] == [("4", 4), ("42", 5)]
    assert [r["number"] for r in stale] == ["777"]


def test_exit_codes(tmp_path):
    (tmp_path / "sections").mkdir()
    (tmp_path / "sections" / "04-a.tex").write_text("We have 5000 requests.\n")
    reg = tmp_path / "reg.csv"
    reg.write_text("section_file,number,source_file,command,note\n"
                   "04-a.tex,5000,design.json,python -c 1,\n")
    args = ["--chapter-dir", str(tmp_path), "--sections", "04", "--registry", str(reg)]
    assert tn.main(args) == 0
    reg.write_text("section_file,number,source_file,command,note\n"
                   "04-a.tex,5000,design.json,UNSURE: not traced,\n")
    assert tn.main(args) == 0  # untraced is mapped, unless --strict
    assert tn.main(args + ["--strict"]) == 1
    reg.write_text("section_file,number,source_file,command,note\n")
    assert tn.main(args) == 1
    assert tn.main(args + ["--report-only"]) == 0
