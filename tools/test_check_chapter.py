"""Tests for check_chapter.py. Small fixture text only, no LaTeX run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import check_chapter as cc  # noqa: E402

LOG = r"""
This is pdfTeX
! Undefined control sequence.
l.12 \foo
LaTeX Warning: Reference `sec:missing' on page 3 undefined on input line 40.
LaTeX Warning: Reference `sec:missing' on page 4 undefined on input line 50.
Package natbib Warning: Citation `smith2020' on page 2 undefined on input line 7.
LaTeX Warning: Citation `jones2021' on page 5 undefined on input line 9.
Overfull \hbox (12.5pt too wide) in paragraph at lines 1--2
Overfull \hbox (9.99pt too wide) in paragraph at lines 3--4
Overfull \hbox (30.0pt too wide) in paragraph at lines 5--6
Output written on check_tmp.pdf (23 pages, 456789 bytes).
"""


def test_parse_log_counts():
    r = cc.parse_log(LOG)
    assert r["errors"] == ["! Undefined control sequence."]
    assert r["undefined_refs"] == ["sec:missing"]  # same key twice counts once
    assert r["undefined_cits"] == ["jones2021", "smith2020"]
    assert r["overfull_big"] == 2  # 12.5 and 30.0; 9.99 is not larger than 10
    assert r["pages_log"] == 23


def test_parse_log_clean():
    r = cc.parse_log("Output written on main.pdf (1 page, 100 bytes).\n")
    assert r["errors"] == [] and r["undefined_refs"] == [] and r["undefined_cits"] == []
    assert r["pages_log"] == 1


def test_parse_log_wrapped_warning():
    # Only the tail "on input line N." wraps in a real log; the key stays on line one.
    text = "LaTeX Warning: Reference `a:b' on page 1 undefined on input line\n2.\n"
    assert cc.parse_log(text)["undefined_refs"] == ["a:b"]


def test_set_todos_only_touches_the_switch_line():
    tex = (
        "% \\showtodosfalse gives a clean version\n"
        "\\newif\\ifshowtodos\n"
        "\\showtodostrue\n"
        "text\n"
    )
    new, n = cc.set_todos(tex, on=False)
    assert n == 1
    assert "\\showtodosfalse\ntext" in new
    assert new.startswith("% \\showtodosfalse gives a clean version\n")  # comment untouched
    back, n2 = cc.set_todos(new, on=True)
    assert n2 == 1 and "\\showtodostrue\ntext" in back


def test_set_todos_missing_switch():
    assert cc.set_todos("no switch here\n", on=True)[1] == 0


def test_count_markers_ignores_comments():
    tex = (
        "A \\TODO{one} and \\TODO{two}. \\pending{x}\n"
        "% \\TODO{commented out}\n"
        "50\\% of it \\TODO{three}\n"  # \% is an escaped percent, not a comment
    )
    c = cc.count_markers(tex)
    assert c["\\TODO"] == 3
    assert c["\\pending"] == 1


def test_copy_chapter_skips_build_products(tmp_path):
    src = tmp_path / "src"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text("x")
    (src / "sections").mkdir()
    (src / "figures").mkdir()
    (src / "main.tex").write_text("x")
    (src / "main.aux").write_text("x")
    (src / "main.pdf").write_text("x")
    (src / "main.bbl").write_text("x")
    (src / "sections" / "a.tex").write_text("x")
    (src / "figures" / "f.png").write_text("x")
    (src / "figures" / "g.pdf").write_text("x")
    dst = tmp_path / "dst"
    dst.mkdir()
    cc.copy_chapter(src, dst)
    got = sorted(str(p.relative_to(dst)).replace("\\", "/") for p in dst.rglob("*") if p.is_file())
    assert got == ["figures/f.png", "figures/g.pdf", "main.tex", "sections/a.tex"]


def test_failed_logic():
    ok = {"errors": [], "undefined_refs": [], "undefined_cits": [], "pdf_made": True}
    assert not cc.failed(ok)
    assert cc.failed({**ok, "errors": ["! x"]})
    assert cc.failed({**ok, "undefined_refs": ["a"]})
    assert cc.failed({**ok, "undefined_cits": ["a"]})
    assert cc.failed({**ok, "pdf_made": False})
    assert cc.failed({"fatal": "x"})
