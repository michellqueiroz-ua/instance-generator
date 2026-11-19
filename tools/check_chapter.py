#!/usr/bin/env python
"""Compile the chapter in a temporary copy and report the build.

Replaces the manual steps: copy the chapter, run latexmk, read the log for
errors, undefined references and citations, and count the page number and the
TODO markers per section.

The chapter folder is only read. The build happens in a temporary copy that is
deleted at the end. Standard library only; needs latexmk (and pdfinfo if
installed, otherwise the page count is read from the log).

Exit code: 1 if there is any error, undefined reference or citation, or no PDF
was produced; else 0.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Build products and things that must not be copied into the temporary folder.
SKIP_DIRS = {".git", "__pycache__"}
SKIP_SUFFIXES = {
    ".aux", ".log", ".bbl", ".blg", ".out", ".fls", ".fdb_latexmk", ".toc",
    ".lof", ".lot", ".synctex.gz", ".pdf",
}
# Marker macros counted per file in the original (the chapter text).
MARKERS = ("\\TODO{", "\\pending{")

OVERFULL_LIMIT_PT = 10.0

RE_ERROR = re.compile(r"^!")
RE_UNDEF_REF = re.compile(r"Reference [`'](?P<key>[^']*)' on page (?P<page>\S+) undefined")
RE_UNDEF_CIT = re.compile(r"Citation [`'](?P<key>[^']*)' on page (?P<page>\S+) undefined")
RE_OVERFULL = re.compile(r"Overfull \\hbox \((?P<pt>[\d.]+)pt too wide\)")
RE_PAGES_LOG = re.compile(r"Output written on .*\((?P<n>\d+) pages?")


def copy_chapter(src: Path, dst: Path) -> None:
    """Copy the chapter without .git and build products.

    PDFs are skipped everywhere except in figures/, because a figure may be a
    PDF and the build needs it. UNSURE: the brief says "without *.pdf"; keeping
    figure PDFs is my reading so that a build with PDF figures still works.
    """
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.is_dir():
            continue
        name = path.name.lower()
        suffix_hit = any(name.endswith(s) for s in SKIP_SUFFIXES)
        if suffix_hit and not (name.endswith(".pdf") and rel.parts[0] == "figures"):
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def set_todos(tex: str, on: bool) -> tuple[str, int]:
    """Set the switch in main.tex. Returns the new text and the lines changed.

    Only a line that is exactly \\showtodostrue or \\showtodosfalse is changed,
    so the comment that mentions the switch stays as it is.
    """
    wanted = "\\showtodostrue" if on else "\\showtodosfalse"
    lines = tex.split("\n")
    changed = 0
    for i, line in enumerate(lines):
        if line.strip() in ("\\showtodostrue", "\\showtodosfalse"):
            lines[i] = wanted
            changed += 1
    return "\n".join(lines), changed


def parse_log(text: str) -> dict:
    """Read a LaTeX log. Returns errors, undefined refs and citations, overfull, pages."""
    errors, refs, cits = [], [], []
    overfull_big = 0
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if RE_ERROR.match(line):
            errors.append(line.strip())
        # The part "Reference `key' on page N undefined" is short enough to
        # stay on one log line; only the tail "on input line N." can wrap.
        m = RE_UNDEF_REF.search(line)
        if m and "Warning" in line:
            refs.append(m.group("key"))
        m = RE_UNDEF_CIT.search(line)
        if m and "Warning" in line:
            cits.append(m.group("key"))
        m = RE_OVERFULL.search(line)
        if m and float(m.group("pt")) > OVERFULL_LIMIT_PT:
            overfull_big += 1
    pages = None
    for m in RE_PAGES_LOG.finditer(text):
        pages = int(m.group("n"))
    return {
        "errors": errors,
        "undefined_refs": sorted(set(refs)),
        "undefined_cits": sorted(set(cits)),
        "overfull_big": overfull_big,
        "pages_log": pages,
    }


def strip_comment(line: str) -> str:
    """Cut a LaTeX line at the first % that is not escaped."""
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def count_markers(tex: str) -> dict:
    """Count the marker macros in a text, ignoring commented lines."""
    body = "\n".join(strip_comment(l) for l in tex.splitlines())
    return {m[:-1]: body.count(m) for m in MARKERS}  # key: \TODO, \pending


def todo_table(chapter: Path) -> list[tuple[str, int, int]]:
    rows = []
    for sub in ("sections", "tables"):
        for f in sorted((chapter / sub).glob("*.tex")):
            c = count_markers(f.read_text(encoding="utf-8", errors="replace"))
            rows.append((f"{sub}/{f.name}", c["\\TODO"], c["\\pending"]))
    return rows


def pdfinfo_pages(pdf: Path) -> int | None:
    exe = shutil.which("pdfinfo")
    if not exe or not pdf.exists():
        return None
    out = subprocess.run([exe, str(pdf)], capture_output=True, text=True).stdout
    m = re.search(r"^Pages:\s+(\d+)", out, re.M)
    return int(m.group(1)) if m else None


def build(chapter: Path, todos_on: bool, jobname: str, main: str, timeout: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="check_chapter_"))
    try:
        copy_chapter(chapter, tmp)
        main_path = tmp / main
        text = main_path.read_text(encoding="utf-8")
        new, changed = set_todos(text, todos_on)
        if changed == 0:
            return {"fatal": f"no \\showtodostrue or \\showtodosfalse line found in {main}"}
        main_path.write_text(new, encoding="utf-8", newline="\n")
        latexmk = shutil.which("latexmk")
        if not latexmk:
            return {"fatal": "latexmk not found on PATH"}
        cmd = [latexmk, "-pdf", "-interaction=nonstopmode", f"-jobname={jobname}", main]
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True,
                                  errors="replace", timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            return {"fatal": f"latexmk timed out after {timeout} s"}
        log = tmp / f"{jobname}.log"
        info = parse_log(log.read_text(encoding="utf-8", errors="replace")) if log.exists() else {
            "errors": ["no log file written"], "undefined_refs": [], "undefined_cits": [],
            "overfull_big": 0, "pages_log": None}
        pdf = tmp / f"{jobname}.pdf"
        info["pages"] = pdfinfo_pages(pdf) or info["pages_log"]
        info["latexmk_rc"] = rc
        info["pdf_made"] = pdf.exists()
        return info
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def format_report(chapter: Path, todos_on: bool, info: dict, rows: list) -> str:
    out = [f"Chapter: {chapter}", f"TODO switch in the build: {'on' if todos_on else 'off'}", ""]
    if "fatal" in info:
        out.append(f"FATAL: {info['fatal']}")
        return "\n".join(out)
    out.append(f"{'Check':<28}{'Result':<12}Detail")
    out.append("-" * 70)

    def line(name, result, detail=""):
        out.append(f"{name:<28}{result:<12}{detail}")

    line("PDF produced", "yes" if info["pdf_made"] else "NO")
    line("Pages", str(info["pages"]) if info["pages"] is not None else "unknown")
    line("Errors (lines with !)", str(len(info["errors"])), "; ".join(info["errors"][:3]))
    line("Undefined references", str(len(info["undefined_refs"])), ", ".join(info["undefined_refs"]))
    line("Undefined citations", str(len(info["undefined_cits"])), ", ".join(info["undefined_cits"]))
    line(f"Overfull hbox > {OVERFULL_LIMIT_PT:g} pt", str(info["overfull_big"]), "count only")
    out.append("")
    todo_h, pend_h = "\\TODO", "\\pending"
    out.append(f"{'File':<36}{todo_h:>6}{pend_h:>10}")
    out.append("-" * 52)
    tot_t = tot_p = 0
    for name, t, p in rows:
        if t or p:
            out.append(f"{name:<36}{t:>6}{p:>10}")
        tot_t += t
        tot_p += p
    out.append(f"{'Total (files with 0 not listed)':<36}{tot_t:>6}{tot_p:>10}")
    return "\n".join(out)


def failed(info: dict) -> bool:
    if "fatal" in info:
        return True
    return bool(info["errors"] or info["undefined_refs"] or info["undefined_cits"]
                or not info["pdf_made"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chapter-dir", required=True, type=Path)
    ap.add_argument("--todos", choices=["on", "off"], default="on")
    ap.add_argument("--jobname", default="check_tmp")
    ap.add_argument("--main", default="main.tex", help="main file inside the chapter folder")
    ap.add_argument("--timeout", type=int, default=600, help="seconds for latexmk")
    a = ap.parse_args(argv)
    chapter = a.chapter_dir.resolve()
    if not (chapter / a.main).exists():
        print(f"FATAL: {chapter / a.main} not found")
        return 1
    on = a.todos == "on"
    info = build(chapter, on, a.jobname, a.main, a.timeout)
    print(format_report(chapter, on, info, todo_table(chapter)))
    return 1 if failed(info) else 0


if __name__ == "__main__":
    sys.exit(main())
