#!/usr/bin/env python
"""List the numbers in chapter sections and check each has a source.

Replaces the manual step "is every number in the text traced to a file and a
command?". The chapter folder is only read. Standard library only.

Registry: tools/number_sources.csv with columns
    section_file,number,source_file,command,note
A number is MAPPED when a row has the same file name (the section file or the
table file where the number stands) and the same number. The number cell is
the plain number, for example 3515 or 0.75; it may also carry the unit or the
percent sign (3600s, 4h, 10%) to match only that form. A row whose command
starts with "UNSURE" is counted as MAPPED-UNTRACED: there is a row, but no
command that reproduces the number yet.

Exit code 1 when a requested file has an unmapped number (unless
--report-only). --strict also fails on untraced rows.

What is not counted as a number (see strip_tex): comments, \\TODO, \\michellF,
\\colleagueF and \\springerF bodies (notes that go away), the arguments of
\\ref, \\label, \\cite*, \\input, \\includegraphics, \\texttt, \\url and of
spacing commands, column specs and widths such as p{3.6cm} or 0.48\\textwidth,
ISO dates, digits glued to a letter (D29, t04), calendar years after a word such
as in, from, to or the (but not "2000 requests"), and single digits 0 to 9
inside math that carry no unit ($c = 1$ is skipped; $L = 3$\\,km and
$25 \\times 25$ stay). A clock time such as 10:00 is
one number.

A registry row whose section_file ends in :LINE (for example
04-instance-generation.tex:28) matches only that line. Use it when one number
stands for different things in the same file. The row goes stale (is listed)
when the text moves, which is the signal to look at the number again.
"""
import argparse
import csv
import re
import sys
from pathlib import Path

# Commands whose braced argument is removed entirely.
DROP_ARG_CMDS = [
    "TODO", "michellF", "colleagueF", "springerF",  # notes, not claims
    "ref", "eqref", "pageref", "autoref", "cref", "Cref", "label",
    "cite", "citep", "citet", "citealp", "citeauthor", "citeyear", "nocite",
    "input", "include", "includegraphics", "bibliography", "bibliographystyle",
    "texttt", "url", "href", "hspace", "vspace", "setlength", "rule", "kern",
]
# Environments whose next brace group is a layout spec (columns, width).
SPEC_ENVS = {"tabular", "tabularx", "array", "subfigure", "minipage", "tabular*"}

UNITS = ["km/h", "m/s", "km", "GB", "MB", "min", "ms", "s", "h", "m", "%"]
# A clock time such as 10:00 is one number. Digits glued to a letter (D29), to a
# backslash command or to an underscore are not numbers.
RE_NUM = re.compile(r"(?<![A-Za-z\\_.\d:])(?:\d{1,2}:\d{2}(?!\d)|\d+(?:\.\d+)?)")
# Words around a year that make it a calendar year and not a count.
YEAR_PREV = {"in", "from", "to", "since", "the", "by", "until", "and", "of", "between"}
COUNT_NEXT = {"requests", "instances", "pairs", "points", "stations", "vehicles", "fleets",
              "draws", "seats", "nodes", "edges", "zones"}
RE_UNIT = re.compile(r"^\$?\s*(?:\\,|~|\s)?\s*(\\%|%|km/h|m/s|km|GB|MB|min|ms|s|h|m)(?![A-Za-z])")


def strip_comment(line: str) -> str:
    return re.split(r"(?<!\\)%", line, maxsplit=1)[0]


def _blank(text: str, start: int, end: int) -> str:
    """Replace text[start:end] by spaces, keeping newlines (so line numbers stay)."""
    seg = re.sub(r"[^\n]", " ", text[start:end])
    return text[:start] + seg + text[end:]


def _group_end(text: str, i: int, open_c="{", close_c="}") -> int:
    """text[i] is open_c; return the index after its matching close_c (or len)."""
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n":
        i += 1
    return i


def strip_tex(text: str) -> str:
    """Return the text with everything that is not a claim blanked out.

    The result has the same length and line structure as the input.
    """
    text = "\n".join(strip_comment(l) for l in text.split("\n"))
    cmd_re = re.compile(r"\\(" + "|".join(re.escape(c) for c in DROP_ARG_CMDS) + r")\*?(?![A-Za-z])")
    pos = 0
    while True:
        m = cmd_re.search(text, pos)
        if not m:
            break
        i = _skip_ws(text, m.end())
        # optional [..] argument (e.g. \includegraphics[width=..]{file})
        while i < len(text) and text[i] == "[":
            i = _group_end(text, i, "[", "]")
            i = _skip_ws(text, i)
        if i < len(text) and text[i] == "{":
            end = _group_end(text, i)
            text = _blank(text, m.start(), end)
            pos = end
        else:
            pos = m.end()
    # \begin{env} and \end{env}: drop the name, and the spec group of SPEC_ENVS.
    pos = 0
    while True:
        m = re.compile(r"\\(begin|end)\{([^}]*)\}").search(text, pos)
        if not m:
            break
        end = m.end()
        if m.group(1) == "begin" and m.group(2) in SPEC_ENVS:
            i = _skip_ws(text, end)
            while i < len(text) and text[i] == "[":
                i = _group_end(text, i, "[", "]")
                i = _skip_ws(text, i)
            if i < len(text) and text[i] == "{":
                end = _group_end(text, i)
        text = _blank(text, m.start(), end)
        pos = end
    # Widths such as 0.48\textwidth, spacing such as \\[2pt], ISO dates.
    text = re.sub(r"[\d.]+\s*\\(?:text|line|column)width|[\d.]+\s*\\textheight",
                  lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"\\\\\[[^\]]*\]", lambda m: " " * len(m.group(0)), text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", lambda m: " " * len(m.group(0)), text)
    return text


def mark_math(text: str) -> list[bool]:
    """For each character, whether it lies inside $...$ or an equation environment."""
    inside = [False] * len(text)
    i = 0
    in_dollar = False
    env_depth = 0
    while i < len(text):
        if text.startswith("\\begin{equation", i) or text.startswith("\\begin{align", i):
            env_depth += 1
        elif text.startswith("\\end{equation", i) or text.startswith("\\end{align", i):
            env_depth = max(0, env_depth - 1)
        c = text[i]
        if c == "\\" and i + 1 < len(text) and text[i + 1] == "$":
            i += 2
            continue
        if c == "$":
            in_dollar = not in_dollar
        inside[i] = in_dollar or env_depth > 0
        i += 1
    return inside


def find_numbers(raw: str) -> list[dict]:
    """Find numbers in a LaTeX text. Each hit: line, number, unit, context."""
    clean = strip_tex(raw)
    # 10\,000 and 10~000 become 10000. Deleting the two characters keeps the
    # line structure, so the line numbers stay right.
    clean = re.sub(r"(?<=\d)(?:\\,|~)(?=\d{3}(?!\d))", "", clean)
    math = mark_math(clean)
    hits = []
    for m in RE_NUM.finditer(clean):
        s, e = m.start(), m.end()
        num = m.group(0)
        um = RE_UNIT.match(clean[e:e + 12])
        unit = um.group(1).replace("\\", "") if um else ""
        if "." not in num and len(num) == 1 and math[s] and not unit:
            continue  # single digit in math: exponent, index, model constant (L = 3\,km stays)
        if re.fullmatch(r"(?:19|20)\d\d", num) and not unit:
            prev = re.search(r"([A-Za-z]+)\W*$", clean[max(0, s - 15):s])
            nxt = re.match(r"\W*([A-Za-z]+)", clean[e:e + 15])
            if (prev and prev.group(1).lower() in YEAR_PREV
                    and not (nxt and nxt.group(1).lower() in COUNT_NEXT)):
                continue  # a calendar year, not a measured number
        line = clean.count("\n", 0, s) + 1
        ctx = re.sub(r"\s+", " ", clean[max(0, s - 20):e + 20])
        hits.append({"line": line, "number": num, "unit": unit, "context": ctx.strip()[:40]})
    return hits


def resolve_files(chapter: Path, sections: list[str]) -> list[Path]:
    """Section files named by their number prefix (04 -> sections/04-*.tex), plus the
    table files they \\input, followed recursively. Order: section, then tables."""
    files: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path):
        if path in seen:
            return
        seen.add(path)
        files.append(path)
        text = "\n".join(strip_comment(l) for l in path.read_text(encoding="utf-8").splitlines())
        for m in re.finditer(r"\\input\{([^}]+)\}", text):
            name = m.group(1)
            target = chapter / (name if name.endswith(".tex") else name + ".tex")
            if target.exists():
                add(target)

    for sec in sections:
        matches = sorted((chapter / "sections").glob(f"{sec}*.tex"))
        if not matches:
            raise SystemExit(f"no section file matches sections/{sec}*.tex")
        for p in matches:
            add(p)
    return files


def all_section_prefixes(chapter: Path) -> list[str]:
    """Every file in sections/ as a prefix that resolve_files accepts (the file stem)."""
    return [p.stem for p in sorted((chapter / "sections").glob("*.tex"))]


def per_file_counts(hits_by_file, mapped, untraced, unmapped) -> list[tuple]:
    """(file, found, mapped, untraced, unmapped) for every file that was read."""
    def n(lst, f):
        return sum(1 for h in lst if h["file"] == f)
    return [(f, len(h), n(mapped, f), n(untraced, f), n(unmapped, f))
            for f, h in hits_by_file.items()]


def load_registry(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        # skip comment lines that start with #
        lines = [l for l in f if not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def keys_for(hit: dict) -> list[str]:
    n, u = hit["number"], hit["unit"]
    keys = [n]
    if u:
        keys.append(n + u)
    return keys


def classify(hits_by_file: dict, registry: list[dict]):
    """Return (mapped, untraced, unmapped, stale_rows)."""
    # key: (file name, line or None, number)
    index: dict[tuple[str, int | None, str], list[dict]] = {}
    for row in registry:
        name, _, line = row["section_file"].strip().partition(":")
        key = (Path(name).name, int(line) if line.strip().isdigit() else None, row["number"].strip())
        index.setdefault(key, []).append(row)
    used: set[int] = set()
    mapped, untraced, unmapped = [], [], []
    for fname, hits in hits_by_file.items():
        base = Path(fname).name
        for h in hits:
            rows = []
            for k in keys_for(h):
                rows += index.get((base, None, k), [])
                rows += index.get((base, h["line"], k), [])
            rec = {**h, "file": fname}
            if not rows:
                unmapped.append(rec)
                continue
            used.update(id(r) for r in rows)
            if any(not r["command"].strip().upper().startswith("UNSURE") for r in rows):
                mapped.append(rec)
            else:
                untraced.append(rec)
    stale = [r for r in registry if id(r) not in used
             and Path(r["section_file"].partition(":")[0]).name
             in {Path(f).name for f in hits_by_file}]
    return mapped, untraced, unmapped, stale


def fmt(hit: dict) -> str:
    unit = f" {hit['unit']}" if hit["unit"] else ""
    return f"{hit['file']}:{hit['line']}  {hit['number']}{unit}  {hit['context']!r}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chapter-dir", required=True, type=Path)
    ap.add_argument("--sections", nargs="+", default=[], help="section numbers, e.g. 04 05")
    ap.add_argument("--section-all", action="store_true",
                    help="run every file in sections/ (and the tables they input)")
    ap.add_argument("--registry", type=Path, default=Path(__file__).with_name("number_sources.csv"))
    ap.add_argument("--report-only", action="store_true", help="always exit 0")
    ap.add_argument("--strict", action="store_true", help="untraced rows also fail")
    ap.add_argument("--all", action="store_true", help="print the mapped numbers too")
    a = ap.parse_args(argv)
    chapter = a.chapter_dir.resolve()
    if a.section_all:
        sections = all_section_prefixes(chapter)
    elif a.sections:
        sections = a.sections
    else:
        ap.error("give --sections NN [NN ...] or --section-all")
    files = resolve_files(chapter, sections)
    hits_by_file = {}
    for f in files:
        rel = f.relative_to(chapter).as_posix()
        hits_by_file[rel] = find_numbers(f.read_text(encoding="utf-8"))
    registry = load_registry(a.registry)
    mapped, untraced, unmapped, stale = classify(hits_by_file, registry)

    total = sum(len(h) for h in hits_by_file.values())
    print(f"Files: {', '.join(hits_by_file)}")
    print(f"Numbers found: {total}")
    print(f"MAPPED: {len(mapped)}  MAPPED-UNTRACED: {len(untraced)}  UNMAPPED: {len(unmapped)}")
    print()
    print(f"{'File':<40}{'found':>6}{'MAPPED':>8}{'UNTRACED':>10}{'UNMAPPED':>10}")
    for f, nf, nm, nu, nx in per_file_counts(hits_by_file, mapped, untraced, unmapped):
        print(f"{f:<40}{nf:>6}{nm:>8}{nu:>10}{nx:>10}")
    print()
    print(f"Registry rows with no matching number in these files (stale): {len(stale)}")
    if a.all:
        print("\nMAPPED (file:line  number  context)")
        for h in mapped:
            print("  " + fmt(h))
    if untraced:
        print("\nMAPPED-UNTRACED (row exists, command is UNSURE)")
        for h in untraced:
            print("  " + fmt(h))
    if unmapped:
        print("\nUNMAPPED (file:line  number  context)")
        for h in unmapped:
            print("  " + fmt(h))
    if stale:
        print("\nSTALE registry rows")
        for r in stale:
            print(f"  {r['section_file']}  {r['number']}")
    bad = bool(unmapped) or (a.strict and bool(untraced))
    return 0 if (a.report_only or not bad) else 1


if __name__ == "__main__":
    sys.exit(main())
