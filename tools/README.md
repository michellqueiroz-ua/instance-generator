# tools/

Scripts for the chapter. Python standard library only. They only read the chapter repository.

## check_chapter.py

Replaces: copy the chapter, compile, grep the log, count the TODOs by hand.

```
python tools/check_chapter.py --chapter-dir "<path to the chapter>" --todos on|off [--jobname NAME]
```

- Copies the chapter to a temporary folder (without `.git`, build products and PDFs, except PDFs in `figures/`), sets `\showtodostrue` or `\showtodosfalse` in the copy, runs `latexmk -pdf -interaction=nonstopmode`, deletes the folder.
- Prints one table: PDF made, pages (`pdfinfo`, else the log), error lines (`!`), undefined references, undefined citations, overfull boxes above 10 pt (count only), and `\TODO{` and `\pending{` per section and table file of the original (commented lines are ignored).
- Exit code 1 on any error, undefined reference or citation, or when no PDF is made. Otherwise 0.

## trace_numbers.py

Replaces: reading the chapter to check that every number has a file and a command behind it.

```
python tools/trace_numbers.py --chapter-dir "<path to the chapter>" --sections 04 [05 06 ...] [--report-only] [--strict] [--all]
```

- Takes `sections/<NN>*.tex` and the tables they `\input`, and lists every number as `file:line  number unit  context`.
- Looks each number up in `tools/number_sources.csv`. Output: MAPPED, MAPPED-UNTRACED (the row exists but its command is `UNSURE`), UNMAPPED, and STALE (rows that match no number in the requested files).
- Exit code 1 if anything is UNMAPPED (`--strict`: also if untraced). `--report-only` always exits 0.
- What is not counted as a number is listed at the top of the script (comments, `\TODO` and footnote notes, refs, cites, widths, years, single digits in math).

### number_sources.csv

Columns: `section_file,number,source_file,command,note`.

- `section_file`: the file where the number stands, section or table, by name (`04-instance-generation.tex`, `t04_requests.tex`). Add `:LINE` to match one line only. Use that when one number means different things in one file. Such a row turns stale when the text moves: look at the number again.
- `number`: the plain number as in the text (`3515`, `0.75`, `10:00`). Add the unit to match only that form (`3600s`, `4h`, `25%`).
- `command`: run from the repository root; it must print the number (or the values it comes from). Write `UNSURE: not traced` when the number comes from a measurement or a result file that is not reproduced yet. Do not invent a source.
- A row covers all occurrences of that number in the file. For numbers that mean different things in one file, the registry is only as good as the rows: check the `note`.
- Each traced row was run on 2026-11-19 and printed the number; the output is in the task report.
