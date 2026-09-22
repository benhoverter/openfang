#!/usr/bin/env python3
"""pdf2md.py -- tagged-PDF -> Markdown via the document logical structure tree.

Invoked by file_convert's pdf->md recipe (never a shell string): argv is an
array, so paths and option values arrive as literal args with no shell parsing,
globbing, or word-splitting.

  pdf2md.py <input.pdf> -o <output.md>
            [--min-chars N] [--require-tables true|false]
            [--provenance true|false] [--debug]

WHY A STRUCTURE TREE AND NOT GEOMETRY
-------------------------------------
Every geometric table extractor infers structure from ruling lines and word
positions. Measured on a real corpus, that inference drops the first row of
each table into the surrounding prose (no ruling above the header => no band),
and on running prose it fabricates tables that do not exist. Both failures
convert, exit 0, and look correct -- the silent-success class.

A tagged PDF STATES its structure: StructTreeRoot -> Table -> THead/TR ->
TH/TD, with each node carrying the mcids of the text it owns. This script reads
that statement instead of guessing at it. Untagged input is REFUSED (exit 4),
not best-efforted.

THREE DESIGN COMMITMENTS, each fixing a defect a naive implementation has:

1. DOCUMENT-level walk, not per-page. An element spanning a page break (a long
   table, a paragraph) appears ONCE in the document tree with mcids on both
   pages. Walking per page splits it in two and invents a header row on the
   continuation. Measured: 7 reported tables where the document has 5.

2. Reading order comes from the TREE and from (page, mcid), not from
   coordinates. Sorting an element's words by (top, x0) reorders inline runs on
   a different baseline -- inline code in a smaller font migrates to the end of
   its own paragraph. MCIDs are assigned as the content stream is written, so
   numeric order IS the order the producer laid the text down. Geometry is used
   only to order words WITHIN one mcid, where the run is contiguous anyway.

3. Word joining happens only across an mcid boundary. pdfplumber already split
   words correctly inside a run; the only real artifact is a word broken by a
   markup boundary. A proximity join applied inside a run welds tightly-set
   text into "All26toolswritetoascene".

Header rows are NOT invented: a markdown header is emitted from row 0 only when
that row's cells are tagged TH. A headerless table gets an EMPTY header row --
GFM requires one, and the emptiness is the disclosure.

KNOWN LIMIT, stated rather than hidden: code blocks lose their fencing. typst
does not tag them, so a fenced block arrives as one paragraph per line. The
content is intact and in order; the ``` markers are gone. Recovering them means
reading font names, which is geometry sneaking back in.

Exit codes (3 is reserved by file_convert for MISSING_DEP):
  0  success
  2  usage / bad input
  3  pdfplumber not importable                    -> MISSING_DEP
  4  input is UNTAGGED, or tagged but empty       -> CONVERT_FAILED
  5  wrote nothing where an output was expected   -> CONVERT_FAILED
  6  fewer than --min-chars chars, or
     --require-tables with zero tables found      -> CONVERT_FAILED
"""
import argparse
import os
import sys

# --- interpreter bootstrap -------------------------------------------------
#
# pdfplumber is a LIBRARY, not a binary, so the recipe's `needs` preflight
# (which resolves names on PATH) cannot see it. It lives in a dedicated,
# pinned venv precisely so a Homebrew python bump cannot silently orphan it.
#
# `#!/usr/bin/env python3` resolves to the system interpreter, which does not
# have it. Rather than hardcode a host-specific absolute path into a shebang
# (which would not survive being version-controlled), re-exec ONCE into the
# venv interpreter if it exists and the import fails here.
#
# OPENFANG_HOME is not on the subprocess env allow-list; HOME is. Fall back to
# it. The marker variable makes the re-exec strictly once -- without it a venv
# whose pdfplumber is broken would loop forever.
_REEXEC_MARKER = "OPENFANG_PDF2MD_REEXEC"


def _venv_python():
    home = os.environ.get("OPENFANG_HOME") or os.path.join(
        os.environ.get("HOME", os.path.expanduser("~")), ".openfang"
    )
    return os.path.join(home, "venv", "pdf", "bin", "python3")


try:
    import pdfplumber
    from pdfplumber.structure import PDFStructTree, StructTreeMissing
except ImportError as _e:
    _py = _venv_python()
    if not os.environ.get(_REEXEC_MARKER) and os.path.exists(_py):
        _env = dict(os.environ)
        _env[_REEXEC_MARKER] = "1"
        os.execve(_py, [_py, os.path.abspath(__file__)] + sys.argv[1:], _env)
    sys.stderr.write(
        f"pdf2md: cannot import pdfplumber: {_e}\n"
        f"pdf2md: tried the pinned interpreter at {_py}\n"
        "pdf2md: create it with:\n"
        "pdf2md:   python3 -m venv ~/.openfang/venv/pdf\n"
        "pdf2md:   ~/.openfang/venv/pdf/bin/python3 -m pip install pdfplumber\n"
    )
    sys.exit(3)

HEADING = {"H": 2, "H1": 1, "H2": 2, "H3": 3, "H4": 4, "H5": 5, "H6": 6}
PARA = {"P", "Caption", "Note", "BlockQuote"}
# Only used to weld a word broken across an mcid boundary. A real inter-word
# space is ~0.25-0.30 em, so this sits well below the smallest true space.
JOIN_FRAC = 0.12
# Glyphs a typesetter emits at a line break. Recognised only when the glyph is
# UNTAGGED (mcid is None) -- that is the producer telling us it is presentation,
# not content.
HYPHENS = {"-", "\u2010", "\u00ad"}


class WordIndex:
    """Lazy per-page (page_number, mcid) -> [word] index."""

    def __init__(self, pdf):
        self.pdf = pdf
        self._cache = {}
        self._hyph = {}

    def page(self, pno):
        if pno not in self._cache:
            idx = {}
            hyph = []
            try:
                page = self.pdf.pages[pno - 1]
                for w in page.extract_words(extra_attrs=["mcid"]):
                    m = w.get("mcid")
                    if m is not None:
                        idx.setdefault(m, []).append(w)
                    elif w["text"] in HYPHENS:
                        hyph.append(w)
            except Exception as e:  # a broken page must not kill the document
                sys.stderr.write(f"pdf2md: page {pno}: word extraction failed: {e}\n")
            self._cache[pno] = idx
            self._hyph[pno] = hyph
        return self._cache[pno]

    def soft_hyphen_after(self, pno, word):
        """True if an UNTAGGED hyphen sits immediately right of `word`.

        typst emits the hyphen it inserts to break a word across lines as an
        artifact OUTSIDE the structure tree, so it never reaches the mcid index
        and 'Candi' + 'dates' would otherwise space-join into 'Candi dates'.
        A HARD hyphen ('row-strip') is part of a tagged word and is handled by
        the endswith('-') rule instead, which keeps it. The tags decide, not a
        heuristic about which hyphens look real.
        """
        self.page(pno)
        for h in self._hyph.get(pno, ()):
            if abs(h["top"] - word["top"]) < 1.0 and -0.5 <= h["x0"] - word["x1"] < 2.0:
                return True
        return False

    def runs(self, el):
        """Yield one list-of-words per mcid, in reading order.

        Ordering key is (page, mcid), NOT tree order: all_mcids() yields an
        element's own mcids before its children's, so a paragraph with inline
        markup emits all its plain text first and all its styled spans after.
        """
        pairs = sorted({(p, m) for p, m in el.all_mcids() if p is not None})
        for pno, mcid in pairs:
            ws = list(self.page(pno).get(mcid, ()))
            if not ws:
                continue
            # Geometry only inside a run, where it is unambiguous.
            ws.sort(key=lambda w: (round(w["top"], 1), w["x0"]))
            yield pno, ws


def text_of(el, widx):
    parts = []
    prev = None      # previous word emitted
    prev_run = None  # its run index, so mcid boundaries are identifiable
    prev_pno = None
    for ri, (pno, ws) in enumerate(widx.runs(el)):
        for w in ws:
            if prev is not None and parts:
                same_line = abs(w["top"] - prev["top"]) < 1.0
                if not same_line and (
                    parts[-1].endswith("-") or widx.soft_hyphen_after(prev_pno, prev)
                ):
                    # Word broken by a line wrap. Two cases, distinguished by
                    # the tags rather than by guessing:
                    #   hard hyphen ("row-" + "strip") -- tagged, already in
                    #     parts[-1], so welding keeps it: "row-strip".
                    #   typesetter hyphen ("Candi" + "dates") -- untagged, never
                    #     entered the index, so welding drops it: "Candidates".
                    parts[-1] += w["text"]
                    prev, prev_pno = w, pno
                    continue
                if ri != prev_run and same_line:
                    gap = w["x0"] - prev["x1"]
                    size = max(prev.get("height") or 0, 1.0)
                    if gap < JOIN_FRAC * size:
                        parts[-1] += w["text"]
                        prev, prev_pno = w, pno
                        continue
            parts.append(w["text"])
            prev, prev_run, prev_pno = w, ri, pno
    return " ".join(parts).strip()


def esc(s):
    return s.replace("|", "\\|").replace("\n", "<br>")


def descend(el, want, out, stop=frozenset()):
    for k in el.children:
        if k.type in want:
            out.append(k)
        if k.type in stop:
            continue
        descend(k, want, out, stop)


def render_table(tb, widx):
    rows = []
    descend(tb, {"TR"}, rows, stop={"Table"})
    grid = []
    for r in rows:
        cells = []
        descend(r, {"TD", "TH"}, cells, stop={"Table"})
        vals = [esc(text_of(c, widx)) for c in cells]
        if any(vals):
            grid.append((bool(cells) and all(c.type == "TH" for c in cells), vals))
    if not grid:
        return None
    width = max(len(v) for _h, v in grid)
    if grid[0][0]:  # row 0 is genuinely tagged as a header row
        head, body = grid[0][1], [v for _h, v in grid[1:]]
    else:  # do not promote data to a header; disclose by emitting an empty one
        head, body = [""] * width, [v for _h, v in grid]
    lines = [
        "| " + " | ".join(head + [""] * (width - len(head))) + " |",
        "|" + "|".join([" --- "] * width) + "|",
    ]
    for v in body:
        lines.append("| " + " | ".join(v + [""] * (width - len(v))) + " |")
    return "\n".join(lines)


def walk(el, widx, blocks, depth=0):
    t = el.type

    if t == "Table":
        md = render_table(el, widx)
        if md:
            blocks.append(("table", md))
        return  # a table's cells are never re-emitted as prose

    if t in HEADING:
        txt = text_of(el, widx)
        if txt:
            blocks.append(("heading", "#" * HEADING[t] + " " + txt))
        return

    if t in PARA:
        txt = text_of(el, widx)
        if txt:
            blocks.append(("para", ("> " if t == "BlockQuote" else "") + txt))
        return

    if t == "LI":
        txt = text_of(el, widx)
        if txt:
            blocks.append(("li", "  " * max(depth - 1, 0) + "- " + txt))
        return

    if t == "Figure":
        alt = el.alt_text or el.actual_text or ""
        blocks.append(("figure", f"![{esc(alt)}]()"))
        return

    nd = depth + 1 if t == "L" else depth
    for k in el.children:
        walk(k, widx, blocks, nd)


def _flag(name, value):
    """Parse a true/false option. Empty means 'caller stated no preference'.

    The recipe's argv is fixed-length, so every option is always passed and the
    empty string is how 'no override' travels.
    """
    if value in ("true", ""):
        return True
    if value == "false":
        return False
    sys.stderr.write(f"pdf2md: --{name} must be true or false, got '{value}'\n")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out")
    ap.add_argument("--min-chars", default="1")
    ap.add_argument("--require-tables", default="false")
    ap.add_argument("--provenance", default="true")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()

    # Validate options BEFORE opening anything: a bad value should cost a
    # refusal, not a full extraction followed by a refusal.
    require_tables = _flag("require-tables", a.require_tables)
    provenance = _flag("provenance", a.provenance)
    min_chars = a.min_chars if a.min_chars != "" else "1"
    if not min_chars.isdigit():
        sys.stderr.write(
            f"pdf2md: --min-chars must be a non-negative integer, got '{a.min_chars}'\n"
        )
        sys.exit(2)
    min_chars = int(min_chars)

    if not os.path.isfile(a.pdf):
        sys.stderr.write(f"pdf2md: input not found: {a.pdf}\n")
        sys.exit(2)

    with pdfplumber.open(a.pdf) as pdf:
        total = len(pdf.pages)
        try:
            tree = PDFStructTree(pdf)
        except StructTreeMissing:
            sys.stderr.write(
                f"pdf2md: {a.pdf} has no StructTreeRoot -- this PDF is UNTAGGED.\n"
                "pdf2md: There is no stated structure to read, and inferring it\n"
                "pdf2md: from ruling lines drops a row per table into the prose\n"
                "pdf2md: and fabricates tables on pages that have none. Refusing\n"
                "pdf2md: rather than returning a plausible-looking wrong answer.\n"
                "pdf2md: Use file_convert pdf->txt for a best-effort text layer.\n"
            )
            sys.exit(4)

        widx = WordIndex(pdf)
        blocks = []
        for el in tree.children:
            walk(el, widx, blocks)

        if a.debug:
            for kind, md in blocks:
                sys.stderr.write(f"  {kind:8} {md.splitlines()[0][:90]}\n")

        n_tables = sum(1 for k, _ in blocks if k == "table")
        n_other = len(blocks) - n_tables
        if not blocks:
            sys.stderr.write(
                f"pdf2md: {a.pdf} is tagged but its structure tree produced no\n"
                "pdf2md: content blocks. Nothing was written.\n"
            )
            sys.exit(4)

        header = ""
        if provenance:
            header = (
                "<!-- extracted-by: pdf2md.py (tier 1, document structure tree) | "
                f"source: {os.path.basename(a.pdf)} | pages: {total} | "
                f"tables: {n_tables} | blocks: {n_other} -->\n\n"
            )
        body = header + "\n\n".join(md for _k, md in blocks) + "\n"

    # Refuse BEFORE writing. A caller that gets an error must not also find a
    # file it might read anyway.
    if require_tables and n_tables == 0:
        sys.stderr.write(
            "pdf2md: --require-tables was set and the structure tree states no\n"
            "pdf2md: Table elements. Nothing was written.\n"
        )
        sys.exit(6)
    if len(body) < min_chars:
        sys.stderr.write(
            f"pdf2md: produced only {len(body)} characters (minimum {min_chars}).\n"
            "pdf2md: Refusing rather than writing a near-empty file that looks\n"
            "pdf2md: like a successful conversion.\n"
        )
        sys.exit(6)

    if not a.out:
        sys.stdout.write(body)
        return

    with open(a.out, "w") as f:
        f.write(body)
    if not os.path.isfile(a.out):
        sys.stderr.write(f"pdf2md: wrote no output at {a.out}\n")
        sys.exit(5)
    sys.stderr.write(
        f"pdf2md: wrote {a.out}: {total} pages, {n_tables} tables, "
        f"{n_other} blocks, {len(body)} characters\n"
    )


if __name__ == "__main__":
    main()
