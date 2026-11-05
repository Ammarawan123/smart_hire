"""
doc_reader_v58.py — layout-aware document reading (PDF / DOCX / TXT) -> clean, ordered *lines*.

Why this exists (V51 failure): V51 read the two-column résumé in raw position order, so a sidebar
("EDUCATION", "LANGUAGES", "INTERESTS") was interleaved row-by-row with the project column, contact
details were glued onto project sentences ("real-time0320-2797793", "livegithub.com/faaiz-ahmed"),
and letter-spaced headings ("I N T E RE S T S") were never recognised as headings.

Pipeline (all deterministic, CPU-only, no model download):
  1. words with geometry      PyMuPDF (preferred) -> pdfplumber fallback -> pseudo-geometry from plain text
  2. contact stripping        e-mail / phone / URL / "City, Country" are removed *inside* tokens (content-based,
                              so it works even when a sidebar overlaps the text column and words are glued)
  3. fragments                words on a row are joined into fragments split at wide gaps
  4. letter-spacing repair    "P R O J E C T S" -> "PROJECTS" (word gaps kept when geometry has them)
  5. column layout            gutter detection by x-occupancy; full-width bands vs. column bands;
                              short fragments straddling the gutter are assigned by their centre
                              (recursive-XY-cut family, cf. XY-Cut++ arXiv:2504.10258; resume-specific
                              layout regeneration is also the key step in arXiv:2510.09722)
  6. lines in reading order   each Line keeps its x-indent, font size, bold flag and source row
"""
from __future__ import annotations

import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from textkit_v58 import nfkc

# ── data ────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class Word:
    text: str
    x0: float
    x1: float
    top: float
    bottom: float
    size: float = 0.0
    bold: bool = False
    page: int = 0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2

    @property
    def cy(self):
        return (self.top + self.bottom) / 2


@dataclass
class Frag:                       # a run of words on one row with no wide gap
    words: List[Word]
    text: str = ""
    x0: float = 0.0
    x1: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    size: float = 0.0
    bold: bool = False
    letterspaced: bool = False
    page: int = 0

    @property
    def cx(self):
        return (self.x0 + self.x1) / 2


@dataclass
class Line:
    text: str
    x0: float = 0.0
    size: float = 0.0
    bold: bool = False
    letterspaced: bool = False
    column: str = "full"         # full | left | right
    page: int = 0
    row: int = 0
    gap_before: float = 0.0      # vertical gap to previous line (in line-heights)


@dataclass
class ContactInfo:
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)


@dataclass
class ReadResult:
    lines: List[Line]
    contact: ContactInfo
    method: str
    layout: str                  # single | two-column | mixed
    gutter: Optional[Tuple[float, float]] = None
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""


# ── contact patterns (content-based; run on word text AND joined text so glued tokens are split) ───
EMAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
PHONE_RX = re.compile(r"(?<![\d.])(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?\d{3,4}[\s-]\d{6,8}(?!\d)|(?<![\d.])\+\d{1,3}[\s-]?\d{8,12}(?!\d)|(?<![\d.])0\d{10}(?!\d)")
URL_RX = re.compile(r"(?:https?://)?(?:www\.)?(?:github\.com|linkedin\.com|gitlab\.com|bitbucket\.org|behance\.net|kaggle\.com|medium\.com|twitter\.com|x\.com|leetcode\.com|huggingface\.co)/[A-Za-z0-9_\-./%?=&#]+", re.I)
_URL_GENERIC = re.compile(r"(?:https?://|www\.)[^\s|,;]+", re.I)
_COUNTRIES = ("Pakistan|India|Bangladesh|Sri Lanka|Nepal|UAE|United Arab Emirates|Saudi Arabia|Qatar|Oman|Kuwait|Bahrain|Turkey|Egypt|Nigeria|Kenya|South Africa|USA|United States|"
              "UK|United Kingdom|England|Canada|Australia|Germany|France|Italy|Spain|Netherlands|Sweden|Norway|Ireland|China|Japan|Singapore|Malaysia|Indonesia|Philippines|Brazil|Mexico")
LOC_RX = re.compile(r"(?<![A-Za-z])((?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)?),\s*(?:" + _COUNTRIES + r"))(?![A-Za-z])")
LETTER_SPACED_RX = re.compile(r"^(?:\S{1,2}\s+){4,}\S{1,2}$")


def strip_contact(text: str, contact: ContactInfo) -> str:
    """Remove contact tokens from `text`, record them, and leave a space where they were."""
    def rec(lst, v):
        v = v.strip(" .,;|")
        if v and v.lower() not in [x.lower() for x in lst]:
            lst.append(v)

    def sub(rx, lst, t):
        def _r(m):
            rec(lst, m.group(0))
            return " "
        return rx.sub(_r, t)
    t = text
    t = sub(EMAIL_RX, contact.emails, t)
    t = sub(URL_RX, contact.links, t)
    t = sub(_URL_GENERIC, contact.links, t)
    t = sub(PHONE_RX, contact.phones, t)

    def _loc(m):
        rec(contact.locations, m.group(1))
        return " "
    t = LOC_RX.sub(_loc, t)
    return t


# ── reading words ───────────────────────────────────────────────────────────────────────────────
def _words_pymupdf(path: str) -> List[Word]:
    try:
        import pymupdf as fitz            # PyMuPDF >= 1.24
    except ImportError:
        import fitz                       # type: ignore
    out: List[Word] = []
    with fitz.open(path) as doc:
        for pno, page in enumerate(doc):
            # span-level font info: map each word box to the span containing it
            spans = []
            try:
                for b in page.get_text("dict")["blocks"]:
                    for l in b.get("lines", []):
                        for s in l.get("spans", []):
                            spans.append((s["bbox"], s.get("size", 0.0), bool(s.get("flags", 0) & 16) or "bold" in s.get("font", "").lower()))
            except Exception:
                spans = []
            for (x0, y0, x1, y1, txt, *_r) in page.get_text("words"):
                size, bold = 0.0, False
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                for (bx0, by0, bx1, by1), sz, bd in spans:
                    if bx0 - 1 <= cx <= bx1 + 1 and by0 - 1 <= cy <= by1 + 1:
                        size, bold = sz, bd
                        break
                out.append(Word(txt, x0, x1, y0, y1, size or (y1 - y0), bold, pno))
    return out


def _words_pdfplumber(path: str) -> List[Word]:
    import pdfplumber
    out: List[Word] = []
    with pdfplumber.open(path) as pdf:
        for pno, page in enumerate(pdf.pages):
            for w in page.extract_words(keep_blank_chars=False, use_text_flow=False, extra_attrs=["size", "fontname"]):
                out.append(Word(w["text"], w["x0"], w["x1"], w["top"], w["bottom"], w.get("size", 0.0), "bold" in str(w.get("fontname", "")).lower(), pno))
    return out


def words_from_text(text: str) -> List[Word]:
    """Pseudo-geometry from plain / layout-preserved text: x = character column, y = line index."""
    out: List[Word] = []
    for row, ln in enumerate(nfkc(text).replace("\t", "    ").split("\n")):
        for m in re.finditer(r"\S+", ln):
            out.append(Word(m.group(0), float(m.start()), float(m.end()), float(row), float(row) + 0.9, 1.0, False, 0))
    return out


def _words_docx(path: str) -> List[Word]:
    try:
        import docx                       # python-docx
    except ImportError as e:
        raise RuntimeError("python-docx is needed for .docx résumés (pip install python-docx)") from e
    d = docx.Document(path)
    lines = [p.text for p in d.paragraphs]
    for t in d.tables:                    # table cells are read row by row; each cell becomes its own row
        for r in t.rows:
            for c in r.cells:
                lines.extend(c.text.split("\n"))
    return words_from_text("\n".join(lines))


def read_words(path_or_text: str, is_text: bool = False) -> Tuple[List[Word], str]:
    if is_text:
        return words_from_text(path_or_text), "text"
    p = path_or_text
    ext = os.path.splitext(p)[1].lower()
    if ext == ".pdf":
        try:
            w = _words_pymupdf(p)
            if w:
                return w, "pymupdf"
        except Exception:
            pass
        w = _words_pdfplumber(p)
        return w, "pdfplumber"
    if ext == ".docx":
        return _words_docx(p), "docx"
    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return words_from_text(f.read()), "text"


# ── rows / fragments ────────────────────────────────────────────────────────────────────────────
def _group_rows(words: List[Word]) -> List[List[Word]]:
    rows: List[List[Word]] = []
    for pno in sorted({w.page for w in words}):
        ws = sorted([w for w in words if w.page == pno], key=lambda w: (w.cy, w.x0))
        cur: List[Word] = []
        cur_y, cur_h = None, 0.0
        for w in ws:
            h = max(w.bottom - w.top, 0.5)
            if cur and abs(w.cy - cur_y) <= 0.55 * max(cur_h, h):
                cur.append(w)
                cur_y = (cur_y * (len(cur) - 1) + w.cy) / len(cur)
                cur_h = max(cur_h, h)
            else:
                if cur:
                    rows.append(sorted(cur, key=lambda x: x.x0))
                cur, cur_y, cur_h = [w], w.cy, h
        if cur:
            rows.append(sorted(cur, key=lambda x: x.x0))
    return rows


def _char_width(words: List[Word], pseudo: bool) -> float:
    if pseudo:
        return 1.0
    ratios = [(w.x1 - w.x0) / len(w.text) for w in words if len(w.text) >= 3]
    return statistics.median(ratios) if ratios else 5.0


def _despace(tokens: List[Word], pseudo: bool, cw: float) -> Tuple[str, bool]:
    """Collapse 'P R O J E C T S' (letter-spaced headings). Word breaks are kept when geometry shows a larger gap."""
    toks = [t.text for t in tokens]
    if len(toks) < 5:
        return " ".join(toks), False
    short = sum(1 for t in toks if len(t) <= 2 and t.isalpha()) / len(toks)
    if short < 0.8:
        return " ".join(toks), False
    gaps = [tokens[i + 1].x0 - tokens[i].x1 for i in range(len(tokens) - 1)]
    out, big = toks[0], (statistics.median(gaps) * 1.9 if gaps and not pseudo else None)
    for i, t in enumerate(toks[1:]):
        if big is not None and gaps[i] > big and gaps[i] > cw * 0.9:
            out += " "
        out += t
    return out, True


def _fragments(row: List[Word], pseudo: bool, cw: float) -> List[Frag]:
    thr = (3.0 if pseudo else 1.7) * cw
    groups: List[List[Word]] = []
    for w in row:
        if groups and w.x0 - groups[-1][-1].x1 <= thr:
            groups[-1].append(w)
        else:
            groups.append([w])
    frags = []
    for g in groups:
        txt, ls = _despace(g, pseudo, cw)
        sizes = [w.size for w in g if w.size]
        frags.append(Frag(g, txt, g[0].x0, g[-1].x1, min(w.top for w in g), max(w.bottom for w in g),
                          statistics.median(sizes) if sizes else 0.0, all(w.bold for w in g), ls, g[0].page))
    return frags


# ── column layout ───────────────────────────────────────────────────────────────────────────────
def _find_gutter(frag_rows: List[List[Frag]], cw: float, pseudo: bool) -> Optional[Tuple[float, float]]:
    frags = [f for r in frag_rows for f in r]
    if len(frags) < 12:
        return None
    xmin, xmax = min(f.x0 for f in frags), max(f.x1 for f in frags)
    width = xmax - xmin
    if width <= 0:
        return None
    longcut = 0.55 * width
    short = [f for f in frags if (f.x1 - f.x0) < longcut]
    nbins = 200
    bw = width / nbins
    occ = [0] * nbins
    for f in short:
        a = int((f.x0 - xmin) / bw)
        b = min(nbins - 1, int((f.x1 - xmin) / bw))
        for i in range(a, b + 1):
            occ[i] += 1
    rows_n = max(len(frag_rows), 1)
    thr = max(1, int(0.10 * rows_n))                  # tolerate a few short straddlers (centred headings)
    best, i = None, 0
    min_gap = max(2.2 * cw, 0.035 * width)
    while i < nbins:
        if occ[i] <= thr:
            j = i
            while j + 1 < nbins and occ[j + 1] <= thr:
                j += 1
            g0, g1 = xmin + i * bw, xmin + (j + 1) * bw
            if g1 - g0 >= min_gap and 0.12 * width < (g0 - xmin) and (xmax - g1) > 0.12 * width:
                gcen = (g0 + g1) / 2
                left = sum(1 for f in short if f.cx < gcen)
                right = len(short) - left
                both = sum(1 for r in frag_rows if any(f.x1 <= g0 + 0.5 * (g1 - g0) for f in r) and any(f.x0 >= g1 - 0.5 * (g1 - g0) for f in r))
                if left >= 0.18 * len(short) and right >= 0.18 * len(short) and both >= max(4, 0.12 * rows_n):
                    score = (g1 - g0) * (1 + min(left, right) / max(left, right, 1))
                    if best is None or score > best[0]:
                        best = (score, g0, g1)
            i = j + 1
        else:
            i += 1
    return (best[1], best[2]) if best else None


def _order_two_column(frag_rows: List[List[Frag]], gutter: Tuple[float, float], width: float) -> List[Tuple[Frag, str, int]]:
    """Return fragments in reading order with their column tag and source row index."""
    g0, g1 = gutter
    gc = (g0 + g1) / 2
    longcut = 0.40 * width

    def kind(r: List[Frag]) -> str:
        for f in r:
            if f.x0 < g0 + 0.5 * (g1 - g0) and f.x1 > g1 - 0.5 * (g1 - g0) and (f.x1 - f.x0) >= longcut:
                return "full"
        return "cols"
    kinds = [kind(r) for r in frag_rows]
    # a row of short fragments sandwiched between full-width rows with no content on one side is still 'full'
    bands: List[Tuple[str, List[int]]] = []
    for i, k in enumerate(kinds):
        if bands and bands[-1][0] == k:
            bands[-1][1].append(i)
        else:
            bands.append((k, [i]))
    out: List[Tuple[Frag, str, int]] = []
    for k, idxs in bands:
        if k == "full":
            for i in idxs:
                for f in frag_rows[i]:
                    out.append((f, "full", i))
            continue
        L: List[Tuple[Frag, int]] = []
        R: List[Tuple[Frag, int]] = []
        for i in idxs:
            for f in frag_rows[i]:
                (L if f.cx < gc else R).append((f, i))
        first_left = min((i for _, i in L), default=None)
        lead = [(f, i) for f, i in R if first_left is not None and i < first_left]      # right-only rows above the sidebar start
        rest = [(f, i) for f, i in R if not (first_left is not None and i < first_left)]
        # a heading-like fragment directly above the sidebar start heads the right column body that follows, not the lead-in
        while lead and (lead[-1][0].letterspaced or (lead[-1][0].text.isupper() and len(lead[-1][0].text.split()) <= 4)):
            rest.insert(0, lead.pop())
        for f, i in lead:
            out.append((f, "right", i))
        for f, i in L:
            out.append((f, "left", i))
        for f, i in rest:
            out.append((f, "right", i))
    return out


# ── main API ────────────────────────────────────────────────────────────────────────────────────
def read_document(path_or_text: str, is_text: bool = False) -> ReadResult:
    words, method = read_words(path_or_text, is_text)
    warnings: List[str] = []
    contact = ContactInfo()
    if not words:
        return ReadResult([], contact, method, "single", None, ["no text could be read (scanned/image-only PDFs need OCR, which V58 does not include)"], "")
    pseudo = method in ("text", "docx")
    # 1. strip contact tokens inside words (handles glued tokens such as "real-time0320-2797793")
    # Do it per row on the joined text first, then rebuild words from what is left.
    cw = _char_width(words, pseudo)
    cleaned: List[Word] = []
    for w in words:
        t = w.text
        if EMAIL_RX.search(t) or URL_RX.search(t) or _URL_GENERIC.search(t) or PHONE_RX.search(t) or LOC_RX.search(t):
            t2 = strip_contact(t, contact)
            parts = [p for p in re.split(r"\s+", t2) if p]
            if not parts:
                continue
            # keep geometry of the kept prefix approximately (glued suffix removed)
            keep = "".join(parts)
            frac = max(0.2, len(keep) / max(len(t), 1))
            cleaned.append(Word(" ".join(parts), w.x0, w.x0 + (w.x1 - w.x0) * frac, w.top, w.bottom, w.size, w.bold, w.page))
        else:
            cleaned.append(w)
    # phone / location can span several words ("Karachi, Pakistan", "+92 300 1234567"): handle on row text
    rows = _group_rows(cleaned)
    final_rows: List[List[Word]] = []
    for r in rows:
        joined = " ".join(w.text for w in r)
        if PHONE_RX.search(joined) or LOC_RX.search(joined) or EMAIL_RX.search(joined) or URL_RX.search(joined):
            stripped = strip_contact(joined, contact)
            if stripped.strip() != joined.strip():
                # rebuild: drop words whose text was fully consumed
                kept_tokens = set(re.findall(r"\S+", stripped))
                r = [w for w in r if any(tok in kept_tokens for tok in re.findall(r"\S+", w.text)) or w.text in kept_tokens] or []
        if r:
            final_rows.append(r)
    frag_rows = [_fragments(r, pseudo, cw) for r in final_rows]
    frag_rows = [fr for fr in frag_rows if fr]
    gutter = _find_gutter(frag_rows, cw, pseudo)
    lines: List[Line] = []
    all_f = [f for r in frag_rows for f in r]
    width = (max(f.x1 for f in all_f) - min(f.x0 for f in all_f)) if all_f else 1.0
    body_size = statistics.median([f.size for f in all_f if f.size]) if any(f.size for f in all_f) else 0.0
    layout = "single"
    if gutter:
        ordered = _order_two_column(frag_rows, gutter, width)
        kinds = {c for _, c, _ in ordered}
        layout = "two-column" if "left" in kinds and "right" in kinds else "single"
    else:
        ordered = [(f, "full", i) for i, r in enumerate(frag_rows) for f in r]
    # merge fragments that belong to the same visual line in the same column (e.g. a title and its right-aligned date)
    prev_key, buf = None, []
    merged: List[Tuple[List[Frag], str, int]] = []
    for f, col, ri in ordered:
        key = (col, ri, f.page)
        if key == prev_key:
            buf.append(f)
        else:
            if buf:
                merged.append((buf, prev_key[0], prev_key[1]))
            buf, prev_key = [f], key
    if buf:
        merged.append((buf, prev_key[0], prev_key[1]))
    last_bottom, last_col, last_page = None, None, None
    for fr, col, ri in merged:
        txt = "  ".join(f.text for f in fr).strip()
        txt = re.sub(r"^[\s|,;·•-]+(?=\S)|(?<=\S)[\s|,;·-]+$", "", txt) if not re.match(r"^[•●▪■◦○·‣►▶➢➤❖✓✔*-]\s", txt) else txt
        if not txt or not re.search(r"[A-Za-z0-9]", txt):
            continue
        h = max((f.bottom - f.top) for f in fr) or 1.0
        gap = 0.0
        if last_bottom is not None and last_col == col and last_page == fr[0].page:
            gap = max(0.0, (fr[0].top - last_bottom) / h)
        lines.append(Line(txt, fr[0].x0, statistics.median([f.size for f in fr if f.size] or [body_size]), any(f.bold for f in fr), any(f.letterspaced for f in fr), col, fr[0].page, ri, gap))
        last_bottom, last_col, last_page = max(f.bottom for f in fr), col, fr[0].page
    # a bullet glyph alone on its own line belongs to the next line ("•" / "Built a ...")
    glyphs = set("•●▪■◦○·‣►▶➢➤❖✓✔*-–—")
    fixed: List[Line] = []
    i = 0
    while i < len(lines):
        if lines[i].text.strip() in glyphs and i + 1 < len(lines):
            nx = lines[i + 1]
            fixed.append(Line("• " + nx.text, nx.x0, nx.size, nx.bold, nx.letterspaced, nx.column, nx.page, nx.row, lines[i].gap_before))
            i += 2
        else:
            fixed.append(lines[i])
            i += 1
    lines = fixed
    raw = "\n".join(l.text for l in lines)
    if layout == "two-column":
        warnings.append("two-column layout detected; reading order rebuilt (full-width bands, then left column, then right column)")
    return ReadResult(lines, contact, method, layout, gutter, warnings, raw)
