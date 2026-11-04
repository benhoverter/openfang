//! Render Markdown (GFM) tables as PNG images for chat surfaces that can't
//! display them (Discord).
//!
//! Discord shows a pipe table as raw `|` characters, which is hard to read.
//! [`render_tables`] finds each table in an outbound message, renders it to a
//! PNG with `typst`, and replaces the table text with a short marker line.
//! The PNGs travel as ordinary `FileData` attachments; nothing is kept on
//! disk afterwards (Discord stores the upload).
//!
//! Safety model: cell text comes from agents, so it is untrusted. It is
//! written to `data.json` and read by a FIXED typst template via `json()`;
//! it is never spliced into typst source, so a cell containing typst markup
//! renders as literal text. `typst` runs with `--root` set to a fresh temp
//! dir, a hard timeout, and `kill_on_drop`. Every failure leaves the table
//! as text, which is the pre-existing behaviour.
//!
//! Styling (agent-controlled, cosmetic only):
//! - A line `<openfang:table mode="light" header="sky_blue"/>` sets the
//!   light/dark theme and header color for the NEXT table. The line is
//!   always removed from the text, even when the table stays text.
//! - A row whose first cell starts with `{red}` or `{#db4035}` gets a left
//!   bar in that exact color and a tint of it, lightened or darkened until
//!   the text keeps [`MIN_CONTRAST`]. The marker is removed. An unknown
//!   `{…}` is left in the cell untouched.
//!
//! Colors are accepted only as a Todoist palette name or a hex code, are
//! re-emitted by Rust as `#rrggbb`, and reach typst through data.json like
//! the cells do. A bad value is ignored and the default is used; styling
//! never costs the table.

use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::sync::Semaphore;
use tokio::time::Instant;
use tracing::warn;

use crate::types::ChannelContent;

const MAX_TABLES: usize = 8;
const MAX_ROWS: usize = 40;
const MAX_COLS: usize = 10;
const MAX_CELL_CHARS: usize = 500;
const RENDER_TIMEOUT: Duration = Duration::from_secs(5);
/// Wall-clock budget for all tables in one message (including waiting for a
/// render slot), so a slow typst can't hold a reply for 8 × 5 s.
const MESSAGE_BUDGET: Duration = Duration::from_secs(12);
const MAX_PNG_BYTES: usize = 8 * 1024 * 1024;
/// Cap on the estimated raster size. The input limits alone allow a
/// ~177M-pixel image (~700 MB RGBA) that typst allocates before the
/// timeout or the PNG-size check can stop it.
const MAX_PIXELS: f64 = 25_000_000.0;
/// Preferred ppi, and the fallback for tables too big at the preferred one.
const PPI_CHOICES: [u32; 2] = [192, 144];
/// Process-wide cap on concurrent typst renders (each can use hundreds of
/// MB), so many agents replying with tables at once queue instead of piling up.
static RENDER_SLOTS: Semaphore = Semaphore::const_new(2);
/// Columns whose longest cell exceeds this many chars get a fixed width so
/// typst wraps them; shorter columns size to content.
const WRAP_AFTER_CHARS: usize = 36;

/// The daemon runs under launchd with a minimal PATH, so check the usual
/// install locations before falling back to a PATH lookup.
const TYPST_CANDIDATES: &[&str] = &["/opt/homebrew/bin/typst", "/usr/local/bin/typst"];

/// Minimum WCAG contrast ratio between a row's text and its tint.
const MIN_CONTRAST: f64 = 4.5;
/// Strongest share of the source color in a row tint, before the contrast
/// check pulls it back toward the page color.
const MAX_TINT: f64 = 0.45;

/// Todoist's API color names (developer docs, "Colors" table).
const TODOIST_COLORS: &[(&str, u32)] = &[
    ("berry_red", 0xB8255F),
    ("red", 0xDC4C3E),
    ("orange", 0xC77100),
    ("yellow", 0xB29104),
    ("olive_green", 0x949C31),
    ("lime_green", 0x65A33A),
    ("green", 0x369307),
    ("mint_green", 0x42A393),
    ("teal", 0x148FAD),
    ("sky_blue", 0x319DC0),
    ("light_blue", 0x6988A4),
    ("blue", 0x4180FF),
    ("grape", 0x692EC2),
    ("violet", 0xCA3FEE),
    ("lavender", 0xA4698C),
    ("magenta", 0xE05095),
    ("salmon", 0xC9766F),
    ("charcoal", 0x808080),
    ("grey", 0x999999),
    ("taupe", 0x8F7A69),
];

/// Fixed template: approved mockup (zebra rows), every color from data.json.
/// Reads all content from data.json — keep it that way (see module docs).
const TEMPLATE: &str = r##"#let d = json("data.json")
#let th = d.theme
#set page(width: auto, height: auto, margin: 14pt, fill: rgb(th.page))
#set text(font: ("Helvetica Neue", "Helvetica", "Arial"), size: 13pt, fill: rgb(th.text))
#let al(a) = if a == "right" { right } else if a == "center" { center } else { left }
#let col(c) = if c == none { none } else { rgb(c) }
#table(
  columns: d.widths.map(w => if w == 0 { auto } else { w * 1em }),
  align: d.align.map(al),
  inset: (x: 10pt, y: 7pt),
  stroke: (x, y) => (
    bottom: if y == 0 { 1.2pt + rgb(th.header_rule) } else { 0.5pt + rgb(th.rule) },
    left: if x == 0 and d.bars.at(y) != none { 4pt + rgb(d.bars.at(y)) } else { none },
  ),
  fill: (x, y) => col(d.fills.at(y)),
  table.header(..d.header.map(h => strong[#h])),
  ..d.rows.flatten().map(c => [#c]),
)
"##;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Rgb(u8, u8, u8);

impl Rgb {
    const fn from_u32(v: u32) -> Self {
        Rgb((v >> 16) as u8, (v >> 8) as u8, v as u8)
    }

    fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }

    /// WCAG relative luminance.
    fn luminance(self) -> f64 {
        let lin = |c: u8| {
            let c = f64::from(c) / 255.0;
            if c <= 0.04045 {
                c / 12.92
            } else {
                ((c + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(self.0) + 0.7152 * lin(self.1) + 0.0722 * lin(self.2)
    }

    fn contrast(self, other: Rgb) -> f64 {
        let (a, b) = (self.luminance(), other.luminance());
        (a.max(b) + 0.05) / (a.min(b) + 0.05)
    }

    /// `w` of `self` blended with `1 - w` of `base`.
    fn mix(self, base: Rgb, w: f64) -> Rgb {
        let m = |a: u8, b: u8| (f64::from(a) * w + f64::from(b) * (1.0 - w)).round() as u8;
        Rgb(m(self.0, base.0), m(self.1, base.1), m(self.2, base.2))
    }
}

/// A Todoist color name or a `#rgb` / `#rrggbb` hex code. Anything else is
/// `None` (the caller then keeps its default).
fn parse_color(s: &str) -> Option<Rgb> {
    let s = s.trim();
    if let Some(hex) = s.strip_prefix('#') {
        if !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        let v = u32::from_str_radix(hex, 16).ok()?;
        return match hex.len() {
            6 => Some(Rgb::from_u32(v)),
            3 => {
                let n = |shift: u32| ((v >> shift) & 0xF) as u8 * 17;
                Some(Rgb(n(8), n(4), n(0)))
            }
            _ => None,
        };
    }
    let s = s.to_ascii_lowercase();
    TODOIST_COLORS
        .iter()
        .find(|(name, _)| *name == s)
        .map(|&(_, v)| Rgb::from_u32(v))
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Mode {
    #[default]
    Dark,
    Light,
}

struct Theme {
    page: Rgb,
    text: Rgb,
    header: Rgb,
    stripe: Rgb,
    header_rule: Rgb,
    rule: Rgb,
}

impl Mode {
    fn theme(self) -> Theme {
        match self {
            // Discord dark: the original palette, unchanged.
            Mode::Dark => Theme {
                page: Rgb::from_u32(0x313338),
                text: Rgb::from_u32(0xdbdee1),
                header: Rgb::from_u32(0x2b2d31),
                stripe: Rgb::from_u32(0x383a40),
                header_rule: Rgb::from_u32(0x949ba4),
                rule: Rgb::from_u32(0x4e5058),
            },
            // Discord light.
            Mode::Light => Theme {
                page: Rgb::from_u32(0xffffff),
                text: Rgb::from_u32(0x060607),
                header: Rgb::from_u32(0xebedef),
                stripe: Rgb::from_u32(0xf2f3f5),
                header_rule: Rgb::from_u32(0x80848e),
                rule: Rgb::from_u32(0xd4d7dc),
            },
        }
    }
}

impl Theme {
    /// The strongest blend of `src` into the page color (at most
    /// [`MAX_TINT`]) that keeps the text at [`MIN_CONTRAST`]. The plain page
    /// color always passes, so this always returns.
    fn tint(&self, src: Rgb) -> Rgb {
        let steps = (MAX_TINT / 0.05).round() as u32;
        (0..=steps)
            .rev()
            .map(|s| src.mix(self.page, f64::from(s) * 0.05))
            .find(|c| c.contrast(self.text) >= MIN_CONTRAST)
            .unwrap_or(self.page)
    }
}

/// Per-table styling from the tag line and row markers.
#[derive(Debug, Clone, Default, PartialEq)]
struct Style {
    mode: Mode,
    header: Option<Rgb>,
    /// One entry per body row.
    rows: Vec<Option<Rgb>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Align {
    Left,
    Center,
    Right,
    /// No colon in the divider: decided from the column's contents.
    Unspecified,
}

#[derive(Debug, Clone, PartialEq)]
struct Table {
    header: Vec<String>,
    align: Vec<Align>,
    rows: Vec<Vec<String>>,
}

/// A table found in the text: its line range `[start, end)` and contents.
#[derive(Debug)]
struct Found {
    start: usize,
    end: usize,
    table: Table,
    style: Style,
}

/// The text with style markup removed, and the tables found in it. Line
/// ranges in `found` index into `lines`.
struct Scan {
    lines: Vec<String>,
    found: Vec<Found>,
}

/// Remove table style markup (tag lines, valid row markers) without
/// rendering. For channels that send tables as text, and for the
/// send-failure fallback.
pub fn strip_markup(text: &str) -> String {
    scan(text).lines.join("\n")
}

/// Replace renderable tables in `text` with marker lines and return the new
/// text plus one PNG `FileData` per rendered table. Tables that fail any
/// limit or fail to render stay in the text unchanged.
pub async fn render_tables(text: &str) -> (String, Vec<ChannelContent>) {
    let Scan { lines, found } = scan(text);
    if found.is_empty() {
        return (lines.join("\n"), Vec::new());
    }
    let Some(typst) = find_typst() else {
        warn!("table_render: typst not found; sending tables as text");
        return (lines.join("\n"), Vec::new());
    };
    let deadline = Instant::now() + MESSAGE_BUDGET;

    let mut out: Vec<String> = Vec::with_capacity(lines.len());
    let mut files = Vec::new();
    let mut cursor = 0;
    for f in found {
        out.extend(lines[cursor..f.start].iter().cloned());
        cursor = f.start;
        if files.len() >= MAX_TABLES {
            continue;
        }
        let ppi = match check_limits(&f.table).and_then(|()| pick_ppi(&f.table)) {
            Ok(ppi) => ppi,
            Err(why) => {
                warn!("table_render: table left as text: {why}");
                continue;
            }
        };
        match render_limited(&typst, &f.table, &f.style, ppi, deadline).await {
            Ok(png) => {
                let n = files.len() + 1;
                files.push(ChannelContent::FileData {
                    data: png,
                    filename: format!("table-{n}.png"),
                    mime_type: "image/png".to_string(),
                });
                out.push(format!("📊 *Table {n} (image below)*"));
                cursor = f.end;
            }
            Err(why) => warn!("table_render: render failed, table left as text: {why}"),
        }
    }
    out.extend(lines[cursor..].iter().cloned());
    (out.join("\n"), files)
}

/// Render under the process-wide slot limit and the message deadline.
async fn render_limited(
    typst: &Path,
    table: &Table,
    style: &Style,
    ppi: u32,
    deadline: Instant,
) -> Result<Vec<u8>, String> {
    let _slot = tokio::time::timeout_at(deadline, RENDER_SLOTS.acquire())
        .await
        .map_err(|_| "message render budget spent waiting for a slot".to_string())?
        .map_err(|e| format!("render slots closed: {e}"))?;
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err("message render budget spent".into());
    }
    render_png(typst, table, style, ppi, left.min(RENDER_TIMEOUT)).await
}

fn scan(text: &str) -> Scan {
    let src: Vec<&str> = text.split('\n').collect();
    let mut lines: Vec<String> = Vec::with_capacity(src.len());
    let mut found = Vec::new();
    let mut fence: Option<&str> = None;
    // Set by a tag line; used by the next table.
    let mut pending: Option<(Mode, Option<Rgb>)> = None;
    let mut i = 0;
    while i < src.len() {
        let line = src[i];
        let trimmed = line.trim_start();
        if let Some(open) = fence {
            if trimmed.starts_with(open) {
                fence = None;
            }
            lines.push(line.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("```") {
            fence = Some("```");
            pending = None;
            lines.push(line.to_string());
            i += 1;
            continue;
        }
        if trimmed.starts_with("~~~") {
            fence = Some("~~~");
            pending = None;
            lines.push(line.to_string());
            i += 1;
            continue;
        }
        if let Some(tag) = parse_tag(line) {
            pending = Some(tag);
            i += 1;
            continue;
        }
        if i + 1 < src.len() && line.contains('|') {
            let header = split_row(line);
            if let Some(align) = parse_divider(src[i + 1]) {
                if align.len() == header.len() && !header.is_empty() {
                    let start = lines.len();
                    lines.push(line.to_string());
                    lines.push(src[i + 1].to_string());
                    let mut j = i + 2;
                    let mut rows = Vec::new();
                    let mut colors = Vec::new();
                    while j < src.len() && src[j].contains('|') && !src[j].trim().is_empty() {
                        let (color, row_line) = take_row_marker(src[j]);
                        let mut row = split_row(&row_line);
                        row.resize(header.len(), String::new());
                        rows.push(row);
                        colors.push(color);
                        lines.push(row_line);
                        j += 1;
                    }
                    let (mode, header_color) = pending.take().unwrap_or_default();
                    found.push(Found {
                        start,
                        end: lines.len(),
                        table: Table {
                            header,
                            align,
                            rows,
                        },
                        style: Style {
                            mode,
                            header: header_color,
                            rows: colors,
                        },
                    });
                    i = j;
                    continue;
                }
            }
        }
        // A tag styles only a table that follows it directly (blank lines
        // allowed); any other text in between cancels it.
        if !line.trim().is_empty() {
            pending = None;
        }
        lines.push(line.to_string());
        i += 1;
    }
    Scan { lines, found }
}

/// Parse a whole-line `<openfang:table key="value" …/>` tag. Unknown keys
/// and bad values are ignored; `None` means the line isn't a tag.
fn parse_tag(line: &str) -> Option<(Mode, Option<Rgb>)> {
    let inner = line
        .trim()
        .strip_prefix("<openfang:table")?
        .strip_suffix("/>")?;
    if inner.chars().next().is_some_and(|c| !c.is_whitespace()) {
        return None; // e.g. `<openfang:tables/>`
    }
    let mut mode = Mode::default();
    let mut header = None;
    for part in inner.split_whitespace() {
        let Some((key, value)) = part.split_once('=') else {
            continue;
        };
        let value = value.trim_matches(|c| c == '"' || c == '\'');
        match key {
            "mode" if value.eq_ignore_ascii_case("light") => mode = Mode::Light,
            "mode" if value.eq_ignore_ascii_case("dark") => mode = Mode::Dark,
            "header" => header = parse_color(value),
            _ => {}
        }
    }
    Some((mode, header))
}

/// Split a leading `{color}` marker off a row's first cell. Returns the
/// color and the row line without the marker, or `(None, line)` unchanged
/// when there's no valid marker. The marker must be followed by a space
/// and more text in the same cell, so a cell that is only `{red}` keeps
/// its text.
fn take_row_marker(line: &str) -> (Option<Rgb>, String) {
    let mut prefix = line.len() - line.trim_start().len();
    if line[prefix..].starts_with('|') {
        prefix += 1;
    }
    prefix += line[prefix..].len() - line[prefix..].trim_start().len();
    let rest = &line[prefix..];
    let color = rest
        .strip_prefix('{')
        .and_then(|r| r.find('}').map(|end| &r[..end]))
        .and_then(|name| parse_color(name).map(|c| (c, name.len() + 2)))
        .filter(|&(_, used)| {
            let after = &rest[used..];
            let text = after.trim_start();
            after.starts_with([' ', '\t']) && !text.is_empty() && !text.starts_with('|')
        });
    match color {
        Some((c, used)) => (
            Some(c),
            format!("{}{}", &line[..prefix], rest[used..].trim_start()),
        ),
        None => (None, line.to_string()),
    }
}

/// Split a table row on unescaped `|`, dropping the optional outer pipes.
fn split_row(line: &str) -> Vec<String> {
    let t = line.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = if t.ends_with('|') && !t.ends_with("\\|") {
        &t[..t.len() - 1]
    } else {
        t
    };
    let mut cells = Vec::new();
    let mut cur = String::new();
    let mut chars = t.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'|') {
            cur.push('|');
            chars.next();
        } else if c == '|' {
            cells.push(clean_cell(&cur));
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    cells.push(clean_cell(&cur));
    cells
}

/// Parse a GFM divider row (`|:---|---:|`). `None` if it isn't one.
fn parse_divider(line: &str) -> Option<Vec<Align>> {
    if !line.contains('-') {
        return None;
    }
    split_row(line)
        .iter()
        .map(|c| {
            let c = c.trim();
            let left = c.starts_with(':');
            let right = c.ends_with(':');
            let dashes = c.trim_matches(':');
            if dashes.is_empty() || !dashes.chars().all(|ch| ch == '-') {
                return None;
            }
            Some(match (left, right) {
                (true, true) => Align::Center,
                (false, true) => Align::Right,
                (true, false) => Align::Left,
                (false, false) => Align::Unspecified,
            })
        })
        .collect()
}

/// Drop inline Markdown the image can't show: bold markers, code ticks, and
/// link targets (`[text](url)` → `text`).
fn clean_cell(raw: &str) -> String {
    let s = raw.trim().replace("**", "").replace('`', "");
    let mut out = String::with_capacity(s.len());
    let mut rest = s.as_str();
    while let Some(open) = rest.find('[') {
        let after = &rest[open + 1..];
        let link = after.find("](").and_then(|close| {
            after[close + 2..]
                .find(')')
                .map(|end| (close, close + 2 + end + 1))
        });
        match link {
            Some((close, consumed)) => {
                out.push_str(&rest[..open]);
                out.push_str(&after[..close]);
                rest = &after[consumed..];
            }
            None => {
                out.push_str(&rest[..=open]);
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

fn check_limits(t: &Table) -> Result<(), String> {
    if t.rows.len() > MAX_ROWS {
        return Err(format!("{} rows > {MAX_ROWS}", t.rows.len()));
    }
    if t.header.len() > MAX_COLS {
        return Err(format!("{} columns > {MAX_COLS}", t.header.len()));
    }
    let too_long = t
        .rows
        .iter()
        .chain(std::iter::once(&t.header))
        .flatten()
        .any(|c| c.chars().count() > MAX_CELL_CHARS);
    if too_long {
        return Err(format!("a cell exceeds {MAX_CELL_CHARS} chars"));
    }
    Ok(())
}

/// Deliberately pessimistic estimate of the rendered page size in points,
/// mirroring TEMPLATE: 13pt text, 10/7pt cell insets, 14pt margin, wrapped
/// columns `WRAP_AFTER_CHARS / 2` em wide. Overestimates char width and
/// line count so real output stays under the cap.
fn estimate_page_pt(t: &Table) -> (f64, f64) {
    const FONT: f64 = 13.0;
    const CHAR_W: f64 = 0.6 * FONT; // Helvetica averages ~0.5em; bold headers wider
    const LINE_H: f64 = 1.3 * FONT;
    const INSET_X: f64 = 20.0;
    const INSET_Y: f64 = 14.0;
    const MARGIN: f64 = 28.0;
    let fixed_w = (WRAP_AFTER_CHARS / 2) as f64 * FONT;
    // Fewer chars per line than fit, so the line count rounds up.
    let chars_per_line = (fixed_w / CHAR_W).floor().max(1.0);

    let widths = template_data(t, &Style::default())["widths"].clone();
    let fixed: Vec<bool> = (0..t.header.len())
        .map(|c| widths[c].as_u64().unwrap_or(0) != 0)
        .collect();
    let longest = |c: usize| {
        t.rows
            .iter()
            .map(|r| r[c].chars().count())
            .chain(std::iter::once(t.header[c].chars().count()))
            .max()
            .unwrap_or(0) as f64
    };
    let width: f64 = (0..t.header.len())
        .map(|c| {
            let content = if fixed[c] {
                fixed_w
            } else {
                longest(c) * CHAR_W
            };
            content + INSET_X
        })
        .sum::<f64>()
        + MARGIN;
    let row_h = |row: &Vec<String>| {
        let lines = row
            .iter()
            .zip(&fixed)
            .map(|(cell, &f)| {
                if f {
                    (cell.chars().count() as f64 / chars_per_line)
                        .ceil()
                        .max(1.0)
                } else {
                    1.0
                }
            })
            .fold(1.0, f64::max);
        lines * LINE_H + INSET_Y
    };
    let height: f64 = std::iter::once(&t.header)
        .chain(&t.rows)
        .map(row_h)
        .sum::<f64>()
        + MARGIN;
    (width, height)
}

/// Highest ppi whose estimated raster fits [`MAX_PIXELS`], or an error when
/// even the lowest doesn't (the table then stays text).
fn pick_ppi(t: &Table) -> Result<u32, String> {
    let (w, h) = estimate_page_pt(t);
    let px_at = |ppi: u32| {
        let s = f64::from(ppi) / 72.0;
        w * s * h * s
    };
    PPI_CHOICES
        .into_iter()
        .find(|&ppi| px_at(ppi) <= MAX_PIXELS)
        .ok_or_else(|| {
            let low = PPI_CHOICES[PPI_CHOICES.len() - 1];
            format!(
                "estimated {:.0}M pixels at {low} ppi > {:.0}M",
                px_at(low) / 1e6,
                MAX_PIXELS / 1e6
            )
        })
}

/// True when a cell reads as a number or measurement ("86 s", "-3.2%", "$5").
fn looks_numeric(cell: &str) -> bool {
    let s = cell.trim_start_matches(['-', '+', '~', '$', '€', '£', '≈']);
    s.chars().next().is_some_and(|c| c.is_ascii_digit())
}

/// Template input. Kept separate from [`Table`] so the JSON shape the
/// template reads is explicit. Every color is emitted here as `#rrggbb`.
fn template_data(t: &Table, style: &Style) -> serde_json::Value {
    let ncols = t.header.len();
    let align: Vec<&str> = (0..ncols)
        .map(|c| match t.align[c] {
            Align::Left => "left",
            Align::Center => "center",
            Align::Right => "right",
            Align::Unspecified => {
                let mut cells = t
                    .rows
                    .iter()
                    .map(|r| r[c].as_str())
                    .filter(|s| !s.is_empty());
                let mut any = false;
                let all_numeric = cells.all(|s| {
                    any = true;
                    looks_numeric(s)
                });
                if any && all_numeric {
                    "right"
                } else {
                    "left"
                }
            }
        })
        .collect();
    // 0 = auto width; otherwise a fixed width in em so long text wraps.
    let widths: Vec<u32> = (0..ncols)
        .map(|c| {
            let longest = t
                .rows
                .iter()
                .map(|r| r[c].chars().count())
                .chain(std::iter::once(t.header[c].chars().count()))
                .max()
                .unwrap_or(0);
            if longest > WRAP_AFTER_CHARS {
                (WRAP_AFTER_CHARS / 2) as u32
            } else {
                0
            }
        })
        .collect();
    let theme = style.mode.theme();
    // Index 0 is the header row, as in typst's `y`.
    let row_color = |i: usize| style.rows.get(i).copied().flatten();
    let fills: Vec<Option<String>> = std::iter::once(Some(
        style.header.map_or(theme.header, |c| theme.tint(c)).hex(),
    ))
    .chain((0..t.rows.len()).map(|i| match row_color(i) {
        Some(c) => Some(theme.tint(c).hex()),
        None if (i + 1) % 2 == 0 => Some(theme.stripe.hex()),
        None => None,
    }))
    .collect();
    let bars: Vec<Option<String>> = std::iter::once(None)
        .chain((0..t.rows.len()).map(|i| row_color(i).map(Rgb::hex)))
        .collect();
    serde_json::json!({
        "header": t.header,
        "align": align,
        "widths": widths,
        "rows": t.rows,
        "theme": {
            "page": theme.page.hex(),
            "text": theme.text.hex(),
            "header_rule": theme.header_rule.hex(),
            "rule": theme.rule.hex(),
        },
        "fills": fills,
        "bars": bars,
    })
}

/// For tests elsewhere in the crate that need a real render.
#[cfg(test)]
pub(crate) fn typst_available() -> bool {
    find_typst().is_some()
}

fn find_typst() -> Option<PathBuf> {
    TYPST_CANDIDATES
        .iter()
        .map(PathBuf::from)
        .chain(
            std::env::var_os("PATH")
                .into_iter()
                .flat_map(|p| std::env::split_paths(&p).collect::<Vec<_>>())
                // An empty PATH entry splits to "" and would resolve
                // `typst` against the daemon's cwd.
                .filter(|d| d.is_absolute())
                .map(|d| d.join("typst")),
        )
        .find(|p| p.is_file())
}

/// Removes the render dir on every exit path.
struct TempDir(PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

async fn render_png(
    typst: &Path,
    table: &Table,
    style: &Style,
    ppi: u32,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    let dir =
        TempDir(std::env::temp_dir().join(format!("openfang-table-{}", uuid::Uuid::new_v4())));
    tokio::fs::create_dir(&dir.0)
        .await
        .map_err(|e| format!("create temp dir: {e}"))?;
    tokio::fs::write(dir.0.join("main.typ"), TEMPLATE)
        .await
        .map_err(|e| format!("write template: {e}"))?;
    let data = serde_json::to_vec(&template_data(table, style)).map_err(|e| e.to_string())?;
    tokio::fs::write(dir.0.join("data.json"), data)
        .await
        .map_err(|e| format!("write data: {e}"))?;

    let child = tokio::process::Command::new(typst)
        .current_dir(&dir.0)
        .arg("compile")
        .arg("--root")
        .arg(&dir.0)
        .args(["--format", "png", "--ppi"])
        .arg(ppi.to_string())
        .args(["main.typ", "out.png"])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|e| format!("spawn typst: {e}"))?;
    let output = tokio::time::timeout(timeout, child.wait_with_output())
        .await
        .map_err(|_| format!("typst timed out after {:.1}s", timeout.as_secs_f64()))?
        .map_err(|e| format!("wait typst: {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "typst exited {}: {}",
            output.status,
            openfang_types::truncate_str(err.trim(), 300)
        ));
    }
    let png = tokio::fs::read(dir.0.join("out.png"))
        .await
        .map_err(|e| format!("read png: {e}"))?;
    if png.len() > MAX_PNG_BYTES {
        return Err(format!("png is {} bytes > {MAX_PNG_BYTES}", png.len()));
    }
    Ok(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tables(text: &str) -> Vec<Found> {
        scan(text).found
    }

    const SAMPLE: &str = "intro\n\n| version | finished | median |\n|---|:---:|---:|\n| staging | 17/21 | 86 s |\n| lower | 17/21 | 90 s |\n\noutro";

    #[test]
    fn finds_plain_table() {
        let f = tables(SAMPLE);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].start, f[0].end), (2, 6));
        let t = &f[0].table;
        assert_eq!(t.header, vec!["version", "finished", "median"]);
        assert_eq!(
            t.align,
            vec![Align::Unspecified, Align::Center, Align::Right]
        );
        assert_eq!(t.rows[1], vec!["lower", "17/21", "90 s"]);
    }

    #[test]
    fn finds_table_without_outer_pipes() {
        let f = tables("a | b\n--- | ---\n1 | 2");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].table.rows, vec![vec!["1", "2"]]);
    }

    #[test]
    fn ignores_tables_in_code_fences() {
        assert!(tables("```\n| a | b |\n|---|---|\n| 1 | 2 |\n```").is_empty());
        assert!(tables("~~~md\n| a | b |\n|---|---|\n~~~").is_empty());
        // A table after the fence closes is still found.
        assert_eq!(tables("```\nx\n```\n| a |\n|---|\n| 1 |").len(), 1);
    }

    #[test]
    fn pads_and_truncates_ragged_rows() {
        let f = tables("| a | b |\n|---|---|\n| 1 |\n| 1 | 2 | 3 |");
        assert_eq!(f[0].table.rows, vec![vec!["1", ""], vec!["1", "2"]]);
    }

    #[test]
    fn escaped_pipe_stays_in_cell() {
        let f = tables("| a | b |\n|---|---|\n| x \\| y | 2 |");
        assert_eq!(f[0].table.rows[0], vec!["x | y", "2"]);
    }

    #[test]
    fn finds_two_tables_and_header_only() {
        let f = tables("| a |\n|---|\n| 1 |\n\n| b |\n|---|");
        assert_eq!(f.len(), 2);
        assert!(f[1].table.rows.is_empty());
    }

    #[test]
    fn rejects_mismatched_divider_and_prose_pipes() {
        assert!(tables("| a | b |\n|---|\n| 1 | 2 |").is_empty());
        assert!(tables("use a | b to pipe\nnext line").is_empty());
    }

    #[test]
    fn cleans_inline_markdown() {
        assert_eq!(clean_cell(" **bold** `code` "), "bold code");
        assert_eq!(clean_cell("[docs](https://x.y/z) and [x]"), "docs and [x]");
    }

    #[test]
    fn limits() {
        let mk = |rows: usize, cols: usize, cell: &str| Table {
            header: vec!["h".into(); cols],
            align: vec![Align::Left; cols],
            rows: vec![vec![cell.to_string(); cols]; rows],
        };
        assert!(check_limits(&mk(MAX_ROWS, MAX_COLS, "x")).is_ok());
        assert!(check_limits(&mk(MAX_ROWS + 1, 2, "x")).is_err());
        assert!(check_limits(&mk(1, MAX_COLS + 1, "x")).is_err());
        assert!(check_limits(&mk(1, 1, &"x".repeat(MAX_CELL_CHARS + 1))).is_err());
    }

    fn grid(rows: usize, cols: usize, cell_chars: usize) -> Table {
        Table {
            header: vec!["header".into(); cols],
            align: vec![Align::Left; cols],
            rows: vec![vec!["x".repeat(cell_chars); cols]; rows],
        }
    }

    #[test]
    fn pixel_budget_picks_ppi_or_rejects() {
        // Security's T1 case: every input limit maxed. Must stay text.
        assert!(pick_ppi(&grid(MAX_ROWS, MAX_COLS, MAX_CELL_CHARS)).is_err());
        // Ordinary tables render at full resolution.
        assert_eq!(pick_ppi(&grid(2, 5, 30)), Ok(192));
        assert_eq!(pick_ppi(&grid(MAX_ROWS, MAX_COLS, 20)), Ok(192));
        // A wide research table with long cells drops to the lower ppi.
        assert_eq!(pick_ppi(&grid(20, 6, 200)), Ok(144));
    }

    /// The estimate must not undershoot what typst actually draws, for both
    /// auto-width and wrapped columns. Skips when typst isn't installed.
    #[tokio::test]
    async fn estimate_bounds_real_render() {
        let Some(typst) = find_typst() else {
            eprintln!("typst not installed; skipping");
            return;
        };
        let words = |n: usize| "word ".repeat(n / 5);
        let cases = [
            grid(3, 4, 20),
            Table {
                header: vec!["name".into(), "notes".into(), "more".into()],
                align: vec![Align::Left; 3],
                rows: vec![vec!["Wide Mixed Case Label".into(), words(300), words(120)]; 6],
            },
        ];
        for t in cases {
            let png = render_png(&typst, &t, &Style::default(), 72, RENDER_TIMEOUT)
                .await
                .unwrap();
            // PNG IHDR: width and height are big-endian u32s at bytes 16..24.
            let w = u32::from_be_bytes(png[16..20].try_into().unwrap()) as f64;
            let h = u32::from_be_bytes(png[20..24].try_into().unwrap()) as f64;
            let (ew, eh) = estimate_page_pt(&t);
            assert!(ew >= w && eh >= h, "estimate {ew}x{eh}pt < real {w}x{h}");
        }
    }

    #[test]
    fn auto_alignment_and_widths() {
        let t = Table {
            header: vec!["name".into(), "time".into()],
            align: vec![Align::Unspecified, Align::Unspecified],
            rows: vec![
                vec!["x".repeat(50), "86 s".into()],
                vec!["y".into(), "-3%".into()],
            ],
        };
        let d = template_data(&t, &Style::default());
        assert_eq!(d["align"], serde_json::json!(["left", "right"]));
        assert_eq!(d["widths"], serde_json::json!([WRAP_AFTER_CHARS / 2, 0]));
    }

    #[test]
    fn template_takes_cells_only_as_data() {
        // Cell text must never be interpolated into typst source.
        assert!(!TEMPLATE.contains("eval"));
        assert!(!TEMPLATE.contains("read("));
        assert!(!TEMPLATE.contains("import"));
        let t = Table {
            header: vec!["#read(\"/etc/passwd\")".into()],
            align: vec![Align::Left],
            rows: vec![vec!["$x$ ] #eval(\"1\") \"".into()]],
        };
        let d = template_data(&t, &Style::default());
        assert_eq!(d["header"][0], "#read(\"/etc/passwd\")");
        assert_eq!(d["rows"][0][0], "$x$ ] #eval(\"1\") \"");
    }

    #[test]
    fn parses_colors() {
        assert_eq!(parse_color("berry_red"), Some(Rgb(0xb8, 0x25, 0x5f)));
        assert_eq!(parse_color("Sky_Blue"), Some(Rgb(0x31, 0x9d, 0xc0)));
        assert_eq!(parse_color("#DB4035"), Some(Rgb(0xdb, 0x40, 0x35)));
        assert_eq!(parse_color("#fa0"), Some(Rgb(0xff, 0xaa, 0x00)));
        for bad in [
            "gray", "crimson", "#12345", "#gggggg", "db4035", "", "#", "#+12",
        ] {
            assert_eq!(parse_color(bad), None, "{bad}");
        }
        assert_eq!(TODOIST_COLORS.len(), 20);
    }

    #[test]
    fn row_markers_color_rows_and_are_removed() {
        let s = scan("| a | b |\n|---|---|\n| {red} x | 1 |\n| y | 2 |\n|{#00ff00} z|3|");
        let f = &s.found[0];
        assert_eq!(f.table.rows[0], vec!["x", "1"]);
        assert_eq!(f.table.rows[2], vec!["z", "3"]);
        assert_eq!(
            f.style.rows,
            vec![Some(Rgb(0xdc, 0x4c, 0x3e)), None, Some(Rgb(0, 0xff, 0))]
        );
        assert_eq!(s.lines[2], "| x | 1 |");
        assert_eq!(s.lines[4], "|z|3|");
    }

    #[test]
    fn unknown_marker_stays_in_cell() {
        let text = "| a |\n|---|\n| {crimson} x |\n| {json} body |\n| x {red} |";
        let s = scan(text);
        assert_eq!(s.found[0].style.rows, vec![None, None, None]);
        assert_eq!(s.found[0].table.rows[0], vec!["{crimson} x"]);
        assert_eq!(s.lines.join("\n"), text);
    }

    #[test]
    fn marker_needs_a_space_and_text_after_it() {
        // A cell that is only a marker, or a marker glued to text, is data.
        let text = "| a | b |\n|---|---|\n| {red} | 1 |\n|{red}| 2 |\n| {red}x | 3 |\n| {#add}  |  4 |\n| {blue}";
        let s = scan(text);
        assert_eq!(s.found[0].style.rows, vec![None; 5]);
        assert_eq!(s.found[0].table.rows[0], vec!["{red}", "1"]);
        assert_eq!(s.lines.join("\n"), text);
        // Tab separator counts as a space.
        let s = scan("| a |\n|---|\n| {red}\tx |");
        assert!(s.found[0].style.rows[0].is_some());
        assert_eq!(s.found[0].table.rows[0], vec!["x"]);
    }

    #[test]
    fn tag_is_cancelled_by_text_before_a_table() {
        let light = "<openfang:table mode=\"light\"/>";
        // Blank lines between tag and table are fine.
        let s = scan(&format!("{light}\n\n| a |\n|---|"));
        assert_eq!(s.found[0].style.mode, Mode::Light);
        // Prose, a fence, or a pipe line that isn't a table cancels it.
        for between in ["some prose", "```\ncode\n```", "a | b"] {
            let s = scan(&format!("{light}\n{between}\n| a |\n|---|"));
            assert_eq!(s.found[0].style, Style::default(), "{between}");
            assert!(!s.lines.iter().any(|l| l.contains("openfang:table")));
        }
    }

    #[test]
    fn tag_styles_next_table_and_is_removed() {
        let s = scan(
            "hi\n<openfang:table mode=\"light\" header=\"grape\"/>\n| a |\n|---|\n\n| b |\n|---|",
        );
        assert_eq!(s.lines[1], "| a |");
        assert_eq!(s.found[0].style.mode, Mode::Light);
        assert_eq!(s.found[0].style.header, Some(Rgb(0x69, 0x2e, 0xc2)));
        // Used once: the second table gets defaults.
        assert_eq!(s.found[1].style, Style::default());
    }

    #[test]
    fn bad_tag_values_fall_back_to_defaults() {
        let s =
            scan("<openfang:table mode=\"sepia\" header='#zzz\") #eval(\"1\")' x/>\n| a |\n|---|");
        assert_eq!(s.lines[0], "| a |");
        assert_eq!(s.found[0].style.mode, Mode::Dark);
        assert_eq!(s.found[0].style.header, None);
        // Not tags: left as text.
        for line in [
            "<openfang:tables/>",
            "say <openfang:table/> inline",
            "<openfang:table>",
        ] {
            assert!(parse_tag(line).is_none(), "{line}");
        }
    }

    #[test]
    fn markup_in_code_fences_is_left_alone() {
        let text = "```\n<openfang:table mode=\"light\"/>\n| {red} a |\n|---|\n```";
        assert_eq!(strip_markup(text), text);
    }

    #[test]
    fn strip_markup_without_markup_is_identity() {
        assert_eq!(strip_markup(SAMPLE), SAMPLE);
        assert_eq!(
            strip_markup("a\n\n| x |\n|---|\n| {y} |\n"),
            "a\n\n| x |\n|---|\n| {y} |\n"
        );
    }

    #[test]
    fn tints_keep_contrast_in_both_modes() {
        for mode in [Mode::Dark, Mode::Light] {
            let theme = mode.theme();
            for &(name, v) in TODOIST_COLORS {
                let tint = theme.tint(Rgb::from_u32(v));
                assert!(
                    tint.contrast(theme.text) >= MIN_CONTRAST,
                    "{name} in {mode:?}"
                );
            }
            // Extremes still return a passing color.
            for c in [Rgb(255, 255, 0), Rgb(0, 0, 0), Rgb(255, 255, 255)] {
                assert!(theme.tint(c).contrast(theme.text) >= MIN_CONTRAST);
            }
        }
    }

    #[test]
    fn template_colors_are_rust_emitted_hex() {
        let t = Table {
            header: vec!["a".into()],
            align: vec![Align::Left],
            rows: vec![vec!["1".into()]; 3],
        };
        let style = Style {
            mode: Mode::Light,
            header: Some(Rgb(1, 2, 3)),
            rows: vec![Some(Rgb(0xdc, 0x4c, 0x3e)), None],
        };
        let d = template_data(&t, &style);
        assert_eq!(d["bars"], serde_json::json!([null, "#dc4c3e", null, null]));
        // Row 2 (y = 2) keeps the stripe; row 3 has no fill.
        assert_eq!(d["fills"][2], "#f2f3f5");
        assert!(d["fills"][3].is_null());
        let is_hex = |v: &serde_json::Value| {
            let s = v.as_str().unwrap();
            s.len() == 7 && s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_hexdigit())
        };
        for v in d["theme"].as_object().unwrap().values() {
            assert!(is_hex(v));
        }
        for v in d["fills"]
            .as_array()
            .unwrap()
            .iter()
            .chain(d["bars"].as_array().unwrap())
        {
            assert!(v.is_null() || is_hex(v));
        }
        // Default style reproduces the original dark palette.
        let d = template_data(&t, &Style::default());
        assert_eq!(d["theme"]["page"], "#313338");
        assert_eq!(
            d["fills"],
            serde_json::json!(["#2b2d31", null, "#383a40", null])
        );
    }

    #[tokio::test]
    async fn renders_styled_table() {
        if find_typst().is_none() {
            eprintln!("typst not installed; skipping");
            return;
        }
        let text = "<openfang:table mode=\"light\" header=\"sky_blue\"/>\n| a | b |\n|---|---|\n| {berry_red} x | 1 |\n| y | 2 |";
        let (text, files) = render_tables(text).await;
        assert_eq!(text, "📊 *Table 1 (image below)*");
        assert_eq!(files.len(), 1);
    }

    #[tokio::test]
    async fn no_tables_is_a_no_op() {
        let (text, files) = render_tables("just prose | with a pipe").await;
        assert_eq!(text, "just prose | with a pipe");
        assert!(files.is_empty());
    }

    /// End-to-end through the real typst binary, including hostile cells.
    /// Skips (passes) when typst isn't installed.
    #[tokio::test]
    async fn renders_png_and_replaces_table() {
        if find_typst().is_none() {
            eprintln!("typst not installed; skipping");
            return;
        }
        let hostile =
            "intro\n| a | b |\n|---|---|\n| #read(\"/etc/passwd\") | $x$ ] #eval(\"1+1\") |\nend";
        let (text, files) = render_tables(hostile).await;
        assert_eq!(text, "intro\n📊 *Table 1 (image below)*\nend");
        assert_eq!(files.len(), 1);
        match &files[0] {
            ChannelContent::FileData {
                data,
                filename,
                mime_type,
            } => {
                assert!(data.starts_with(b"\x89PNG"));
                assert_eq!(filename, "table-1.png");
                assert_eq!(mime_type, "image/png");
            }
            other => panic!("expected FileData, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn over_limit_table_stays_text() {
        let mut s = String::from("| a |\n|---|\n");
        for i in 0..=MAX_ROWS {
            s.push_str(&format!("| {i} |\n"));
        }
        let (text, files) = render_tables(&s).await;
        assert_eq!(text, s);
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn missing_or_broken_typst_is_an_error() {
        let t = Table {
            header: vec!["a".into()],
            align: vec![Align::Left],
            rows: vec![],
        };
        assert!(render_png(
            Path::new("/nonexistent/typst"),
            &t,
            &Style::default(),
            192,
            RENDER_TIMEOUT
        )
        .await
        .is_err());
        // A binary that exits non-zero.
        assert!(render_png(
            Path::new("/usr/bin/false"),
            &t,
            &Style::default(),
            192,
            RENDER_TIMEOUT
        )
        .await
        .is_err());
    }
}
