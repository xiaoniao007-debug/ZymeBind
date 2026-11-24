#!/usr/bin/env python3
"""Build a rigorous Nature-style Graphical Architecture Diagram for Figure 1.

Strict design & scientific rules:
  - Font: Arial across all labels.
  - Zero developer shorthands (no PerMod, h_fused, z_prot, EC4*, Mod-drop, Cross-KD, etc.).
  - Zero cluttered micro-footnotes; crisp, self-explanatory scientific terminology.
  - Panel c: Query enzyme & Novel EC prototype labels positioned cleanly outside the unit sphere.
  - Panel d: Four biologically faithful pure-vector enzymology schematics (catalytic pocket + SAE,
    unannotated sequence -> 3D fold, metabolic pathway hole completion, mutational landscape).
"""
from __future__ import annotations

import base64
from pathlib import Path

BASE = Path("/Users/chenchen/Paper/ZymeBind")
OUT_SVG = BASE / "figures/Fig1_zymebind_architecture.svg"
THUMB_3D = BASE / "structures/plate_6_chitinase_superpos_tight.png"
FONT = "Arial, Helvetica, sans-serif"


def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_aa_beads(x0: float, y0: float) -> str:
    aas = [
        ("M", "#2B5C8F"), ("K", "#4D908E"), ("V", "#6B9AC4"), ("S", "#D97A54"),
        ("T", "#D97A54"), ("F", "#C98B63"), ("K", "#4D908E"), ("L", "#6B9AC4"),
        ("G", "#7B8FA6"), ("A", "#2B5C8F"),
    ]
    out = [f'<line x1="{x0+5}" y1="{y0}" x2="{x0+5+(len(aas)-1)*12.4}" y2="{y0}" stroke="#94A3B8" stroke-width="1.2"/>']
    for i, j, h in [(1, 4, 8), (3, 6, 10), (0, 7, 13), (4, 8, 7)]:
        xi = x0 + 5 + i * 12.4
        xj = x0 + 5 + j * 12.4
        xm = 0.5 * (xi + xj)
        out.append(f'<path d="M {xi:.1f} {y0-5} Q {xm:.1f} {y0-5-h} {xj:.1f} {y0-5}" fill="none" stroke="#2B5C8F" stroke-width="0.65" opacity="0.55"/>')
    for idx, (aa, col) in enumerate(aas):
        cx = x0 + 5 + idx * 12.4
        out.append(f'<circle cx="{cx:.1f}" cy="{y0}" r="4.8" fill="{col}" stroke="#FFFFFF" stroke-width="0.7"/>')
        out.append(f'<text x="{cx:.1f}" y="{y0+1.8}" text-anchor="middle" font-family="{FONT}" font-size="4.8" font-weight="700" fill="#FFFFFF">{aa}</text>')
    return "\n  ".join(out)


def build_3di_strip(x0: float, y0: float) -> str:
    toks = [("d", "#3B7A78"), ("v", "#66A5A3"), ("p", "#4D908E"), ("p", "#4D908E"),
            ("q", "#7BB3B1"), ("s", "#4D908E"), ("c", "#2F6664"), ("a", "#3B7A78")]
    out = []
    for idx, (t, col) in enumerate(toks):
        x = x0 + idx * 10.8
        out.append(f'<rect x="{x:.1f}" y="{y0}" width="9.4" height="9.4" rx="1.8" fill="{col}" opacity="0.92"/>')
        out.append(f'<text x="{x+4.7:.1f}" y="{y0+6.6}" text-anchor="middle" font-family="{FONT}" font-size="5.0" font-weight="700" fill="#FFFFFF">{t}</text>')
    return "\n  ".join(out)


def build_domain_track(x0: float, y0: float) -> str:
    return "\n  ".join([
        f'<line x1="{x0}" y1="{y0+5.2}" x2="{x0+130}" y2="{y0+5.2}" stroke="#64748B" stroke-width="1.6" stroke-linecap="round"/>',
        f'<rect x="{x0+8}" y="{y0}" width="38" height="10.4" rx="5.2" fill="#6B9AC4" stroke="#2B5C8F" stroke-width="0.6"/>',
        f'<text x="{x0+27}" y="{y0+7.1}" text-anchor="middle" font-family="{FONT}" font-size="4.8" font-weight="700" fill="#FFFFFF">PF00144</text>',
        f'<rect x="{x0+54}" y="{y0}" width="44" height="10.4" rx="5.2" fill="#2B5C8F" stroke="#1E3A5F" stroke-width="0.6"/>',
        f'<text x="{x0+76}" y="{y0+7.1}" text-anchor="middle" font-family="{FONT}" font-size="4.8" font-weight="700" fill="#FFFFFF">IPR012338</text>',
        f'<rect x="{x0+105}" y="{y0}" width="20" height="10.4" rx="5.2" fill="#C6D9EC" stroke="#2B5C8F" stroke-width="0.6"/>',
        f'<text x="{x0+115}" y="{y0+7.1}" text-anchor="middle" font-family="{FONT}" font-size="4.6" font-weight="700" fill="#1E3A5F">Site</text>',
    ])


def build_attn_matrix(x0: float, y0: float) -> str:
    labels = ["CLS", "Seq", "Struct", "Text", "Dom"]
    active = [True, True, True, False, False]
    out = []
    cell = 11.4
    gap = 1.8
    for j, lb in enumerate(labels):
        cx = x0 + 21 + j * (cell + gap) + cell / 2
        col = "#1E293B" if active[j] else "#94A3B8"
        out.append(f'<text x="{cx:.1f}" y="{y0-3.2}" text-anchor="middle" font-family="{FONT}" font-size="4.6" font-weight="700" fill="{col}">{lb}</text>')
    for i, lb in enumerate(labels):
        ry = y0 + i * (cell + gap)
        col = "#1E293B" if active[i] else "#94A3B8"
        out.append(f'<text x="{x0+17}" y="{ry+7.6:.1f}" text-anchor="end" font-family="{FONT}" font-size="4.6" font-weight="700" fill="{col}">{lb}</text>')
        for j in range(5):
            rx = x0 + 21 + j * (cell + gap)
            if active[i] and active[j]:
                shades = [
                    ["#1E3A5F", "#2B5C8F", "#4D908E"],
                    ["#2B5C8F", "#1E3A5F", "#5BA19F"],
                    ["#4D908E", "#5BA19F", "#2F6664"],
                ]
                out.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{cell}" height="{cell}" rx="1.8" fill="{shades[i][j]}"/>')
            else:
                out.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{cell}" height="{cell}" rx="1.8" fill="#F1F5F9" stroke="#CBD5E1" stroke-width="0.5" stroke-dasharray="1.8,1.2"/>')
                out.append(f'<line x1="{rx+2.6:.1f}" y1="{ry+2.6:.1f}" x2="{rx+cell-2.6:.1f}" y2="{ry+cell-2.6:.1f}" stroke="#CBD5E1" stroke-width="0.6"/>')
    return "\n  ".join(out)


def build_ec_tree_and_sphere(x0: float, y0: float) -> str:
    """Left: EC hierarchy tree. Right: Cosine embedding sphere with non-overlapping external callouts."""
    out = []
    nodes = [
        (x0 + 40, y0 + 10, "EC 1",    "#1E3A5F", 7.8),
        (x0 + 18, y0 + 32, "1.1",     "#2B5C8F", 6.6),
        (x0 + 62, y0 + 32, "1.14",    "#2B5C8F", 6.6),
        (x0 + 10, y0 + 54, "1.1.1",   "#4D908E", 6.4),
        (x0 + 33, y0 + 54, "1.1.3",   "#94A3B8", 5.8),
        (x0 + 62, y0 + 54, "1.14.14", "#4D908E", 8.2),
        (x0 + 10, y0 + 76, "305",     "#6B9AC4", 6.4),
        (x0 + 62, y0 + 76, "157*",    "#D97A54", 7.0),
    ]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (5, 7)]
    for u, v in edges:
        xu, yu = nodes[u][0], nodes[u][1]
        xv, yv = nodes[v][0], nodes[v][1]
        st = 'stroke="#D97A54" stroke-dasharray="2.2,1.4"' if v == 7 else 'stroke="#94A3B8"'
        out.append(f'<line x1="{xu}" y1="{yu}" x2="{xv}" y2="{yv}" {st} stroke-width="1.0"/>')
    for cx, cy, txt, col, r in nodes:
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{col}" stroke="#FFFFFF" stroke-width="0.8"/>')
        fs = "4.4" if len(txt) >= 6 else "4.9"
        out.append(f'<text x="{cx}" y="{cy+1.7}" text-anchor="middle" font-family="{FONT}" font-size="{fs}" font-weight="700" fill="#FFFFFF">{txt}</text>')
    out.append(f'<text x="{x0+38}" y="{y0+96}" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">EC Reaction Ontology</text>')

    # Connecting arrow
    out.append(f'<line x1="{x0+82}" y1="{y0+45}" x2="{x0+104}" y2="{y0+45}" stroke="#2B5C8F" stroke-width="1.15" marker-end="url(#arr-blue)"/>')

    # Cosine embedding sphere (shifted slightly left so external labels sit cleanly in white space!)
    sx, sy, sr = x0 + 138, y0 + 44, 29
    out.append(f'<circle cx="{sx}" cy="{sy}" r="{sr}" fill="url(#grad-sphere)" stroke="#2B5C8F" stroke-width="0.85" stroke-dasharray="2.6,1.6"/>')
    out.append(f'<ellipse cx="{sx}" cy="{sy}" rx="{sr}" ry="8.5" fill="none" stroke="#94A3B8" stroke-width="0.6" opacity="0.65"/>')
    # Vectors from sphere center
    qx, qy = sx + 16, sy - 18   # Query enzyme tip inside/on sphere
    ex, ey = sx + 21, sy - 7    # Unseen EC prototype tip
    out.append(f'<line x1="{sx}" y1="{sy}" x2="{qx}" y2="{qy}" stroke="#4D908E" stroke-width="1.5" marker-end="url(#arr-teal)"/>')
    out.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="#D97A54" stroke-width="1.5" marker-end="url(#arr-terra)"/>')
    out.append(f'<circle cx="{sx-14}" cy="{sy+11}" r="2.6" fill="#94A3B8"/>')
    out.append(f'<circle cx="{sx-7}"  cy="{sy-13}" r="2.6" fill="#94A3B8"/>')
    out.append(f'<circle cx="{sx}" cy="{sy}" r="2.2" fill="#1E293B"/>')
    # Crisp leader lines to external labels completely outside the dashed circle (x >= sx + sr + 6 = x0 + 173)
    lx = sx + sr + 7
    out.append(f'<polyline points="{qx+1.5},{qy-1} {sx+sr+2},{qy-4} {lx},{qy-4}" fill="none" stroke="#4D908E" stroke-width="0.75"/>')
    out.append(f'<text x="{lx+2}" y="{qy-2.2}" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#2F6664">Query enzyme</text>')
    out.append(f'<polyline points="{ex+1.5},{ey} {sx+sr+2},{ey+4} {lx},{ey+4}" fill="none" stroke="#D97A54" stroke-width="0.75"/>')
    out.append(f'<text x="{lx+2}" y="{ey+5.8}" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#B85D3B">Novel EC*</text>')
    out.append(f'<text x="{sx+12}" y="{y0+96}" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">Shared Cosine Space</text>')
    return "\n  ".join(out)


def build_panel_d_schematics() -> str:
    """Four biologically faithful, clean pure-vector enzymology schematics in Panel d."""
    return f"""
  <!-- Card 1: Multi-level interpretability (Catalytic pocket cleft + sparse feature bars) -->
  <rect x="269" y="283" width="115" height="57" fill="#FFFFFF" stroke="#B8D4D2" stroke-width="0.8" rx="4"/>
  <!-- Left: Enzyme binding cleft cradling substrate diamond + 2 catalytic pins -->
  <path d="M 279 294 C 279 314, 291 317, 297 308 C 301 302, 305 302, 309 308 C 315 317, 325 314, 325 294" fill="#EAF4F3" stroke="#4D908E" stroke-width="1.2" stroke-linecap="round"/>
  <polygon points="302,293 308,299 302,305 296,299" fill="#D97A54" stroke="#FFFFFF" stroke-width="0.6"/>
  <circle cx="292" cy="305" r="2.3" fill="#00C4CC"/>
  <circle cx="312" cy="305" r="2.3" fill="#00C4CC"/>
  <line x1="329" y1="304" x2="340" y2="304" stroke="#94A3B8" stroke-width="0.85" marker-end="url(#arr-dark)"/>
  <!-- Right: Sparse monosemantic SAE neuron activation bars -->
  <line x1="344" y1="315" x2="376" y2="315" stroke="#64748B" stroke-width="0.8"/>
  <rect x="346" y="311" width="4.2" height="4"  fill="#CBD5E1" rx="0.7"/>
  <rect x="352" y="293" width="4.2" height="22" fill="#2B5C8F" rx="0.7"/>
  <rect x="358" y="312" width="4.2" height="3"  fill="#CBD5E1" rx="0.7"/>
  <rect x="364" y="298" width="4.2" height="17" fill="#4D908E" rx="0.7"/>
  <rect x="370" y="311" width="4.2" height="4"  fill="#CBD5E1" rx="0.7"/>
  <text x="326.5" y="331" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">Mechanistic interpretability</text>

  <!-- Card 2: Dark-proteome discovery (Unannotated sequence strands -> 3D fold + EC assignment) -->
  <rect x="389" y="283" width="114" height="57" fill="#FFFFFF" stroke="#E5C2B3" stroke-width="0.8" rx="4"/>
  <!-- Left: Unannotated metagenomic sequence waves with '?' badge -->
  <path d="M 398 297 Q 405 292, 412 297 T 426 297" fill="none" stroke="#8D99AE" stroke-width="1.25" stroke-linecap="round"/>
  <path d="M 398 304 Q 405 299, 412 304 T 426 304" fill="none" stroke="#2B5C8F" stroke-width="1.35" stroke-linecap="round"/>
  <path d="M 398 311 Q 405 306, 412 311 T 426 311" fill="none" stroke="#8D99AE" stroke-width="1.25" stroke-linecap="round"/>
  <circle cx="429" cy="296" r="4.5" fill="#FAF1EC" stroke="#D97A54" stroke-width="0.8"/>
  <text x="429" y="298.2" text-anchor="middle" font-family="{FONT}" font-size="5.2" font-weight="700" fill="#B85D3B">?</text>
  <line x1="436" y1="304" x2="449" y2="304" stroke="#94A3B8" stroke-width="0.85" marker-end="url(#arr-dark)"/>
  <!-- Right: Stylized 3D alpha/beta enzyme fold ribbon + active ligand -->
  <rect x="453" y="291" width="40" height="26" rx="4" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="0.8"/>
  <path d="M 459 311 C 461 295, 467 295, 469 311 C 471 295, 477 295, 479 311 L 487 298" fill="none" stroke="#2B5C8F" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="474" cy="303" r="2.8" fill="#D97A54"/>
  <text x="446" y="331" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">Dark-proteome discovery</text>

  <!-- Card 3: Metabolic pathway completion -->
  <rect x="269" y="345" width="115" height="57" fill="#FFFFFF" stroke="#C6D9EC" stroke-width="0.8" rx="4"/>
  <!-- Metabolite nodes S1 to S2 with newly filled enzymatic gap to S3 and S4 -->
  <circle cx="285" cy="366" r="5.0" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="1.0"/>
  <line x1="291" y1="366" x2="306" y2="366" stroke="#2B5C8F" stroke-width="1.1" marker-end="url(#arr-blue)"/>
  <circle cx="313" cy="366" r="5.0" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="1.0"/>
  <!-- Highlighted newly filled pathway hole step! -->
  <rect x="322" y="356" width="24" height="19" rx="3" fill="#FAF1EC" stroke="#D97A54" stroke-width="0.85" stroke-dasharray="2.0,1.2"/>
  <line x1="319" y1="366" x2="349" y2="366" stroke="#D97A54" stroke-width="1.4" marker-end="url(#arr-terra)"/>
  <circle cx="334" cy="366" r="3.4" fill="#D97A54"/>
  <circle cx="356" cy="366" r="5.0" fill="#EAF4F3" stroke="#4D908E" stroke-width="1.0"/>
  <line x1="356" y1="360" x2="369" y2="355" stroke="#4D908E" stroke-width="0.95"/>
  <circle cx="372" cy="354" r="3.6" fill="#FFFFFF" stroke="#4D908E" stroke-width="0.9"/>
  <text x="326.5" y="393" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">Metabolic pathway completion</text>

  <!-- Card 4: Mutational fitness landscape (Residue track + catalytic DMS sensitivity spikes) -->
  <rect x="389" y="345" width="114" height="57" fill="#FFFFFF" stroke="#E5C2B3" stroke-width="0.8" rx="4"/>
  <!-- Sequence baseline + residue mutational sensitivity profile -->
  <line x1="401" y1="376" x2="491" y2="376" stroke="#64748B" stroke-width="1.1"/>
  <!-- Background tolerant positions -->
  <line x1="408" y1="376" x2="408" y2="370" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="416" y1="376" x2="416" y2="368" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="432" y1="376" x2="432" y2="371" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="440" y1="376" x2="440" y2="369" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="456" y1="376" x2="456" y2="370" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="472" y1="376" x2="472" y2="369" stroke="#94A3B8" stroke-width="1.2"/>
  <line x1="484" y1="376" x2="484" y2="371" stroke="#94A3B8" stroke-width="1.2"/>
  <!-- Sharp catalytic active-site sensitivity spikes -->
  <line x1="424" y1="376" x2="424" y2="355" stroke="#D97A54" stroke-width="1.6"/>
  <circle cx="424" cy="355" r="2.5" fill="#00C4CC" stroke="#1E3A5F" stroke-width="0.5"/>
  <line x1="448" y1="376" x2="448" y2="353" stroke="#D97A54" stroke-width="1.6"/>
  <circle cx="448" cy="353" r="2.5" fill="#00C4CC" stroke="#1E3A5F" stroke-width="0.5"/>
  <line x1="464" y1="376" x2="464" y2="357" stroke="#2B5C8F" stroke-width="1.5"/>
  <circle cx="464" cy="357" r="2.3" fill="#2B5C8F"/>
  <text x="446" y="393" text-anchor="middle" font-family="{FONT}" font-size="5.4" font-weight="700" fill="#1E293B">Mutational fitness landscape</text>
"""


def main() -> None:
    b64_3d = img_to_b64(THUMB_3D)
    aa_svg = build_aa_beads(22, 57)
    di_svg = build_3di_strip(72, 123)
    dom_svg = build_domain_track(22, 219)
    attn_svg = build_attn_matrix(270, 62)
    ec_svg = build_ec_tree_and_sphere(16, 288)
    d_svg = build_panel_d_schematics()

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"
     width="183mm" height="148mm" viewBox="0 0 518.74 419.53">
  <defs>
    <marker id="arr-dark" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#1E293B"/>
    </marker>
    <marker id="arr-blue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#2B5C8F"/>
    </marker>
    <marker id="arr-teal" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#4D908E"/>
    </marker>
    <marker id="arr-terra" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="4.8" markerHeight="4.8" orient="auto">
      <path d="M 0 1.5 L 9 5 L 0 8.5 z" fill="#D97A54"/>
    </marker>
    <linearGradient id="grad-enc" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#2B5C8F"/>
      <stop offset="100%" stop-color="#1E3A5F"/>
    </linearGradient>
    <radialGradient id="grad-sphere" cx="38%" cy="35%" r="65%">
      <stop offset="0%" stop-color="#FFFFFF"/>
      <stop offset="70%" stop-color="#EEF5FB"/>
      <stop offset="100%" stop-color="#D7E6F5"/>
    </radialGradient>
  </defs>

  <rect x="0" y="0" width="518.74" height="419.53" fill="#FFFFFF"/>

  <!-- ==================== PANEL a ==================== -->
  <text x="8" y="15" font-family="{FONT}" font-size="9" font-weight="700" fill="#111111">a</text>
  <rect x="10" y="21" width="166" height="226" fill="#FAFBFD" stroke="#CBD5E1" stroke-width="0.7" rx="5"/>

  <!-- 1. Sequence Representation -->
  <rect x="16" y="28" width="154" height="62" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="0.85" rx="4"/>
  <text x="22" y="39.5" font-family="{FONT}" font-size="5.9" font-weight="700" fill="#1E3A5F">Sequence Representation</text>
  {aa_svg}
  <rect x="22" y="67" width="68" height="17" fill="#2B5C8F" rx="2.5"/>
  <text x="56" y="77.5" text-anchor="middle" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#FFFFFF">ESM-2 (650M)</text>
  <rect x="96" y="67" width="68" height="17" fill="#1E3A5F" rx="2.5"/>
  <text x="130" y="77.5" text-anchor="middle" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#FFFFFF">ProstT5 (3B)</text>

  <!-- 2. Structural Representation -->
  <rect x="16" y="96" width="154" height="50" fill="#EAF4F3" stroke="#4D908E" stroke-width="0.85" rx="4"/>
  <rect x="21" y="101" width="45" height="40" fill="#FFFFFF" stroke="#B8D4D2" stroke-width="0.6" rx="3"/>
  <image x="22.5" y="102.5" width="42" height="37" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/png;base64,{b64_3d}"/>
  <text x="72" y="109" font-family="{FONT}" font-size="5.9" font-weight="700" fill="#2F6664">Structural Representation</text>
  <text x="72" y="118.5" font-family="{FONT}" font-size="5.0" fill="#334155">Foldseek 3Di tokens</text>
  {di_svg}

  <!-- 3. Catalytic Reaction Text -->
  <rect x="16" y="152" width="154" height="40" fill="#FAF1EC" stroke="#D97A54" stroke-width="0.8" rx="4"/>
  <rect x="22" y="158" width="24" height="28" fill="#FFFFFF" stroke="#D97A54" stroke-width="0.75" rx="2.5"/>
  <line x1="26" y1="165" x2="42" y2="165" stroke="#D97A54" stroke-width="0.9"/>
  <line x1="26" y1="170" x2="39" y2="170" stroke="#CBD5E1" stroke-width="0.9"/>
  <text x="34" y="181" text-anchor="middle" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#B85D3B">S→P</text>
  <text x="53" y="168" font-family="{FONT}" font-size="5.9" font-weight="700" fill="#9A482A">Catalytic Reaction Text</text>
  <text x="53" y="180" font-family="{FONT}" font-size="5.1" fill="#334155">BioBERT semantic encoder</text>

  <!-- 4. Domain Signatures -->
  <rect x="16" y="198" width="154" height="41" fill="#F4F6FA" stroke="#7B8FA6" stroke-width="0.8" rx="4"/>
  <text x="22" y="211" font-family="{FONT}" font-size="5.9" font-weight="700" fill="#1E3A5F">Domain Signatures</text>
  {dom_svg}

  <!-- Input arrows -->
  <line x1="176" y1="59"  x2="193" y2="59"  stroke="#2B5C8F" stroke-width="1.15" marker-end="url(#arr-blue)"/>
  <line x1="176" y1="121" x2="193" y2="121" stroke="#4D908E" stroke-width="1.15" marker-end="url(#arr-teal)"/>
  <line x1="176" y1="172" x2="193" y2="172" stroke="#94A3B8" stroke-width="1.0" stroke-dasharray="2.4,1.6"/>
  <line x1="176" y1="218" x2="193" y2="218" stroke="#94A3B8" stroke-width="1.0" stroke-dasharray="2.4,1.6"/>

  <!-- ==================== PANEL b ==================== -->
  <text x="187" y="15" font-family="{FONT}" font-size="9" font-weight="700" fill="#111111">b</text>
  <rect x="189" y="21" width="321" height="226" fill="#FAFBFD" stroke="#CBD5E1" stroke-width="0.7" rx="5"/>

  <!-- Column 1: Structured Modality Masking -->
  <rect x="195" y="30" width="62" height="206" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="0.85" rx="4"/>
  <text x="226" y="42" text-anchor="middle" font-family="{FONT}" font-size="5.5" font-weight="700" fill="#1E3A5F">Structured</text>
  <text x="226" y="50" text-anchor="middle" font-family="{FONT}" font-size="5.5" font-weight="700" fill="#1E3A5F">Masking</text>

  <rect x="203" y="59" width="46" height="16" fill="#1E293B" rx="2.5"/>
  <text x="226" y="69.2" text-anchor="middle" font-family="{FONT}" font-size="5.2" font-weight="700" fill="#FFFFFF">[CLS]</text>
  <rect x="203" y="81" width="46" height="16" fill="#2B5C8F" rx="2.5"/>
  <text x="226" y="91.2" text-anchor="middle" font-family="{FONT}" font-size="5.2" font-weight="700" fill="#FFFFFF">Sequence</text>
  <rect x="203" y="103" width="46" height="16" fill="#4D908E" rx="2.5"/>
  <text x="226" y="113.2" text-anchor="middle" font-family="{FONT}" font-size="5.2" font-weight="700" fill="#FFFFFF">Structure</text>
  <rect x="203" y="125" width="46" height="16" fill="#F8FAFC" stroke="#94A3B8" stroke-width="0.75" stroke-dasharray="2.2,1.4" rx="2.5"/>
  <text x="226" y="135.2" text-anchor="middle" font-family="{FONT}" font-size="5.0" fill="#64748B">Text (masked)</text>
  <rect x="203" y="147" width="46" height="16" fill="#F8FAFC" stroke="#94A3B8" stroke-width="0.75" stroke-dasharray="2.2,1.4" rx="2.5"/>
  <text x="226" y="157.2" text-anchor="middle" font-family="{FONT}" font-size="5.0" fill="#64748B">Dom (masked)</text>

  <path d="M 226 174 L 241 180 L 241 193 C 241 202, 226 208, 226 208 C 226 208, 211 202, 211 193 L 211 180 Z" fill="#FAF1EC" stroke="#D97A54" stroke-width="0.9"/>
  <text x="226" y="193.5" text-anchor="middle" font-family="{FONT}" font-size="5.3" font-weight="700" fill="#B85D3B">ASL</text>
  <text x="226" y="219" text-anchor="middle" font-family="{FONT}" font-size="5.0" font-weight="700" fill="#1E293B">+ Cross-Modal</text>
  <text x="226" y="227" text-anchor="middle" font-family="{FONT}" font-size="5.0" font-weight="700" fill="#1E293B">Distillation</text>

  <line x1="257" y1="112" x2="271" y2="112" stroke="#1E293B" stroke-width="1.15" marker-end="url(#arr-dark)"/>

  <!-- Column 2: Presence-Aware Transformer Encoder -->
  <rect x="273" y="30" width="106" height="206" fill="#FFFFFF" stroke="#2B5C8F" stroke-width="0.95" rx="4"/>
  <rect x="273" y="30" width="106" height="22" fill="url(#grad-enc)" rx="4"/>
  <rect x="273" y="44" width="106" height="8" fill="#1E3A5F"/>
  <text x="326" y="43.5" text-anchor="middle" font-family="{FONT}" font-size="5.6" font-weight="700" fill="#FFFFFF">Presence-Aware Transformer</text>

  {attn_svg}
  <text x="326" y="136.5" text-anchor="middle" font-family="{FONT}" font-size="4.9" font-weight="700" fill="#475569">Masked Self-Attention</text>

  <!-- 3-Layer Encoder Stack -->
  <rect x="285" y="146" width="88" height="44" rx="3" fill="#F8FAFC" stroke="#CBD5E1" stroke-width="0.6"/>
  <rect x="282" y="149" width="88" height="44" rx="3" fill="#F1F5F9" stroke="#94A3B8" stroke-width="0.65"/>
  <rect x="279" y="152" width="88" height="44" rx="3" fill="#FFFFFF" stroke="#2B5C8F" stroke-width="0.8"/>
  <rect x="284" y="157" width="78" height="10" rx="2" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="0.45"/>
  <text x="323" y="163.8" text-anchor="middle" font-family="{FONT}" font-size="4.7" font-weight="700" fill="#1E3A5F">Multi-Head Attention (8H)</text>
  <rect x="284" y="170" width="78" height="10" rx="2" fill="#EAF4F3" stroke="#4D908E" stroke-width="0.45"/>
  <text x="323" y="176.8" text-anchor="middle" font-family="{FONT}" font-size="4.7" font-weight="700" fill="#2F6664">Feed-Forward Network</text>
  <text x="323" y="190" text-anchor="middle" font-family="{FONT}" font-size="4.9" font-weight="700" fill="#2B5C8F">×3 Transformer Layers</text>

  <text x="326" y="213" text-anchor="middle" font-family="{FONT}" font-size="5.1" font-weight="700" fill="#1E293B">Joint + Modality Tokens</text>
  <text x="326" y="224" text-anchor="middle" font-family="{FONT}" font-size="4.9" fill="#64748B">768-dim latent space</text>

  <!-- Arrows Encoder -> Output Heads -->
  <line x1="379" y1="64"  x2="394" y2="64"  stroke="#D97A54" stroke-width="1.05" marker-end="url(#arr-terra)"/>
  <line x1="379" y1="125" x2="394" y2="125" stroke="#4D908E" stroke-width="1.05" marker-end="url(#arr-teal)"/>

  <!-- Column 3: Joint Multimodal Head + Modality-Specific Auxiliary Heads + Presence-Weighted Ensemble -->
  <rect x="396" y="30" width="107" height="54" fill="#FAF1EC" stroke="#D97A54" stroke-width="0.85" rx="4"/>
  <text x="449.5" y="42.5" text-anchor="middle" font-family="{FONT}" font-size="5.7" font-weight="700" fill="#9A482A">Joint Multimodal Head</text>
  <rect x="410" y="50" width="8" height="16" fill="#CBD5E1" rx="1"/>
  <rect x="422" y="56" width="8" height="10" fill="#CBD5E1" rx="1"/>
  <rect x="434" y="47" width="8" height="19" fill="#D97A54" rx="1"/>
  <rect x="446" y="58" width="8" height="8"  fill="#CBD5E1" rx="1"/>
  <rect x="458" y="53" width="8" height="13" fill="#CBD5E1" rx="1"/>
  <rect x="470" y="59" width="8" height="7"  fill="#CBD5E1" rx="1"/>
  <rect x="482" y="55" width="8" height="11" fill="#CBD5E1" rx="1"/>
  <text x="449.5" y="76.5" text-anchor="middle" font-family="{FONT}" font-size="5.0" fill="#334155">5,855 EC4 classes</text>

  <rect x="396" y="94" width="107" height="58" fill="#EAF4F3" stroke="#4D908E" stroke-width="0.85" rx="4"/>
  <text x="449.5" y="106" text-anchor="middle" font-family="{FONT}" font-size="5.5" font-weight="700" fill="#2F6664">Modality-Specific</text>
  <text x="449.5" y="114" text-anchor="middle" font-family="{FONT}" font-size="5.5" font-weight="700" fill="#2F6664">Auxiliary Heads</text>
  <rect x="401" y="121" width="22" height="13" fill="#2B5C8F" rx="2"/>
  <text x="412" y="129.4" text-anchor="middle" font-family="{FONT}" font-size="4.5" font-weight="700" fill="#FFFFFF">Seq</text>
  <rect x="426" y="121" width="22" height="13" fill="#4D908E" rx="2"/>
  <text x="437" y="129.4" text-anchor="middle" font-family="{FONT}" font-size="4.5" font-weight="700" fill="#FFFFFF">Struct</text>
  <rect x="451" y="121" width="22" height="13" fill="#D97A54" rx="2"/>
  <text x="462" y="129.4" text-anchor="middle" font-family="{FONT}" font-size="4.5" font-weight="700" fill="#FFFFFF">Text</text>
  <rect x="476" y="121" width="22" height="13" fill="#7B8FA6" rx="2"/>
  <text x="487" y="129.4" text-anchor="middle" font-family="{FONT}" font-size="4.5" font-weight="700" fill="#FFFFFF">Dom</text>
  <text x="449.5" y="145" text-anchor="middle" font-family="{FONT}" font-size="4.9" fill="#334155">Active-modality supervision</text>

  <line x1="449.5" y1="152" x2="449.5" y2="165" stroke="#1E293B" stroke-width="1.05" marker-end="url(#arr-dark)"/>

  <!-- Presence-Weighted Ensemble -->
  <rect x="396" y="167" width="107" height="69" fill="#EEF5FB" stroke="#2B5C8F" stroke-width="0.95" rx="4"/>
  <text x="449.5" y="179.5" text-anchor="middle" font-family="{FONT}" font-size="5.6" font-weight="700" fill="#1E3A5F">Presence-Weighted</text>
  <text x="449.5" y="187.5" text-anchor="middle" font-family="{FONT}" font-size="5.6" font-weight="700" fill="#1E3A5F">Ensemble Prediction</text>
  <rect x="403" y="194" width="33" height="32" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="0.65" rx="2"/>
  <rect x="406" y="216" width="5" height="8"  fill="#CBD5E1" rx="0.8"/>
  <rect x="413" y="200" width="5" height="24" fill="#4D908E" rx="0.8"/>
  <rect x="420" y="211" width="5" height="13" fill="#CBD5E1" rx="0.8"/>
  <rect x="427" y="206" width="5" height="18" fill="#D97A54" rx="0.8"/>
  <line x1="404" y1="208" x2="435" y2="208" stroke="#D97A54" stroke-width="0.8" stroke-dasharray="1.8,1.2"/>
  <text x="441" y="205" font-family="{FONT}" font-size="5.2" font-weight="700" fill="#1E3A5F">Calibrated P(EC)</text>
  <text x="441" y="217" font-family="{FONT}" font-size="5.0" font-weight="700" fill="#B85D3B">τ = 0.62</text>

  <!-- ==================== PANEL c ==================== -->
  <text x="8" y="269" font-family="{FONT}" font-size="9" font-weight="700" fill="#111111">c</text>
  <rect x="10" y="275" width="242" height="134" fill="#FAFBFD" stroke="#CBD5E1" stroke-width="0.7" rx="5"/>
  {ec_svg}

  <!-- ==================== PANEL d ==================== -->
  <text x="260" y="269" font-family="{FONT}" font-size="9" font-weight="700" fill="#111111">d</text>
  <rect x="262" y="275" width="248" height="134" fill="#FAFBFD" stroke="#CBD5E1" stroke-width="0.7" rx="5"/>
  {d_svg}
</svg>
"""
    OUT_SVG.write_text(svg, encoding="utf-8")
    print(f"[BUILT] {OUT_SVG} ({len(svg)} bytes)")


if __name__ == "__main__":
    main()
