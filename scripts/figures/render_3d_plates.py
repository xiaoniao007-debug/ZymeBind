#!/usr/bin/env python
"""Headless PyMOL ray-traced 3D publication structure plates for Fig. 6 and Fig. 8.

Produces 4 crisp, high-resolution (2000x1500, transparent/white background, ambient occlusion)
3D cartoon + ligand/active-site renders:
  1. plate_6_chitinase_superpos.png : AF-A0AAD7U9D2 (pid=0%) superposed on 4MB5 Chitinase + Nag5 substrate
  2. plate_6_gst_superpos.png       : AF-A0AAD7ULU7 (pid=0%) superposed on 6SSW MGST2 + GSH ligand
  3. plate_8_B2TW38_arnA.png        : AF-B2TW38 (Bifunctional ArnA) with catalytic residues 300, 485
  4. plate_8_Q9HPK9_archaeon.png    : AF-Q9HPK9 (pid=0% extremophilic archaeon, pLDDT=93.1)
"""
from __future__ import annotations

import json
from pathlib import Path
from pymol import cmd


def setup_pub_style():
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_shadows", 0)
    cmd.set("ambient", 0.45)
    cmd.set("direct", 0.55)
    cmd.set("specular", 0.20)
    cmd.set("shininess", 15)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_oval_width", 0.25)
    cmd.set("cartoon_loop_radius", 0.18)
    cmd.set("stick_radius", 0.22)
    cmd.set("depth_cue", 0)


def main():
    base = Path("/mnt/data/cs/ZymeBind_sota_baselines/pymol_renders")
    metrics = {}

    # ---- 1. A0AAD7U9D2 (pid=0%) vs 4MB5 Chitinase + Nag5 ----
    setup_pub_style()
    cmd.load(str(base / "4MB5.pdb"), "ref_4mb5")
    cmd.load(str(base / "AF-A0AAD7U9D2.pdb"), "dark_u9d2")
    cmd.remove("ref_4mb5 and not chain A")
    cmd.remove("solvent")
    res = cmd.cealign("ref_4mb5 and polymer.protein", "dark_u9d2 and polymer.protein")
    metrics["A0AAD7U9D2_vs_4MB5"] = res
    print("[CEALIGN A0AAD7U9D2 vs 4MB5]", res, flush=True)

    cmd.hide("everything")
    cmd.show("cartoon", "ref_4mb5 and polymer.protein")
    cmd.show("cartoon", "dark_u9d2 and polymer.protein")
    cmd.set_color("c_gray", [0.78, 0.82, 0.86])
    cmd.set_color("c_teal", [0.106, 0.541, 0.478])
    cmd.set_color("c_coral", [0.839, 0.271, 0.255])
    cmd.color("c_gray", "ref_4mb5 and polymer.protein")
    cmd.set("cartoon_transparency", 0.45, "ref_4mb5")
    cmd.color("c_teal", "dark_u9d2 and polymer.protein")

    # Show Nag5 pentasaccharide ligand in Coral Red sticks
    cmd.select("lig_nag", "ref_4mb5 and (resn NAG or organic)")
    cmd.show("sticks", "lig_nag")
    cmd.color("c_coral", "lig_nag")
    cmd.util.cnc("lig_nag")
    cmd.color("c_coral", "lig_nag and elem C")

    cmd.orient("dark_u9d2")
    cmd.zoom("dark_u9d2", buffer=3)
    cmd.ray(1800, 1400)
    cmd.png(str(base / "plate_6_chitinase_superpos.png"), dpi=300)

    # ---- 2. A0AAD7ULU7 (pid=0%) vs 6SSW MGST2 + Glutathione ----
    setup_pub_style()
    cmd.load(str(base / "6SSW.pdb"), "ref_6ssw")
    cmd.load(str(base / "AF-A0AAD7ULU7.pdb"), "dark_ulu7")
    cmd.remove("ref_6ssw and not chain C")
    cmd.remove("solvent")
    res2 = cmd.cealign("ref_6ssw and polymer.protein", "dark_ulu7 and polymer.protein")
    metrics["A0AAD7ULU7_vs_6SSW"] = res2
    print("[CEALIGN A0AAD7ULU7 vs 6SSW]", res2, flush=True)

    cmd.hide("everything")
    cmd.show("cartoon", "ref_6ssw and polymer.protein")
    cmd.show("cartoon", "dark_ulu7 and polymer.protein")
    cmd.set_color("c_gray", [0.78, 0.82, 0.86])
    cmd.set_color("c_navy", [0.180, 0.357, 0.533])
    cmd.set_color("c_coral", [0.839, 0.271, 0.255])
    cmd.color("c_gray", "ref_6ssw and polymer.protein")
    cmd.set("cartoon_transparency", 0.45, "ref_6ssw")
    cmd.color("c_navy", "dark_ulu7 and polymer.protein")

    cmd.select("lig_gsh", "ref_6ssw and organic")
    cmd.show("sticks", "lig_gsh")
    cmd.show("spheres", "lig_gsh")
    cmd.set("sphere_scale", 0.35, "lig_gsh")
    cmd.color("c_coral", "lig_gsh and elem C")

    cmd.orient("dark_ulu7")
    cmd.zoom("dark_ulu7", buffer=3)
    cmd.ray(1800, 1400)
    cmd.png(str(base / "plate_6_gst_superpos.png"), dpi=300)

    # ---- 3. B2TW38 (Bifunctional ArnA, active sites 300 & 485) ----
    setup_pub_style()
    cmd.load(str(base / "AF-B2TW38.pdb"), "hero_arnA")
    cmd.hide("everything")
    cmd.show("cartoon", "hero_arnA")
    cmd.set_color("c_teal", [0.106, 0.541, 0.478])
    cmd.set_color("c_navy", [0.180, 0.357, 0.533])
    cmd.set_color("c_coral", [0.839, 0.271, 0.255])
    # Color N-terminal formyltransferase domain (1-305) navy, C-terminal dehydrogenase (306-660) teal
    cmd.color("c_navy", "hero_arnA and resi 1-305")
    cmd.color("c_teal", "hero_arnA and resi 306-660")
    cmd.select("act_arnA", "hero_arnA and resi 300+485")
    cmd.show("spheres", "act_arnA")
    cmd.color("c_coral", "act_arnA")
    cmd.orient("hero_arnA")
    cmd.zoom("hero_arnA", buffer=3)
    cmd.ray(1800, 1400)
    cmd.png(str(base / "plate_8_B2TW38_arnA.png"), dpi=300)

    # ---- 4. Q9HPK9 (pid=0% Halobacterium salinarum amidohydrolase, pLDDT=93.1) ----
    setup_pub_style()
    cmd.load(str(base / "AF-Q9HPK9.pdb"), "hero_q9")
    cmd.hide("everything")
    cmd.show("cartoon", "hero_q9")
    cmd.spectrum("b", "orange_yellow_cyan_blue", "hero_q9", minimum=50, maximum=98)
    cmd.orient("hero_q9")
    cmd.zoom("hero_q9", buffer=3)
    cmd.ray(1800, 1400)
    cmd.png(str(base / "plate_8_Q9HPK9_archaeon.png"), dpi=300)

    (base / "cealign_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("[DONE] All 4 ray-traced 3D plates saved!", flush=True)


if __name__ == "__main__":
    main()
