"""Figures for the construction: features, templates, catalog, state, actions."""

from __future__ import annotations

import math

from svgkit import (A, C, H, R_DISK, Svg, box_feature, disk_on_face, draw_box,
                    draw_disk, draw_environment, scene_at)


def features(path):
    """The finite feature inventory of every body in the scene."""
    g = Svg(940, 352, ox=0, oy=0, scale=1, cls="fig")
    g.header("Every body carries a finite, named feature inventory",
             y=34, crop=52, size=18)

    # --- box, left third
    g.ox, g.oy, g.s = 156, 180, 116
    draw_box(g, (0, 0), 0.0, mark=False)
    for name, lab, off in [("+x", "+x&#7495;", (0.78, 0.0)),
                           ("-x", "&#8722;x&#7495;", (-0.80, 0.0)),
                           ("+z", "+z&#7495;", (0.0, 0.76)),
                           ("-z", "&#8722;z&#7495;", (0.0, -0.66))]:
        q = box_feature((0, 0), 0.0, name)
        g.line(q[0], q[1], stroke=C["box"], sw=4.4)
        g.text(off, lab, size=14, fill=C["box"], weight="600", dy=5,
               family="'Latin Modern Math',Georgia,serif")
    for v in ("v++", "v+-", "v-+", "v--"):
        g.dot(box_feature((0, 0), 0.0, v), r_px=5.2, fill=C["gold"])
    g.px_text((156, 284), "box", size=14, fill=C["ink"], weight="700")
    g.px_text((156, 306), "4 edges + 4 vertices = 8", size=13.5,
              fill=C["muted"])

    # --- disk, middle
    g.ox, g.oy, g.s = 400, 180, 116
    g.circle((0, 0), R_DISK, fill=C["man_fill"], stroke=C["man"], sw=3.0)
    g.px_text((400, 284), "disk", size=14, fill=C["ink"], weight="700")
    g.px_text((400, 306), "1 boundary curve", size=13.5, fill=C["muted"])

    # --- environment, right
    scene_at(g, cx=716, floor_y=214, scale=76, x_lo=-1.7, x_hi=1.9, label=True)
    g.px_text((716, 284), "floor, riser, platform", size=14, fill=C["ink"],
              weight="700")
    g.px_text((716, 306), "1 patch each", size=13.5, fill=C["muted"])

    for x in (278, 540):
        g.add(f'<line x1="{x}" y1="66" x2="{x}" y2="312" '
              f'stroke="{C["rule"]}" stroke-width="1.4"/>')
    g.px_text((470, 338), "named once, from geometry &#8212; never re-derived "
              "at plan time", size=13, fill=C["faint"])
    return g.save(path, "Feature inventories")


# sign colours for the gap function, shared by the caption and the bracket
SIGN_POS = C["gold"]        # phi > 0, separated
SIGN_ZERO = C["primary"]    # phi = 0, touching
SIGN_NEG = C["hot"]         # phi < 0, penetrating


def template_gap(path):
    """A contact template is a named feature pair plus its signed gap."""
    g = Svg(940, 362, ox=0, oy=0, scale=1, cls="fig")
    g.header("A contact template is one named feature pair with a signed gap",
             y=34, crop=52, size=18)
    g.px_text((470, 68), "s = ( riser, F&#8347;, box, +x&#7495;, "
              "&#966;&#8347; )", size=20, fill=C["primary"], weight="600",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    g.ox, g.oy, g.s = 560, 200, 230
    AMP = 0.46      # world displacement when separated (away from the riser)
    PEN = 0.15      # world overlap at maximum penetration
    DURT = "7s"
    # Displacements are WORLD quantities: -x_W moves the box off the riser,
    # +x_W drives it in.  g.dx() carries the drawing direction, so this stays
    # correct whichever way the page is mirrored.
    sep, pen = g.dx(-AMP), g.dx(PEN)
    tvals = f"{sep:.1f},0; 0,0; {pen:.1f},0; 0,0; {sep:.1f},0"
    tkeys = "0;0.34;0.5;0.66;1"
    tspl = "0.4 0 0.4 1;0.4 0 0.4 1;0.4 0 0.4 1;0.4 0 0.4 1"

    # The two features are not the same size: +x_B spans the whole box side a,
    # while F_S only rises to the step height h = 3a/8.  Drawn to that ratio,
    # with the floor at the bottom of both.
    ZB, ZT = -0.40, 0.40                  # the box face, representing a
    ZH = ZB + (ZT - ZB) * (H / A)         # the riser face, representing h
    g.line((-0.95, ZB), (0.34, ZB), stroke=C["env"], sw=1.4)
    g.poly([(0, ZB), (0.24, ZB), (0.24, ZH), (0, ZH)],
           fill=C["env_fill"], stroke=C["env"], sw=1.4)
    g.line((0, ZB), (0, ZH), stroke=C["env"], sw=6.0)
    g.text((0.36, -0.25), "F&#8347;", size=15, fill=C["env"],
           weight="700", family="'Latin Modern Math',Georgia,serif")
    g.text((0.12, ZB), "riser, height h", size=12.5, fill=C["muted"], dy=20)

    # The gap is the distance between a MOVING face and a FIXED one, so the
    # bracket has to change length, not translate rigidly.  One end is pinned
    # to the riser; the other tracks the box face.
    ZG = -0.25
    xr, yg = g.P(0, ZG)
    x_far, x_pen = xr + sep, xr + pen
    x1v = f"{x_far:.1f};{xr:.1f};{x_pen:.1f};{xr:.1f};{x_far:.1f}"
    midv = ";".join(f"{(v + xr) / 2:.1f}"
                    for v in (x_far, xr, x_pen, xr, x_far))
    anim = (f'keyTimes="{tkeys}" dur="{DURT}" calcMode="spline" '
            f'keySplines="{tspl}" repeatCount="indefinite"')
    # The gap never stops existing -- it goes negative.  Keep the bracket
    # visible throughout and let its COLOUR carry the sign, using the same
    # three colours as the phase caption on the left.
    sign_kt = "0;0.31;0.45;0.57;0.71;1"
    sign_v = ";".join([SIGN_POS, SIGN_ZERO, SIGN_NEG,
                       SIGN_ZERO, SIGN_POS, SIGN_POS])
    sign = (f'keyTimes="{sign_kt}" values="{sign_v}" calcMode="discrete" '
            f'dur="{DURT}" repeatCount="indefinite"')
    bracket = (f'<g>'
          f'<line x1="{x_far:.1f}" y1="{yg:.1f}" x2="{xr:.1f}" y2="{yg:.1f}" '
          f'stroke="{SIGN_POS}" stroke-width="1.8" stroke-dasharray="5 4">'
          f'<animate attributeName="x1" values="{x1v}" {anim}/>'
          f'<animate attributeName="stroke" {sign}/></line>'
          f'<text x="{(x_far + xr) / 2:.1f}" y="{yg - 9:.1f}" font-size="16" '
          f'fill="{SIGN_POS}" text-anchor="middle" font-weight="700" '
          f'font-family="\'Latin Modern Math\',Georgia,serif">&#966;'
          f'<tspan baseline-shift="sub" font-size="0.68em">s</tspan>'
          f'<animate attributeName="x" values="{midv}" {anim}/>'
          f'<animate attributeName="fill" {sign}/></text></g>')

    # box face, moving
    g.add(f'<g transform="translate({sep:.1f},0)">'
          f'<animateTransform attributeName="transform" type="translate" '
          f'values="{tvals}" keyTimes="{tkeys}" dur="{DURT}" '
          f'calcMode="spline" keySplines="{tspl}" repeatCount="indefinite"/>')
    g.poly([(-0.80, ZB), (0, ZB), (0, ZT), (-0.80, ZT)],
           fill=C["box_fill"], stroke=C["box"], sw=1.8,
           extra='fill-opacity="0.62"')
    g.line((0, ZB), (0, ZT), stroke=C["box"], sw=6.0)
    g.text((-0.40, ZT), "+x&#7495; &#160; box side a", size=14, fill=C["box"],
           dy=-14, weight="600", family="'Latin Modern Math',Georgia,serif")
    g.add("</g>")

    g.add(bracket)   # annotation on top of the geometry it measures

    for t0, t1, txt, col in [
            (0.00, 0.30, "&#966;&#8347; &gt; 0 &#160; separated", SIGN_POS),
            (0.32, 0.44, "&#966;&#8347; = 0 &#160; touching", SIGN_ZERO),
            (0.46, 0.56, "&#966;&#8347; &lt; 0 &#160; penetration", SIGN_NEG),
            (0.58, 0.70, "&#966;&#8347; = 0 &#160; touching", SIGN_ZERO),
            (0.72, 1.00, "&#966;&#8347; &gt; 0 &#160; separated", SIGN_POS)]:
        op = "1" if t0 == 0 else "0"
        g.add(f'<g opacity="{op}"><animate attributeName="opacity" '
              f'values="0;0;1;1;0;0" keyTimes="0;{max(t0-0.01,0):.3f};'
              f'{t0:.3f};{t1:.3f};{min(t1+0.01,1):.3f};1" dur="{DURT}" '
              f'repeatCount="indefinite"/>')
        g.px_text((210, 200), txt, size=19, fill=col, weight="600",
                  family="'Latin Modern Math',Georgia,serif")
        g.add("</g>")

    g.px_text((470, 350), "&#966;&#8347; is a function of the configuration "
              "alone &#8212; it is defined whether or not the pair touches",
              size=13, fill=C["faint"])
    return g.save(path, "Contact template and its signed gap")


CATALOG_ROWS = [
    ("{B, M}", 8, C["man"], ["m&#8339;", "m&#7764;"], [0, 1], False),
    ("{B, L}", 8, C["env"], ["g", "g&#7525;"], [0, 1], False),
    ("{B, S}", 8, C["env"], ["s"], [0], False),
    ("{B, P}", 8, C["env"], ["t"], [0], False),
    ("{M, L}, {M, S}, {M, P}", 3, C["faint"], [], [], True),
]


def catalog(path):
    """The 35 templates of the step scene, grouped by body pair."""
    g = Svg(940, 336, ox=0, oy=0, scale=1, cls="fig")
    g.header("Every feature pair on every body pair that can move",
             y=32, crop=50, size=18)

    x0, y0, cw, ch, gx = 316, 62, 30, 26, 6
    for r, (pair, n, col, names, idxs, forb) in enumerate(CATALOG_ROWS):
        y = y0 + r * (ch + 14)
        g.px_text((x0 - 18, y + 18), pair, size=13.5, fill=C["muted"],
                  anchor="end")
        for i in range(n):
            x = x0 + i * (cw + gx)
            dash = ' stroke-dasharray="3 2"' if forb else ""
            g.add(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
                  f'fill="{"#fafbfa" if forb else "#f4f6f4"}" '
                  f'stroke="{C["rule"] if forb else C["faint"]}" '
                  f'stroke-width="1.4"{dash}/>')
            if i in idxs:
                g.px_text((x + cw / 2, y + ch / 2 + 5), names[idxs.index(i)],
                          size=14, fill=col, weight="700",
                          family="'Latin Modern Math',Georgia,serif")
        if forb:
            g.px_text((x0 + n * (cw + gx) + 12, y + 18),
                      "forbidden &#8212; manipulator against environment",
                      size=13, fill=C["faint"], anchor="start")

    g.px_text((470, 292), "|&#119966;| = 8 + 8 + 8 + 8 + 3 = 35 templates",
              size=19, fill=C["ink"], weight="600",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((470, 318), "six are named in this talk; the other twenty-nine "
              "are ordinary catalog members", size=13, fill=C["faint"])
    return g.save(path, "The contact catalog")


def state(path):
    """The central picture: discrete state is a bit vector over the catalog."""
    g = Svg(940, 406, ox=0, oy=0, scale=1, cls="fig")
    g.header("The discrete state is a bit vector over the catalog",
             y=32, crop=50, size=20)

    # --- left: the continuous configuration
    g.px_text((186, 70), "continuous pose &#967;", size=15, fill=C["muted"],
              weight="700")
    scene_at(g, cx=178, floor_y=216, scale=74, x_lo=-1.7, x_hi=1.5)
    p, th = (-A / 2, A / 2), 0.0
    draw_box(g, p, th)
    d = disk_on_face(p, th, "-x", alpha=0.6)
    draw_disk(g, d)
    g.line(*box_feature(p, th, "-z"), stroke=C["hot"], sw=3.2)
    g.line((0, 0), (0, H), stroke=C["hot"], sw=3.2)
    g.dot((d[0] + R_DISK, d[1]), r_px=4.0)
    g.px_text((186, 282), "&#967; = ( q&#7495; , p&#7742; )", size=17,
              fill=C["ink"],
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")
    g.px_text((186, 306), "&#8712; SE(2) &#215; &#8477;&#178;", size=15,
              fill=C["muted"],
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    g.add(f'<line x1="366" y1="60" x2="366" y2="326" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')

    # --- right: the bit vector
    x0, y0, cw, ch, gx = 470, 92, 30, 26, 6
    # The bit vector must correspond exactly to the pose beside it: held are
    # m_x, g and s.  Nothing flips here -- the pose is static, and flipping a
    # bit is the next slide's subject.
    rows = [("{B, M}", 8, ["m&#8339;", "m&#7764;"], [0, 1], {0}, None),
            ("{B, L}", 8, ["g", "g&#7525;"], [0, 1], {0}, None),
            ("{B, S}", 8, ["s"], [0], {0}, None),
            ("{B, P}", 8, ["t"], [0], set(), None)]
    g.px_text((680, 70), "asserted contacts &#963;", size=15, fill=C["muted"],
              weight="700",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")
    for r, (pair, n, names, idxs, on, flip) in enumerate(rows):
        y = y0 + r * (ch + 12)
        g.px_text((x0 - 12, y + 18), pair, size=13, fill=C["muted"],
                  anchor="end")
        for i in range(n):
            x = x0 + i * (cw + gx)
            lit = i in on
            anim = ""
            if flip is not None and i == flip:
                anim = ('<animate attributeName="fill" values="#f4f6f4;'
                        '#f4f6f4;#175b48;#175b48;#f4f6f4" '
                        'keyTimes="0;0.45;0.56;0.86;1" dur="6s" '
                        'repeatCount="indefinite"/>')
            g.add(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
                  f'fill="{C["primary"] if lit else "#f4f6f4"}" '
                  f'stroke="{C["primary"] if lit else C["faint"]}" '
                  f'stroke-width="1.4">{anim}</rect>')
            if i in idxs:
                tanim = ""
                if flip is not None and i == flip:
                    tanim = ('<animate attributeName="fill" values="#6b7a74;'
                             '#6b7a74;#ffffff;#ffffff;#6b7a74" '
                             'keyTimes="0;0.45;0.56;0.86;1" dur="6s" '
                             'repeatCount="indefinite"/>')
                g.add(f'<text x="{x+cw/2:.1f}" y="{y+ch/2+5:.1f}" '
                      f'font-size="14" text-anchor="middle" font-weight="700" '
                      f'fill="{"#ffffff" if lit else C["muted"]}" '
                      f'font-family="\'Latin Modern Math\',Georgia,serif">'
                      f'{names[idxs.index(i)]}{tanim}</text>')

    g.px_text((680, 282), "&#963; = { m&#8339; , g , s }", size=20,
              fill=C["primary"], weight="700",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")
    g.px_text((680, 306), "what the controller is committed to hold",
              size=13.5, fill=C["muted"])

    g.px_text((470, 352), "search state &#160; n = ( &#967; , &#963; )",
              size=21, fill=C["ink"], weight="700",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")
    g.px_text((470, 380), "&#963; is a <tspan font-style='italic'>commitment"
              "</tspan>, not a measurement: it says what must stay in contact, "
              "not what happens to be touching", size=13, fill=C["faint"])
    return g.save(path, "State: pose plus bit vector")


def actions(path):
    """make / break / transport."""
    g = Svg(940, 338, ox=0, oy=0, scale=1, cls="fig")
    g.header("Three actions, each flipping at most one bit", y=32, crop=50,
             size=20)

    for i, (name, op, sub, col) in enumerate([
            ("make(c)", "&#963; &#8746; { c }",
             "start holding a contact not yet held", C["primary"]),
            ("break(c)", "&#963; &#8726; { c }",
             "stop holding a contact that is held", C["hot"]),
            ("transport", "&#963;",
             "keep &#963; and drive toward the goal", C["gold"])]):
        x = 56 + i * 285
        g.add(f'<rect x="{x}" y="60" width="262" height="104" rx="8" '
              f'fill="#ffffff" stroke="{col}" stroke-width="1.8"/>')
        g.px_text((x + 131, 90), name, size=18, fill=col, weight="700",
                  family="'Latin Modern Math',Georgia,serif")
        g.px_text((x + 131, 120), op, size=17, fill=C["ink"],
                  family="'Latin Modern Math','STIX Two Math',Georgia,serif")
        g.px_text((x + 131, 146), sub, size=13, fill=C["muted"])

    g.px_text((470, 224), "A plan is a sequence of these.", size=17,
              fill=C["ink"], weight="700")
    g.px_text((470, 252), "At most |&#119966;| + 1 at any state: one flip per "
              "catalog bit, plus transport.", size=14, fill=C["muted"],
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((470, 276), "that is 36 here &#8212; the two feasibility tests "
              "cut it to a handful", size=13, fill=C["faint"])
    g.px_text((470, 312), "all the difficulty is in <tspan font-style='italic'>"
              "which flips are realizable from this pose</tspan>", size=16,
              fill=C["primary"], weight="600")
    return g.save(path, "The three actions")


# ------------------------------------------------------- the plan as a walk

CHIP = ["m&#8339;", "m&#7764;", "g", "g&#7525;", "s", "t"]


def bit_chip(g, cx, cy, bits, cw=19, ch=22, gap=3, hot=None, labels=None):
    """One marking drawn as a row of cells; `hot` outlines the flipping bit."""
    for i, b in enumerate(bits):
        x = cx + i * (cw + gap)
        on = hot == i
        g.add(f'<rect x="{x}" y="{cy}" width="{cw}" height="{ch}" rx="3" '
              f'fill="{C["primary"] if b else "#f4f6f4"}" '
              f'stroke="{C["gold"] if on else (C["primary"] if b else C["faint"])}" '
              f'stroke-width="{2.2 if on else 1.3}"/>')
        if labels:
            g.px_text((x + cw / 2, cy + ch + 13), labels[i], size=10,
                      fill=C["muted"], weight="700",
                      family="'Latin Modern Math',Georgia,serif")
    return cx + len(bits) * (cw + gap) - gap
PLAN = [
    ("&#963;&#8320;", [1, 0, 1, 0, 0, 0], "M0"),
    ("&#963;&#8321;", [1, 0, 1, 0, 1, 0], "M1"),
    ("&#963;&#8322;", [1, 0, 1, 1, 1, 0], ""),
    ("&#963;&#8323;", [1, 0, 0, 1, 1, 0], "M2"),
    ("&#963;&#8324;", [1, 0, 0, 0, 1, 0], ""),
    ("&#963;&#8325;", [1, 0, 0, 0, 1, 1], "M3"),
    ("&#963;&#8326;", [0, 0, 0, 0, 1, 1], ""),
    ("&#963;&#8327;", [0, 1, 0, 0, 1, 1], ""),
    ("&#963;&#8328;", [0, 1, 0, 0, 0, 1], "M4"),
]
OPS = ["make s", "make g&#7525;", "break g", "break g&#7525;", "make t",
       "break m&#8339;", "make m&#7764;", "break s"]


def walk(path):
    """The running example's plan, drawn as a walk on the marking graph."""
    g = Svg(940, 392, ox=0, oy=0, scale=1, cls="fig")
    g.header("The plan is a walk on the graph of markings", y=30, crop=48)

    cw, ch, gap = 19, 22, 3
    chip_w = 6 * cw + 5 * gap
    per_row = 5
    x0, y0, dy = 44, 92, 158
    step = (940 - x0 - 30 - chip_w) / (per_row - 1)

    def chip(cx, cy, bits, name, panel, changed=None):
        bit_chip(g, cx, cy, bits, cw, ch, gap, changed)
        g.px_text((cx + chip_w / 2, cy - 9), name, size=15, weight="700",
                  fill=C["ink"],
                  family="'Latin Modern Math','STIX Two Math',Georgia,serif")
        if panel:
            g.px_text((cx + chip_w / 2, cy + ch + 16), panel, size=12,
                      weight="700", fill=C["muted"])

    for k, (name, bits, panel) in enumerate(PLAN):
        r, c = divmod(k, per_row)
        cx, cy = x0 + c * step, y0 + r * dy
        prev = PLAN[k - 1][1] if k else None
        changed = next((i for i in range(6) if prev and prev[i] != bits[i]),
                       None)
        chip(cx, cy, bits, name, panel, changed)
        if k < len(PLAN) - 1 and c < per_row - 1:
            ax, ay = cx + chip_w + 6, cy + ch / 2
            g.add(f'<line x1="{ax}" y1="{ay}" x2="{ax+step-chip_w-22}" '
                  f'y2="{ay}" stroke="{C["faint"]}" stroke-width="1.6"/>'
                  f'<polygon points="{ax+step-chip_w-12},{ay} '
                  f'{ax+step-chip_w-22},{ay-5} {ax+step-chip_w-22},{ay+5}" '
                  f'fill="{C["faint"]}"/>')
            g.px_text((ax + (step - chip_w - 16) / 2, ay - 10), OPS[k],
                      size=11.5, fill=C["gold"], weight="700",
                      family="'Latin Modern Math',Georgia,serif")

    # wrap from the end of row one down to the start of row two, routed
    # below row one and back along the left margin so it crosses nothing
    c4x = x0 + (per_row - 1) * step + chip_w / 2
    midy = y0 + dy - 52
    lx0 = x0 - 18
    ry = y0 + dy + ch / 2
    g.add(f'<path d="M {c4x} {y0+ch+10} L {c4x} {midy} L {lx0} {midy} '
          f'L {lx0} {ry} L {x0-12} {ry}" fill="none" stroke="{C["faint"]}" '
          f'stroke-width="1.6" stroke-linejoin="round"/>'
          f'<polygon points="{x0-2},{ry} {x0-12},{ry-5} {x0-12},{ry+5}" '
          f'fill="{C["faint"]}"/>')
    g.px_text(((c4x + lx0) / 2, midy - 8), OPS[4], size=11.5, fill=C["gold"],
              weight="700", family="'Latin Modern Math',Georgia,serif")

    # the final transport into the goal set
    lx = x0 + 3 * step + chip_w
    ly = y0 + dy + ch / 2
    g.add(f'<line x1="{lx+6}" y1="{ly}" x2="{lx+56}" y2="{ly}" '
          f'stroke="{C["primary"]}" stroke-width="1.8"/>'
          f'<polygon points="{lx+66},{ly} {lx+56},{ly-5} {lx+56},{ly+5}" '
          f'fill="{C["primary"]}"/>')
    g.px_text((lx + 36, ly - 10), "transport", size=11.5, fill=C["primary"],
              weight="700")
    g.add(f'<rect x="{lx+74}" y="{ly-17}" width="86" height="34" rx="6" '
          f'fill="{C["primary_soft"]}" stroke="{C["primary"]}" '
          f'stroke-width="1.6"/>')
    g.px_text((lx + 117, ly + 5), "&#967; &#8712; &#119970;(q)", size=14,
              fill=C["primary"], weight="700",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    # legend
    ly2 = 338
    for i, lab in enumerate(CHIP):
        x = x0 + i * (cw + gap)
        g.add(f'<rect x="{x}" y="{ly2}" width="{cw}" height="{ch}" rx="3" '
              f'fill="#ffffff" stroke="{C["faint"]}" stroke-width="1.3"/>')
        g.px_text((x + cw / 2, ly2 + 15), lab, size=11, fill=C["muted"],
                  weight="700",
                  family="'Latin Modern Math',Georgia,serif")
    g.px_text((x0 + chip_w + 16, ly2 + 15), "one cell per candidate contact "
              "&#8212; filled = held, gold outline = the bit this action flips",
              size=12.5, fill=C["muted"], anchor="start")
    g.px_text((470, 380), "nine actions, nine calls to the local solver; "
              "consecutive markings differ by exactly one bit", size=13,
              fill=C["faint"])
    return g.save(path, "The plan as a walk on the marking graph")


# -------------------------------------------------- what the marking graph is

def markgraph(path):
    """The real object: valid subsets, layered by |sigma|, one bit per edge."""
    from itertools import combinations

    g = Svg(940, 470, ox=0, oy=0, scale=1, cls="fig")
    g.header("The marking graph is the cube minus the excluded markings",
             y=30, crop=48)

    EXCL = frozenset({2, 5})                     # g and t cannot both be held
    allsub = [frozenset(c) for k in range(7) for c in combinations(range(6), k)]
    valid = [s for s in allsub if not EXCL <= s]
    plan = [frozenset(b) for b in
            [{0, 2}, {0, 2, 4}, {0, 2, 3, 4}, {0, 3, 4}, {0, 4},
             {0, 4, 5}, {4, 5}, {1, 4, 5}, {1, 5}]]
    planset = set(plan)

    # Walk vertices are placed by their POSITION IN THE WALK, so the path
    # marches steadily down while zigzagging between columns and can be
    # followed by eye.  The rest fill their column, nudged clear.
    idx = {t: i for i, t in enumerate(plan)}
    x0, dx, ytop, ybot = 108, 122, 84, 326
    pos = {t: (x0 + dx * len(t), ytop + (ybot - ytop) * idx[t] / (len(plan) - 1))
           for t in plan}
    cols = {k: sorted([t for t in valid if len(t) == k], key=sorted)
            for k in range(7)}
    for k in range(7):
        rest = [t for t in cols[k] if t not in planset]
        taken = [pos[t][1] for t in plan if len(t) == k]
        n = len(rest)
        for i, t in enumerate(rest):
            y = (ytop + ybot) / 2 if n == 1 else ytop + i * (ybot - ytop) / (n - 1)
            for u in taken:
                if abs(y - u) < 15:
                    y = u + 15 if y >= u else u - 15
            pos[t] = (x0 + dx * k, y)

    walkedge = {(plan[i], plan[i + 1]) for i in range(len(plan) - 1)}
    for a in valid:                       # background lattice, undirected
        for b in valid:
            if (len(b) == len(a) + 1 and a < b
                    and (a, b) not in walkedge and (b, a) not in walkedge):
                (x1, y1), (x2, y2) = pos[a], pos[b]
                g.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" '
                      f'y2="{y2:.1f}" stroke="{C["rule"]}" stroke-width="0.9" '
                      f'opacity="0.85"/>')
    for i in range(len(plan) - 1):        # the walk, directed, in order
        (x1, y1), (x2, y2) = pos[plan[i]], pos[plan[i + 1]]
        a = math.atan2(y2 - y1, x2 - x1)
        sx, sy = x1 + 7 * math.cos(a), y1 + 7 * math.sin(a)
        ex, ey = x2 - 9 * math.cos(a), y2 - 9 * math.sin(a)
        g.add(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
              f'stroke="{C["primary"]}" stroke-width="2.6"/>')
        w = 4.2
        g.add(f'<polygon points="{x2-4*math.cos(a):.1f},{y2-4*math.sin(a):.1f} '
              f'{ex-w*math.sin(a):.1f},{ey+w*math.cos(a):.1f} '
              f'{ex+w*math.sin(a):.1f},{ey-w*math.cos(a):.1f}" '
              f'fill="{C["primary"]}"/>')
    for t in valid:
        x, y = pos[t]
        on = t in planset
        g.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{5.4 if on else 3.4}" '
              f'fill="{C["primary"] if on else "#ffffff"}" '
              f'stroke="{C["primary"] if on else C["faint"]}" '
              f'stroke-width="{1.6 if on else 1.2}"/>')
    for i, t in enumerate(plan):
        x, y = pos[t]
        # put the label on the side the walk is NOT heading, so it never
        # sits under the outgoing hop
        nb = plan[i + 1] if i < len(plan) - 1 else plan[i - 1]
        right = pos[nb][0] < x
        g.px_text((x + (15 if right else -15), y + 5),
                  f"&#963;<tspan baseline-shift='sub' font-size='0.68em'>{i}"
                  f"</tspan>", size=13, weight="700", fill=C["primary"],
                  anchor="start" if right else "end",
                  family="'Latin Modern Math','STIX Two Math',Georgia,serif",
                  extra='stroke="#ffffff" stroke-width="3.2" '
                        'paint-order="stroke" stroke-linejoin="round"')

    for k in range(7):
        n = len(cols[k])
        g.px_text((x0 + k * dx, 370), f"|&#963;| = {k}", size=12.5,
                  fill=C["muted"], weight="700",
                  family="'Latin Modern Math',Georgia,serif")
        g.px_text((x0 + k * dx, 390), f"{n}" if n else "0 &#8212; excluded",
                  size=12.5, fill=C["hot"] if not n else C["faint"],
                  weight="700" if not n else "400")

    g.add(f'<path d="M {x0+150} 66 L {x0+230} 66" stroke="{C["faint"]}" '
          f'stroke-width="1.4"/>'
          f'<polygon points="{x0+240},66 {x0+230},61 {x0+230},71" '
          f'fill="{C["faint"]}"/>')
    g.px_text((x0 + 195, 58), "make", size=12, fill=C["muted"], weight="700")
    g.add(f'<path d="M {x0+470} 66 L {x0+390} 66" stroke="{C["faint"]}" '
          f'stroke-width="1.4"/>'
          f'<polygon points="{x0+380},66 {x0+390},61 {x0+390},71" '
          f'fill="{C["faint"]}"/>')
    g.px_text((x0 + 430, 58), "break", size=12, fill=C["muted"], weight="700")

    g.px_text((470, 424), "48 of the 64 subsets of the six named contacts; "
              "the 16 holding both g and t are gone", size=14, fill=C["ink"],
              weight="600")
    g.px_text((470, 448), "on the full catalog this is the 35-cube with those "
              "markings deleted &#8212; 2&#179;&#8309; vertices, never built",
              size=13, fill=C["faint"])
    return g.save(path, "The marking graph")


# ----------------------------------------------------- the method at a glance

def overview(path):
    """The whole method as one picture: setup once, then a loop."""
    g = Svg(940, 476, ox=0, oy=0, scale=1, cls="fig")
    g.header("The method at a glance", y=30, crop=46)

    # ---------- setup: the scene fixes a finite list of candidate contacts
    scene_at(g, cx=118, floor_y=116, scale=40, x_lo=-1.9, x_hi=1.9)
    p0 = (-A / 2, A / 2)
    draw_box(g, p0, 0.0, mark=False)
    draw_disk(g, disk_on_face(p0, 0.0, "-x", 0.55))
    g.line((0, 0), (0, H), stroke=C["violet"], sw=2.4, dash="3 3")
    g.line((0, H), (2 * A, H), stroke=C["violet"], sw=2.4, dash="3 3")

    g.add(f'<line x1="196" y1="96" x2="232" y2="96" stroke="{C["faint"]}" '
          f'stroke-width="1.6"/><polygon points="242,96 232,91 232,101" '
          f'fill="{C["faint"]}"/>')
    cw, ch, gp = 20, 24, 4
    for i in range(11):
        x = 256 + i * (cw + gp)
        g.add(f'<rect x="{x}" y="84" width="{cw}" height="{ch}" rx="3" '
              f'fill="#f4f6f4" stroke="{C["faint"]}" stroke-width="1.2"/>')
    for i, lab in ((0, "m&#8339;"), (2, "g"), (5, "s"), (7, "t")):
        g.px_text((256 + i * (cw + gp) + cw / 2, 101), lab, size=11.5,
                  fill=C["muted"], weight="700",
                  family="'Latin Modern Math',Georgia,serif")
    g.px_text((520, 101), "&#8943;", size=16, fill=C["faint"])
    g.px_text((556, 101), "|&#119966;| = 35 candidate contacts", size=14,
              anchor="start", fill=C["ink"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((556, 122), "every named feature pair &#8212; touching or not",
              size=12.5, anchor="start", fill=C["faint"])
    g.px_text((118, 140), "the scene", size=12.5, fill=C["faint"])

    g.add(f'<line x1="30" y1="164" x2="910" y2="164" stroke="{C["rule"]}" '
          f'stroke-width="1.2"/>')

    # ---------- the loop
    box_y, box_h = 194, 168
    stages = [(30, 216), (274, 196), (500, 174), (704, 206)]

    def frame(x, w, title):
        g.add(f'<rect x="{x}" y="{box_y}" width="{w}" height="{box_h}" '
              f'rx="8" fill="#ffffff" stroke="{C["rule"]}" '
              f'stroke-width="1.6"/>')
        g.px_text((x + w / 2, box_y + 24), title, size=14, weight="700",
                  fill=C["primary"])

    # 1 state
    x, w = stages[0]
    frame(x, w, "state")
    scene_at(g, cx=x + 66, floor_y=296, scale=30, x_lo=-1.8, x_hi=1.8)
    draw_box(g, p0, 0.0, mark=False)
    draw_disk(g, disk_on_face(p0, 0.0, "-x", 0.55))
    bit_chip(g, x + 118, 274, [1, 0, 1, 0, 0, 0], 13, 17, 2)
    g.px_text((x + w / 2, 334), "pose &#967; + bit vector &#963;", size=12.5,
              fill=C["ink"],
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    # 2 one action
    x, w = stages[1]
    frame(x, w, "one action")
    bit_chip(g, x + 34, 252, [1, 0, 1, 0, 0, 0], 17, 21, 3)
    g.add(f'<rect x="{x+34+4*20}" y="252" width="17" height="21" rx="3" '
          f'fill="#f4f6f4" stroke="{C["gold"]}" stroke-width="2.2">'
          f'<animate attributeName="fill" '
          f'values="#f4f6f4;#f4f6f4;#175b48;#175b48;#f4f6f4" '
          f'keyTimes="0;0.35;0.5;0.85;1" dur="4s" '
          f'repeatCount="indefinite"/></rect>')
    g.px_text((x + w / 2, 300), "flip exactly one bit", size=13,
              fill=C["ink"], weight="700")
    for i, (lab, col) in enumerate([("make", C["primary"]),
                                    ("break", C["hot"]),
                                    ("transport", C["gold"])]):
        g.px_text((x + 34 + i * 54, 326), lab, size=11.5, fill=col,
                  weight="700", anchor="start")
    g.px_text((x + w / 2, 348), "&#8804; |&#119966;| + 1 at any state",
              size=12, fill=C["faint"],
              family="'Latin Modern Math',Georgia,serif")

    # 3 two tests
    x, w = stages[2]
    frame(x, w, "two cheap tests")
    for i, (q, col) in enumerate([("near enough?", C["gold"]),
                                  ("still supported?", C["gold"])]):
        g.add(f'<rect x="{x+16}" y="{250+i*46}" width="{w-32}" height="32" '
              f'rx="16" fill="#fbf6e8" stroke="{col}" stroke-width="1.4"/>')
        g.px_text((x + w / 2, 271 + i * 46), q, size=12.5, fill=C["ink"],
                  weight="700")
    g.px_text((x + w / 2, 340), "refuse here, and &#923;", size=12,
              fill=C["faint"])
    g.px_text((x + w / 2, 356), "is never called", size=12, fill=C["faint"])

    # 4 the solver
    x, w = stages[3]
    frame(x, w, "&#923; &#160; short-horizon optimizer")
    for k in range(7):
        sp = (k - 3) * 7
        g.add(f'<path d="M {x+28} 300 Q {x+80} {280+sp} {x+132} {272+sp*1.7}" '
              f'fill="none" stroke="{C["faint"]}" stroke-width="1.2" '
              f'opacity="0.5"><animate attributeName="stroke-dasharray" '
              f'values="0 240;240 0" dur="2.6s" begin="{k*0.07}s" '
              f'repeatCount="indefinite"/></path>')
    g.add(f'<path d="M {x+28} 300 Q {x+82} 276 {x+136} 260" fill="none" '
          f'stroke="{C["primary"]}" stroke-width="2.6">'
          f'<animate attributeName="stroke-dasharray" values="0 240;240 0" '
          f'dur="2.6s" begin="0.5s" repeatCount="indefinite"/></path>')
    g.add(f'<circle cx="{x+28}" cy="300" r="4" fill="{C["ink"]}"/>')
    g.add(f'<rect x="{x+22}" y="316" width="78" height="26" rx="5" '
          f'fill="#f2f8f4" stroke="{C["primary"]}" stroke-width="1.4"/>')
    g.px_text((x + 61, 334), "accept", size=12.5, fill=C["primary"],
              weight="700")
    g.add(f'<rect x="{x+110}" y="316" width="74" height="26" rx="5" '
          f'fill="{C["hot_soft"]}" stroke="{C["hot"]}" stroke-width="1.4"/>')
    g.px_text((x + 147, 334), "reject", size=12.5, fill=C["hot"],
              weight="700")

    for a, b in [(246, 274), (470, 500), (674, 704)]:
        g.add(f'<line x1="{a}" y1="278" x2="{b-10}" y2="278" '
              f'stroke="{C["faint"]}" stroke-width="1.8"/>'
              f'<polygon points="{b},278 {b-10},273 {b-10},283" '
              f'fill="{C["faint"]}"/>')

    # accept loops back to the state; reject returns to the action list
    # both feedback branches travel upward, so the heads must point up
    bb = box_y + box_h
    g.add(f'<path d="M {stages[3][0]+61} 344 L {stages[3][0]+61} 392 '
          f'L 138 392 L 138 {bb+12}" fill="none" '
          f'stroke="{C["primary"]}" stroke-width="1.8"/>'
          f'<polygon points="138,{bb} 133,{bb+12} 143,{bb+12}" '
          f'fill="{C["primary"]}"/>')
    g.px_text((470, 386), "accept &#8594; new state, and one more "
              "&#923; call spent", size=13, fill=C["primary"], weight="700")
    rx = stages[1][0] + 98
    g.add(f'<path d="M {stages[3][0]+147} 344 L {stages[3][0]+147} 420 '
          f'L {rx} 420 L {rx} {bb+12}" '
          f'fill="none" stroke="{C["hot"]}" stroke-width="1.6" '
          f'stroke-dasharray="5 4"/>'
          f'<polygon points="{rx},{bb} {rx-5},{bb+12} {rx+5},{bb+12}" '
          f'fill="{C["hot"]}"/>')
    g.px_text((470, 438), "reject &#8594; try the next action, same state",
              size=13, fill=C["hot"], weight="700")
    g.px_text((470, 464), "the plan is the accepted sequence; its cost is the "
              "number of &#923; calls", size=13, fill=C["faint"])
    return g.save(path, "The method at a glance")
