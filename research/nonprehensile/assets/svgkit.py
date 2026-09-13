"""Minimal dependency-free SVG builder for the nonprehensile-manipulation deck.

World convention, matching the report exactly:

    +x_W to the LEFT on the page, +z_W up, +y_W out of the page, so
    z_W x x_W = y_W.  theta is the rotation about +y_W; positive theta is
    the counterclockwise topple, carrying +x_B from "left" toward "down".

    R(theta) (xi_x, xi_z) = (xi_x cos t + xi_z sin t, -xi_x sin t + xi_z cos t)

At theta = pi/2 this is (xi_x, xi_z) -> (xi_z, -xi_x), the report's Lemma
"Box faces at M3".  The box therefore travels right-to-left across the page,
as it does in the report's figures.

Drawing is controlled by X_RIGHT below.  All world-space geometry is written
once and is independent of it; only the world->pixel map, SMIL translation
deltas (Svg.dx) and SMIL rotation angles (Svg.rot_deg) carry the sign.

Scene geometry of S_step, world origin at the floor-riser corner:

    lower floor   L : x <= 0, z <= 0
    riser         S : x >= 0, 0 <= z <= h
    upper platform P : 0 <= x <= 2a, z = h
    box           B : square of side a,  h = 3a/8
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------- palette

C = {
    "ink": "#1c2b26",
    "muted": "#6b7a74",
    "faint": "#aab5b0",
    "rule": "#dce3dd",
    "primary": "#175b48",
    "primary_soft": "#e4efe9",
    "box": "#2f6690",
    "box_fill": "#dae8f2",
    "env": "#7b8a85",
    "env_fill": "#e7ebe8",
    "man": "#c4702f",
    "man_fill": "#f7e3cf",
    "hot": "#c0392b",
    "hot_soft": "#f6dedb",
    "gold": "#b8860b",
    "violet": "#6b4c9a",
    "paper": "#ffffff",
}

# The report draws +x_W to the left, so the manipulation reads right-to-left.
# Set True to mirror the page (the geometry and every proof are unchanged).
X_RIGHT = False
XS = 1.0 if X_RIGHT else -1.0

A = 1.0           # box side
H = 3 * A / 8     # step height
R_DISK = A / 8     # manipulator radius
PIVOT = (0.0, H)  # riser top corner



# ---------------------------------------------------------------- entities

def _sub(t):
    return f'<tspan baseline-shift="sub" font-size="0.68em">{t}</tspan>'


# SVG is XML: only the five predefined entities plus numeric references are
# legal, and several "subscript letter" code points (notably subscript z)
# simply do not exist in Unicode.  Everything typed as a convenient shorthand
# in the figure sources is rewritten here, once, at render time.
ENTITIES = [
    ("&nbsp;", "&#160;"),
    ("m&#8339;", "m" + _sub("x")),
    ("m&#7764;", "m" + _sub("z")),
    ("g&#7525;", "g" + _sub("v")),
    ("&#966;&#8347;", "&#966;" + _sub("s")),
    ("&#966;&#8348;", "&#966;" + _sub("t")),
    ("F&#8347;", "F" + _sub("S")),
    ("F&#8348;", "F" + _sub("P")),
    ("F&#8556;", "F" + _sub("L")),
    ("x&#7495;", "x" + _sub("B")),
    ("z&#7495;", "z" + _sub("B")),
    ("q&#7495;", "q" + _sub("B")),
    ("p&#7742;", "p" + _sub("M")),
    ("&#119970;", "<tspan font-style=\"italic\">G</tspan>"),
    ("&#119966;", "<tspan font-style=\"italic\">C</tspan>"),
    ("&#119861;", "<tspan font-style=\"italic\">B</tspan>"),
    ("&#119872;", "<tspan font-style=\"italic\">M</tspan>"),
]
for _d in range(10):
    ENTITIES.append((f"&#963;&#832{_d};", "&#963;" + _sub(str(_d))))


def fix_entities(markup):
    for a, b in ENTITIES:
        markup = markup.replace(a, b)
    return markup


# Figure headers duplicate the slide headings when these figures are used in
# the deck, so they are off by default.  Turn them on to reuse a figure
# standalone (in the report, or as a shared image).
SHOW_TITLES = False


# ---------------------------------------------------------------- canvas


class Svg:
    """Accumulates SVG markup in world coordinates via an affine map."""

    def __init__(self, w, h, ox, oy, scale, bg=None, cls="fig"):
        self.w, self.h = w, h
        self.ox, self.oy, self.s = ox, oy, scale
        self.parts = []
        self.defs = []
        self.cls = cls
        self.crop = 0
        if bg:
            self.parts.append(
                f'<rect x="0" y="0" width="{w}" height="{h}" fill="{bg}"/>'
            )

    # world -> svg
    def P(self, x, z):
        return (self.ox + self.s * XS * x, self.oy - self.s * z)

    def L(self, v):
        return self.s * v

    def dx(self, world_dx):
        """Horizontal pixel delta for a world-space displacement."""
        return self.s * XS * world_dx

    @staticmethod
    def rot_deg(world_deg):
        """SVG rotation angle for a rotation of `world_deg` about +y_W."""
        return XS * world_deg

    def header(self, title, sub=None, y=32, crop=None, size=19,
               subsize=14):
        """Figure title and subtitle, drawn only when SHOW_TITLES is set.

        When they are suppressed the empty band they occupied is cropped out
        of the viewBox, so the figure keeps its composition either way.
        """
        if SHOW_TITLES:
            self.px_text((self.w / 2, y), title, size=size, weight="700",
                         fill=C["ink"])
            if sub:
                self.px_text((self.w / 2, y + 24), sub, size=subsize,
                             fill=C["muted"])
        else:
            self.crop = crop if crop is not None else y + (28 if sub else 16)
        return self

    def add(self, markup):
        self.parts.append(markup)
        return self

    def define(self, markup):
        self.defs.append(markup)
        return self

    # ------------------------------------------------------------ shapes

    def poly(self, pts_world, fill="none", stroke="none", sw=1.4,
             opacity=None, extra="", close=True):
        pts = " ".join(f"{a:.3f},{b:.3f}" for a, b in
                       (self.P(x, z) for x, z in pts_world))
        tag = "polygon" if close else "polyline"
        op = f' opacity="{opacity}"' if opacity is not None else ""
        return self.add(
            f'<{tag} points="{pts}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linejoin="round"{op} {extra}/>'
        )

    def line(self, p0, p1, stroke=C["ink"], sw=1.4, dash=None, extra="",
             cap="round", opacity=None):
        x0, y0 = self.P(*p0)
        x1, y1 = self.P(*p1)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        return self.add(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{x1:.3f}" y2="{y1:.3f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"'
            f'{d}{op} {extra}/>'
        )

    def circle(self, c, r_world, fill="none", stroke=C["ink"], sw=1.4,
               extra="", opacity=None):
        cx, cy = self.P(*c)
        op = f' opacity="{opacity}"' if opacity is not None else ""
        return self.add(
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{self.L(r_world):.3f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{op} {extra}/>'
        )

    def dot(self, c, r_px=4.2, fill=C["hot"], stroke="#ffffff", sw=1.2,
            extra=""):
        cx, cy = self.P(*c)
        return self.add(
            f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{r_px}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw}" {extra}/>'
        )

    def text(self, p, s, size=15, fill=C["ink"], anchor="middle",
             weight="400", dx=0, dy=0, style="", extra="", family=None):
        x, y = self.P(*p)
        fam = family or "'Source Sans Pro','Helvetica Neue',Arial,sans-serif"
        return self.add(
            f'<text x="{x + dx:.2f}" y="{y + dy:.2f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{fam}" style="{style}" {extra}>{s}</text>'
        )

    def mathtext(self, p, s, size=15, **kw):
        kw.setdefault("family", "'Latin Modern Math','STIX Two Math',Georgia,serif")
        kw.setdefault("style", "font-style:italic")
        return self.text(p, s, size=size, **kw)

    def px_text(self, xy, s, size=14, fill=C["ink"], anchor="middle",
                weight="400", style="", extra="", family=None):
        """Text placed directly in pixel coordinates."""
        fam = family or "'Source Sans Pro','Helvetica Neue',Arial,sans-serif"
        return self.add(
            f'<text x="{xy[0]:.2f}" y="{xy[1]:.2f}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}" '
            f'font-family="{fam}" style="{style}" {extra}>{s}</text>'
        )

    def arrow(self, p0, p1, stroke=C["ink"], sw=1.6, head=7.0, extra="",
              dash=None, opacity=None):
        x0, y0 = self.P(*p0)
        x1, y1 = self.P(*p1)
        ang = math.atan2(y1 - y0, x1 - x0)
        bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity}"' if opacity is not None else ""
        wing = head * 0.46
        p_a = (bx - wing * math.sin(ang), by + wing * math.cos(ang))
        p_b = (bx + wing * math.sin(ang), by - wing * math.cos(ang))
        self.add(
            f'<line x1="{x0:.3f}" y1="{y0:.3f}" x2="{bx:.3f}" y2="{by:.3f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"'
            f'{d}{op} {extra}/>'
        )
        return self.add(
            f'<polygon points="{x1:.2f},{y1:.2f} {p_a[0]:.2f},{p_a[1]:.2f} '
            f'{p_b[0]:.2f},{p_b[1]:.2f}" fill="{stroke}"{op} {extra}/>'
        )

    def arc_arrow(self, c, r_world, a0_deg, a1_deg, stroke=C["man"], sw=1.8,
                  head=7.0):
        """Circular arrow in world coords; angles measured CCW from +x_W."""
        cx, cy = self.P(*c)
        r = self.L(r_world)
        a0, a1 = math.radians(a0_deg), math.radians(a1_deg)
        # svg y is down, so a world CCW angle maps to a negative svg angle
        x0, y0 = cx + XS * r * math.cos(a0), cy - r * math.sin(a0)
        x1, y1 = cx + XS * r * math.cos(a1), cy - r * math.sin(a1)
        large = 1 if abs(a1_deg - a0_deg) > 180 else 0
        sweep = (0 if a1_deg > a0_deg else 1) if XS > 0 else \
                (1 if a1_deg > a0_deg else 0)
        self.add(
            f'<path d="M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {large} '
            f'{sweep} {x1:.2f} {y1:.2f}" fill="none" stroke="{stroke}" '
            f'stroke-width="{sw}" stroke-linecap="round"/>'
        )
        tang = a1 + (math.pi / 2 if a1_deg > a0_deg else -math.pi / 2)
        ang = math.atan2(-math.sin(tang), XS * math.cos(tang))
        wing = head * 0.5
        p_a = (x1 - head * math.cos(ang) - wing * math.sin(ang),
               y1 - head * math.sin(ang) + wing * math.cos(ang))
        p_b = (x1 - head * math.cos(ang) + wing * math.sin(ang),
               y1 - head * math.sin(ang) - wing * math.cos(ang))
        return self.add(
            f'<polygon points="{x1:.2f},{y1:.2f} {p_a[0]:.2f},{p_a[1]:.2f} '
            f'{p_b[0]:.2f},{p_b[1]:.2f}" fill="{stroke}"/>'
        )

    # ------------------------------------------------------------ output

    def render(self, title=""):
        defs = f"<defs>{''.join(self.defs)}</defs>" if self.defs else ""
        t = f"<title>{title}</title>" if title else ""
        body = fix_entities("".join(self.parts))
        top = self.crop
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 {top} '
            f'{self.w} {self.h - top}" width="100%" class="{self.cls}" '
            f'role="img" aria-label="{title}">{t}{defs}'
            + body
            + "</svg>\n"
        )

    def save(self, path, title=""):
        with open(path, "w") as fh:
            fh.write(self.render(title))
        return path


# ---------------------------------------------------------------- geometry


def rot(theta, xi):
    """R(theta) acting on a box-frame vector; positive theta topples the box."""
    c, s = math.cos(theta), math.sin(theta)
    return (xi[0] * c + xi[1] * s, -xi[0] * s + xi[1] * c)


def box_corners(p, theta, a=A):
    """World polygon of the box at pose (p, theta), CCW in box frame."""
    half = a / 2
    local = [(+half, +half), (-half, +half), (-half, -half), (+half, -half)]
    return [(p[0] + rot(theta, v)[0], p[1] + rot(theta, v)[1]) for v in local]


def box_feature(p, theta, name, a=A):
    """World endpoints of a named box face, or the point of a named vertex."""
    half = a / 2
    faces = {
        "+x": ((+half, -half), (+half, +half)),
        "-x": ((-half, -half), (-half, +half)),
        "+z": ((-half, +half), (+half, +half)),
        "-z": ((-half, -half), (+half, -half)),
    }
    verts = {
        "v++": (+half, +half), "v+-": (+half, -half),
        "v-+": (-half, +half), "v--": (-half, -half),
    }
    if name in verts:
        r = rot(theta, verts[name])
        return (p[0] + r[0], p[1] + r[1])
    q0, q1 = faces[name]
    r0, r1 = rot(theta, q0), rot(theta, q1)
    return ((p[0] + r0[0], p[1] + r0[1]), (p[0] + r1[0], p[1] + r1[1]))


def pivot_pose(theta, a=A, h=H):
    """Box centre while pivoting about the riser top corner (0, h)."""
    start = (-a / 2, a / 2)
    rel = (start[0] - PIVOT[0], start[1] - PIVOT[1])
    r = rot(theta, rel)
    return (PIVOT[0] + r[0], PIVOT[1] + r[1])


def disk_on_face(p, theta, face, alpha=0.5, r=R_DISK, a=A):
    """Centre of a disk resting on a box face at parameter alpha in [0,1]."""
    (x0, z0), (x1, z1) = box_feature(p, theta, face, a)
    mx, mz = x0 + alpha * (x1 - x0), z0 + alpha * (z1 - z0)
    # outward normal of that face in world
    nloc = {"+x": (1, 0), "-x": (-1, 0), "+z": (0, 1), "-z": (0, -1)}[face]
    n = rot(theta, nloc)
    return (mx + r * n[0], mz + r * n[1])


# ---------------------------------------------------------------- scenery


FLOOR_DEPTH = 0.30


def draw_environment(g, x_lo=-2.4, x_hi=2.4, a=A, h=H, label=False):
    """Lower floor L, riser S, upper platform P.  Returns the lowest world z."""
    d = FLOOR_DEPTH
    g.poly([(x_lo, 0), (0, 0), (0, -d), (x_lo, -d)],
           fill=C["env_fill"], stroke=C["env"], sw=1.4)
    g.poly([(0, -d), (x_hi, -d), (x_hi, h), (0, h)],
           fill=C["env_fill"], stroke=C["env"], sw=1.4)
    # emphasise the three named patches
    g.line((x_lo, 0), (0, 0), stroke=C["env"], sw=2.6)          # F_L
    g.line((0, 0), (0, h), stroke=C["env"], sw=2.6)             # F_S
    g.line((0, h), (min(2 * a, x_hi), h), stroke=C["env"], sw=2.6)  # F_P
    if label:
        g.text((x_lo / 2, 0), "F&#8556;", size=14, fill=C["muted"], dy=18)
        g.text((0, h / 2), "F&#8347;", size=14, fill=C["muted"], dx=-15, dy=5)
        g.text((a, h), "F&#8348;", size=14, fill=C["muted"], dy=-9)
    return -d


def scene_at(g, cx, floor_y, scale, x_lo=-2.0, x_hi=2.0, label=False):
    """Place a scene and return the first pixel row safely below it."""
    g.ox, g.oy, g.s = cx, floor_y, scale
    z_lo = draw_environment(g, x_lo=x_lo, x_hi=x_hi, label=label)
    return floor_y - scale * z_lo + 30


def draw_box(g, p, theta, a=A, fill=C["box_fill"], stroke=C["box"], sw=1.8,
             mark=True, opacity=None, extra=""):
    g.poly(box_corners(p, theta, a), fill=fill, stroke=stroke, sw=sw,
           opacity=opacity, extra=extra)
    if mark:
        # body-fixed tick that makes the 90 degree rotation visible
        m0 = (p[0] + rot(theta, (0.10, 0.34))[0], p[1] + rot(theta, (0.10, 0.34))[1])
        m1 = (p[0] + rot(theta, (0.34, 0.34))[0], p[1] + rot(theta, (0.34, 0.34))[1])
        g.line(m0, m1, stroke=C["box"], sw=2.6, opacity=opacity)
    return g


def draw_disk(g, c, r=R_DISK, opacity=None, extra=""):
    return g.circle(c, r, fill=C["man_fill"], stroke=C["man"], sw=1.8,
                    opacity=opacity, extra=extra)
