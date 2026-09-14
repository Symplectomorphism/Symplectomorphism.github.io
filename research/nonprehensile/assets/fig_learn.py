"""Figures for the learned layer, the baselines, and the whole pipeline."""

from __future__ import annotations

import math

from svgkit import (A, C, H, PIVOT, R_DISK, Svg, box_feature, disk_on_face,
                    draw_box, draw_disk, draw_environment, pivot_pose,
                    scene_at)

# shared node layout, so the graph figures stay registered with one another
NODES = {
    "m_x": (256, 122, "m&#8339;", "man"),
    "m_z": (392, 122, "m&#7764;", "man"),
    "g":   (256, 242, "g", "env"),
    "g_v": (392, 242, "g&#7525;", "env"),
    "s":   (612, 122, "s", "env"),
    "t":   (748, 122, "t", "env"),
}
EDGES = [("m_x", "m_z", "tr"), ("g", "g_v", "tr"),
         ("s", "t", "sh"), ("g", "t", "ex")]
ESTYLE = {"tr": (C["box"], None, "transfer-adjacent"),
          "sh": (C["box"], "7 5", "shared object face"),
          "ex": (C["hot"], "2 4", "exclusion")}


NODE_R = 21          # must match the circle drawn by _node


def _edge(g, u, v, kind, opacity=1.0, sw=2.0):
    """An edge between two vertices, stopping at their boundaries.

    Drawing centre-to-centre and relying on the nodes being painted
    afterwards fails as soon as a node is semi-transparent, as the faded
    base layer of the live subgraph is: the line then shows through the
    disc.  Trimming by the node radius is order-independent.
    """
    (x1, y1, _, _), (x2, y2, _, _) = NODES[u], NODES[v]
    col, dash, _ = ESTYLE[kind]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    if kind == "ex":
        # already leaves and enters below the discs
        g.add(f'<path d="M {x1} {y1+NODE_R+3} C {x1+40} {y1+190} '
              f'{x2-30} {y2+210} {x2} {y2+NODE_R+3}" fill="none" '
              f'stroke="{col}" stroke-width="{sw}"{d} opacity="{opacity}"/>')
    else:
        a = math.atan2(y2 - y1, x2 - x1)
        gap = NODE_R + 1.5
        sx, sy = x1 + gap * math.cos(a), y1 + gap * math.sin(a)
        ex, ey = x2 - gap * math.cos(a), y2 - gap * math.sin(a)
        g.add(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" '
              f'y2="{ey:.1f}" stroke="{col}" stroke-width="{sw}"{d} '
              f'opacity="{opacity}"/>')


def _node(g, key, marked=False, ring=None, opacity=1.0):
    x, y, lab, kind = NODES[key]
    base = C["man"] if kind == "man" else C["env"]
    g.add(f'<circle cx="{x}" cy="{y}" r="21" '
          f'fill="{base if marked else "#ffffff"}" stroke="{base}" '
          f'stroke-width="2" opacity="{opacity}"/>')
    if ring:
        g.add(f'<circle cx="{x}" cy="{y}" r="27" fill="none" stroke="{ring}" '
              f'stroke-width="2.4" opacity="{opacity}"/>')
    g.add(f'<text x="{x}" y="{y+6}" font-size="17" text-anchor="middle" '
          f'fill="{"#ffffff" if marked else base}" font-weight="700" '
          f'font-family="\'Latin Modern Math\',Georgia,serif" '
          f'opacity="{opacity}">{lab}</text>')


def contact_graph(path):
    """Where the graph finally enters: as the GNN's message-passing substrate."""
    g = Svg(940, 424, ox=0, oy=0, scale=1, cls="fig")
    g.header("The graph is the learner's substrate, not the search space",
             "vertices are contacts; edges say which contacts exchange "
             "information", y=32, crop=72, size=18)

    for u, v, k in EDGES:
        _edge(g, u, v, k)
    for key in NODES:
        _node(g, key, marked=key in ("m_x", "g"))

    g.px_text((324, 110), "transfer-adjacent", size=12.5, fill=C["box"])
    g.px_text((324, 230), "transfer-adjacent", size=12.5, fill=C["box"])
    g.px_text((680, 110), "shared face +x&#7495;", size=12.5, fill=C["box"],
              family="'Latin Modern Math',Georgia,serif")

    y0 = 356
    for i, kind in enumerate(["tr", "sh", "ex"]):
        col, dash, name = ESTYLE[kind]
        label = {"tr": "same body pair, adjacent features",
                 "sh": "same object face, different environment body",
                 "ex": "cannot both be held"}[kind]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        g.add(f'<line x1="150" y1="{y0+i*22}" x2="190" y2="{y0+i*22}" '
              f'stroke="{col}" stroke-width="2.4"{d}/>')
        g.px_text((200, y0 + i * 22 + 5), f"{name} &#8212; {label}", size=13,
                  fill=C["muted"], anchor="start")
    g.px_text((830, 242), "filled = held", size=13, fill=C["muted"])
    g.px_text((830, 262), "hollow = not held", size=13, fill=C["muted"])
    return g.save(path, "The contact graph")


def live_subgraph(path):
    """Seeding and one-hop dilation; t becomes visible through a shared face."""
    g = Svg(940, 400, ox=0, oy=0, scale=1, cls="fig")
    g.header("A neighbourhood, not the whole catalog", y=32, crop=50)

    DURA = "8s"
    # base layer: the whole thing, faint, so a still frame still reads
    for u, v, k in EDGES:
        _edge(g, u, v, k, opacity=0.3)
    for key in NODES:
        _node(g, key, opacity=0.3)

    def reveal(t0):
        return (f'<animate attributeName="opacity" values="0;0;1;1" '
                f'keyTimes="0;{t0:.2f};{min(t0+0.07,1):.2f};1" dur="{DURA}" '
                f'repeatCount="indefinite"/>')

    for key in ("m_x", "g"):
        _node(g, key, marked=True, ring=C["primary"])
    g.add(f'<g opacity="0">{reveal(0.20)}')
    _node(g, "s", ring=C["primary"])
    g.add("</g>")
    for key, t in (("m_z", 0.44), ("g_v", 0.50), ("t", 0.58)):
        g.add(f'<g opacity="0">{reveal(t)}')
        for u, v, k in EDGES:
            if key in (u, v) and k != "ex":
                _edge(g, u, v, k, sw=3.0)
        _node(g, key)
        g.add("</g>")

    for t0, t1, txt, col in [
            (0.00, 0.20, "seed: the contacts currently held", C["primary"]),
            (0.20, 0.44, "+ the contacts inside the gap gate", C["primary"]),
            (0.44, 0.68, "+ everything one hop away on the graph", C["box"]),
            (0.68, 1.00, "t is in view long before it is reachable",
             C["violet"])]:
        op = "1" if t0 == 0 else "0"
        g.add(f'<g opacity="{op}"><animate attributeName="opacity" '
              f'values="0;0;1;1;0;0" keyTimes="0;{max(t0-0.01,0):.2f};'
              f'{t0:.2f};{t1:.2f};{min(t1+0.01,1):.2f};1" dur="{DURA}" '
              f'repeatCount="indefinite"/>')
        g.px_text((470, 348), txt, size=17, fill=col, weight="600")
        g.add("</g>")
    g.px_text((470, 378), "the shared-face edge is why the platform contact is "
              "already visible while the box is still on the floor", size=13,
              fill=C["faint"])
    return g.save(path, "The live subgraph")


def ranker(path):
    """Attention on the live subgraph produces one score per available action."""
    g = Svg(940, 356, ox=0, oy=0, scale=1, cls="fig")
    g.header("The learned layer orders the available actions", y=32, crop=50)

    # compact graph, drawn directly rather than reusing the wide layout
    pos = {"m_x": (110, 128), "m_z": (232, 128), "g": (110, 234),
           "g_v": (232, 234), "s": (352, 128), "t": (352, 234)}
    lab = {"m_x": ("m&#8339;", C["man"]), "m_z": ("m&#7764;", C["man"]),
           "g": ("g", C["env"]), "g_v": ("g&#7525;", C["env"]),
           "s": ("s", C["env"]), "t": ("t", C["env"])}
    held = {"m_x", "g"}
    links = [("m_x", "m_z", None), ("g", "g_v", None), ("s", "t", "7 5")]
    for u, v, dash in links:
        (x1, y1), (x2, y2) = pos[u], pos[v]
        d = f' stroke-dasharray="{dash}"' if dash else ""
        g.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
              f'stroke="{C["box"]}" stroke-width="2" opacity="0.6"{d}/>')
    for i, (u, v, _) in enumerate(links):
        (x1, y1), (x2, y2) = pos[u], pos[v]
        g.add(f'<circle r="5" fill="{C["primary"]}" cx="{x1}" cy="{y1}">'
              f'<animate attributeName="cx" values="{x1};{x2};{x1}" dur="3s" '
              f'begin="{i*0.4}s" repeatCount="indefinite"/>'
              f'<animate attributeName="cy" values="{y1};{y2};{y1}" dur="3s" '
              f'begin="{i*0.4}s" repeatCount="indefinite"/>'
              f'<animate attributeName="opacity" values="0;1;1;0" dur="3s" '
              f'begin="{i*0.4}s" repeatCount="indefinite"/></circle>')
    for k, (x, y) in pos.items():
        t, col = lab[k]
        on = k in held
        g.add(f'<circle cx="{x}" cy="{y}" r="20" '
              f'fill="{col if on else "#ffffff"}" stroke="{col}" '
              f'stroke-width="2"/>')
        g.add(f'<text x="{x}" y="{y+6}" font-size="16" text-anchor="middle" '
              f'fill="{"#ffffff" if on else col}" font-weight="700" '
              f'font-family="\'Latin Modern Math\',Georgia,serif">{t}</text>')
    g.px_text((231, 296), "attention over the live subgraph", size=13.5,
              fill=C["muted"])

    g.add(f'<line x1="440" y1="70" x2="440" y2="310" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')

    x0, w = 656, 232
    g.px_text((x0 + w / 2 - 40, 92),
              "score for every available action at this node", size=13.5,
              fill=C["muted"], weight="700")
    for i, (name, v, best) in enumerate([
            ("make(s)", 0.93, True), ("make(g&#7525;)", 0.61, False),
            ("transport", 0.34, False), ("break(g)", 0.09, False)]):
        y = 116 + i * 46
        col = C["primary"] if best else C["faint"]
        g.px_text((x0 - 14, y + 21), name, size=15, anchor="end",
                  fill=C["ink"] if best else C["muted"],
                  weight="700" if best else "400",
                  family="'Latin Modern Math',Georgia,serif")
        g.add(f'<rect x="{x0}" y="{y}" width="{w}" height="30" rx="5" '
              f'fill="#f4f6f4" stroke="{C["rule"]}" stroke-width="1.2"/>')
        g.add(f'<rect x="{x0}" y="{y}" width="{w*v:.0f}" height="30" rx="5" '
              f'fill="{col}" opacity="{0.9 if best else 0.4}">'
              f'<animate attributeName="width" values="0;{w*v:.0f}" dur="1s" '
              f'begin="{0.8+i*0.12}s" fill="freeze"/></rect>')
    g.add(f'<polygon points="{x0+w+8},{116} {x0+w+8},{128} {x0+w+20},{122}" '
          f'fill="{C["primary"]}"/>')
    g.px_text((x0 + w / 2 - 40, 326), "try the best one first; on rejection, fall "
              "through to the next", size=14, fill=C["ink"], weight="600")

    return g.save(path, "Ranking the available actions")


def greedy_misses(path):
    """The two hops a geometry-only rule gets wrong."""
    g = Svg(940, 412, ox=0, oy=0, scale=1, cls="fig")
    g.header("Why a learned order is worth anything",
             "a closest-gap-first rule loses the task at exactly two hops",
             y=32, crop=72)

    g.px_text((235, 92), "at the start", size=16, fill=C["ink"], weight="700")
    scene_at(g, cx=214, floor_y=228, scale=74, x_lo=-1.9, x_hi=2.0)
    p = (-0.95, A / 2)
    draw_box(g, p, 0.0)
    draw_disk(g, disk_on_face(p, 0.0, "-x", 0.55))
    g.dot(box_feature(p, 0.0, "v+-"), r_px=5.4, fill=C["hot"])
    g.line((0, 0), (0, H), stroke=C["primary"], sw=3.6)
    g.px_text((235, 286), "closest gap: g&#7525;, already touching &#160;"
              "&#10007;", size=14, fill=C["hot"],
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((235, 310), "right answer: s, still a body-width away &#160;"
              "&#10003;", size=14, fill=C["primary"], weight="600",
              family="'Latin Modern Math',Georgia,serif")

    g.add(f'<line x1="470" y1="76" x2="470" y2="326" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')

    pl = pivot_pose(math.pi / 2)
    g.px_text((705, 92), "after landing", size=16, fill=C["ink"], weight="700")
    scene_at(g, cx=684, floor_y=228, scale=74, x_lo=-1.9, x_hi=2.0)
    draw_box(g, pl, math.pi / 2)
    draw_disk(g, disk_on_face(pl, math.pi / 2, "-z", 0.5))
    g.line(*box_feature(pl, math.pi / 2, "-z"), stroke=C["primary"], sw=3.6)
    g.px_text((705, 286), "closest gap: do nothing, or release s &#160;"
              "&#10007;", size=14, fill=C["hot"],
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((705, 310), "right answer: slide the disk to the new back face "
              "&#160;&#10003;", size=14, fill=C["primary"], weight="600")

    g.add(f'<rect x="130" y="344" width="680" height="48" rx="8" '
          f'fill="{C["primary_soft"]}" stroke="{C["primary"]}" '
          f'stroke-width="1.4"/>')
    g.px_text((470, 374), "both hops pay now and benefit later &#8212; that is "
              "the combinatorial work a learned order has to do", size=14.5,
              fill=C["primary"], weight="600")
    return g.save(path, "Two hops a greedy rule misses")


def pipeline(path):
    """The whole system on one slide."""
    g = Svg(940, 432, ox=0, oy=0, scale=1, cls="fig")
    g.header("The whole system", y=28, crop=46, size=20)

    def card(x, y, w, h, title, body, col, fill="#ffffff", dash=False):
        d = ' stroke-dasharray="6 4"' if dash else ""
        g.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" '
              f'fill="{fill}" stroke="{col}" stroke-width="1.8"{d}/>')
        g.px_text((x + w / 2, y + 26), title, size=15, fill=col, weight="700")
        for i, line in enumerate(body):
            g.px_text((x + w / 2, y + 50 + i * 19), line, size=12.5,
                      fill=C["muted"])

    def arrow(x1, x2, y):
        g.add(f'<line x1="{x1}" y1="{y}" x2="{x2-9}" y2="{y}" '
              f'stroke="{C["faint"]}" stroke-width="1.8"/>'
              f'<polygon points="{x2},{y} {x2-9},{y-5} {x2-9},{y+5}" '
              f'fill="{C["faint"]}"/>')

    g.px_text((74, 92), "built once, offline", size=13, fill=C["faint"],
              weight="700", anchor="start")
    card(72, 102, 176, 92, "scene", ["bodies, features,", "friction, simulator"],
         C["env"])
    card(276, 102, 176, 92, "catalog + graph",
         ["every feature pair,", "three edge types"], C["box"])
    card(480, 102, 186, 92, "teacher",
         ["tries every legal action,", "records what worked"], C["ink"])
    card(694, 102, 174, 92, "learned ranker",
         ["GNN over the graph,", "orders the actions"], C["primary"],
         fill=C["primary_soft"], dash=True)
    g.px_text((838, 212), "optional", size=12, fill=C["primary"],
              weight="700")
    for x1, x2 in [(248, 276), (452, 480), (666, 694)]:
        arrow(x1, x2, 148)

    g.px_text((74, 254), "run at a query", size=13, fill=C["faint"],
              weight="700", anchor="start")
    card(72, 264, 176, 92, "state", ["pose &#967; and", "bit vector &#963;"],
         C["ink"])
    card(276, 264, 176, 92, "available actions",
         ["gap gate,", "support test"], C["gold"])
    card(480, 264, 186, 92, "order the candidates",
         ["fixed rule, or the", "learned ranker"], C["primary"],
         fill=C["primary_soft"], dash=True)
    card(694, 264, 174, 92, "&#923;", ["short-horizon MPC,", "accept or reject"],
         C["ink"])
    for x1, x2 in [(248, 276), (452, 480), (666, 694)]:
        arrow(x1, x2, 310)

    def head(at, ang, col, size=10.0):
        """Arrowhead at `at`, aimed along `ang`.  Derived from the path's
        own end tangent rather than placed by hand, which is how the two
        loops here came to disagree with the curves they terminated."""
        w = size * 0.42
        bx, by = at[0] - size * math.cos(ang), at[1] - size * math.sin(ang)
        g.add(f'<polygon points="{at[0]:.1f},{at[1]:.1f} '
              f'{bx-w*math.sin(ang):.1f},{by+w*math.cos(ang):.1f} '
              f'{bx+w*math.sin(ang):.1f},{by-w*math.cos(ang):.1f}" '
              f'fill="{col}"/>')

    def bez(p0, p1, p2, p3, t):
        u = 1.0 - t
        return tuple(u**3 * a + 3 * u * u * t * b + 3 * u * t * t * c
                     + t**3 * d for a, b, c, d in zip(p0, p1, p2, p3))

    # the learned ranker feeds the ordering stage, not the solver.  The
    # curve lands on the ordering card at a slant, and the head takes its
    # angle from the curve rather than from the endpoint alone -- with the
    # last handle vertical the true tangent straightens in the final few
    # pixels and the head no longer matches the stroke the eye follows.
    P = (781, 194), (781, 230), (660, 226), (605, 262)
    g.add(f'<path d="M {P[0][0]} {P[0][1]} C {P[1][0]} {P[1][1]} '
          f'{P[2][0]} {P[2][1]} {P[3][0]} {P[3][1]}" fill="none" '
          f'stroke="{C["primary"]}" stroke-width="1.8"/>')
    near = bez(*P, 0.94)
    head(P[3], math.atan2(P[3][1] - near[1], P[3][0] - near[0]), C["primary"])

    # the reject loop returns to the state.  Straight segments, so the
    # final approach is visibly vertical and the head cannot look detached
    g.add(f'<path d="M 781 356 L 781 398 L 160 398 L 160 368" fill="none" '
          f'stroke="{C["faint"]}" stroke-width="1.8" stroke-dasharray="5 4" '
          f'stroke-linejoin="round"/>')
    head((160, 357), -math.pi / 2, C["faint"])
    g.px_text((470, 420), "the solid path is a complete planner; the dashed "
              "box only changes the order the list is tried in",
              size=13, fill=C["muted"])
    return g.save(path, "System overview")


def budget(path):
    """What gets measured, and against what."""
    g = Svg(940, 410, ox=0, oy=0, scale=1, cls="fig")
    g.header("One number decides whether any of this is worth it", y=32,
             crop=46)
    g.px_text((470, 58), "N&#923; = calls to the local solver, from the query "
              "until the goal is reached", size=15, fill=C["primary"],
              weight="600",
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    bars = [
        ("no discrete search at all", 0.0, C["violet"], "run this first"),
        ("a different discrete object", 0.0, C["violet"],
         "contact intention + subgoal"),
        ("teacher, tries everything", 0.92, C["ink"], ""),
        ("closest-gap-first", 0.74, C["faint"], "misses two hops"),
        ("learned ranker", 0.40, C["primary"], "the claim to test"),
        ("shortest certified plan", 0.22, C["gold"], "the floor"),
    ]
    x0, y0, w = 316, 86, 384
    for i, (name, v, col, note) in enumerate(bars):
        y = y0 + i * 44
        g.px_text((x0 - 16, y + 21), name, size=13.5, anchor="end",
                  fill=C["ink"])
        g.add(f'<rect x="{x0}" y="{y}" width="{w}" height="30" rx="5" '
              f'fill="#f4f6f4" stroke="{C["rule"]}" stroke-width="1"/>')
        if v > 0:
            g.add(f'<rect x="{x0}" y="{y}" width="{w*v:.0f}" height="30" '
                  f'rx="5" fill="{col}" opacity="0.85">'
                  f'<animate attributeName="width" values="0;{w*v:.0f}" '
                  f'dur="1.1s" begin="{0.3+i*0.14}s" fill="freeze"/></rect>')
        else:
            g.px_text((x0 + 12, y + 20), "not yet measured", size=13,
                      anchor="start", fill=C["violet"], weight="600")
        if note:
            g.px_text((x0 + w + 12, y + 20), note, size=12.5, anchor="start",
                      fill=C["faint"])

    g.px_text((470, 374), "the learned layer earns its place only if it sits "
              "below every baseline,", size=13.5, fill=C["muted"])
    g.px_text((470, 394), "at equal budget and equal success rate", size=13.5,
              fill=C["muted"])
    return g.save(path, "The measurement")


# ------------------------------------------------- where the solver calls go

def search_cost(path):
    """Where the calls go, without inventing a rejection rate.

    Two things are actually derivable and nothing else is asserted: the
    candidate list at a node has at most |C| + 1 entries before gating, and
    the calls spent at that node equal the rank of the first action that
    works.  How many candidates survive the gates, and how often Lambda
    rejects an admissible one, are the unmeasured quantities.
    """
    g = Svg(940, 452, ox=0, oy=0, scale=1, cls="fig")
    g.header("What a call is spent on", y=30, crop=48)

    # ---- the funnel, top row
    g.px_text((92, 78), "at one node", size=13.5, fill=C["muted"],
              weight="700")
    g.px_text((92, 98), "(&#967;, &#963;)", size=15, fill=C["ink"],
              family="'Latin Modern Math','STIX Two Math',Georgia,serif")

    cw, gp = 15, 3
    def grid(x0, y0, n, lit, col):
        for i in range(n):
            r, c = divmod(i, 6)
            on = i < lit
            g.add(f'<rect x="{x0+c*(cw+gp)}" y="{y0+r*(cw+gp)}" '
                  f'width="{cw}" height="{cw}" rx="2.5" '
                  f'fill="{col if on else "#f4f6f4"}" '
                  f'stroke="{col if on else C["faint"]}" stroke-width="1.1" '
                  f'opacity="{1 if on else 0.55}"/>')

    grid(176, 70, 36, 36, C["box"])
    g.px_text((228, 190), "|&#119966;| + 1 = 36", size=14, fill=C["ink"],
              weight="700", family="'Latin Modern Math',Georgia,serif")
    g.px_text((228, 210), "single-bit flips, plus transport", size=12.5,
              fill=C["muted"])

    g.add(f'<line x1="300" y1="122" x2="356" y2="122" stroke="{C["faint"]}" '
          f'stroke-width="1.8"/><polygon points="366,122 356,117 356,127" '
          f'fill="{C["faint"]}"/>')
    g.px_text((333, 110), "gap gate", size=12, fill=C["gold"], weight="700")
    g.px_text((333, 138), "support test", size=12, fill=C["gold"],
              weight="700")

    grid(384, 70, 36, 7, C["primary"])
    g.px_text((436, 132), "?", size=44, fill=C["hot"], weight="700",
              extra='opacity="0.55"')
    g.px_text((436, 190), "? admissible", size=14, fill=C["primary"],
              weight="700")
    g.px_text((436, 210), "how many survive the gates is unmeasured",
              size=12.5, fill=C["hot"])

    g.add(f'<line x1="508" y1="122" x2="564" y2="122" stroke="{C["faint"]}" '
          f'stroke-width="1.8"/><polygon points="574,122 564,117 564,127" '
          f'fill="{C["faint"]}"/>')
    g.px_text((541, 110), "tried in", size=12, fill=C["muted"], weight="700")
    g.px_text((541, 138), "rank order", size=12, fill=C["muted"],
              weight="700")

    # ---- the same admissible set, two orders
    rows = 4
    for k, (title, hit, col) in enumerate(
            [("gap-greedy order", 3, C["faint"]),
             ("a learned order", 0, C["primary"])]):
        x0 = 596 + k * 172
        g.px_text((x0 + 60, 78), title, size=13, fill=C["ink"], weight="700")
        for i in range(rows):
            y = 92 + i * 26
            done = i <= hit
            g.add(f'<rect x="{x0}" y="{y}" width="120" height="21" rx="4" '
                  f'fill="{"#f2f8f4" if i == hit else "#ffffff"}" '
                  f'stroke="{col if i == hit else C["rule"]}" '
                  f'stroke-width="{1.6 if i == hit else 1.1}"/>')
            mark = "&#10003;" if i == hit else ("&#10007;" if done else "")
            mc = C["primary"] if i == hit else C["hot"]
            g.px_text((x0 + 13, y + 15), mark, size=13, fill=mc, weight="700")
            if done:
                g.px_text((x0 + 108, y + 15), "1 call", size=11,
                          fill=C["muted"], anchor="end")
        g.px_text((x0 + 60, 92 + rows * 26 + 16),
                  f"{hit+1} &#923; call{'s' if hit else ''} spent here",
                  size=13.5, fill=col if k else C["hot"], weight="700")

    g.px_text((470, 248), "same node, same admissible set, different order",
              size=13, fill=C["faint"])
    # "closest" needs saying in terms of the quantity that defines it
    g.px_text((470, 268), "gap-greedy scores a make by &#8722;&#966;(added "
              "contact), so the smallest remaining gap wins;", size=12.5,
              fill=C["muted"])
    g.px_text((470, 285), "breaks and transport add nothing and all score 0",
              size=12.5, fill=C["muted"])
    g.px_text((470, 304), "ranks drawn for illustration: no rejection rate is "
              "claimed here", size=12, fill=C["hot"])

    g.add(f'<rect x="150" y="322" width="640" height="52" rx="8" '
          f'fill="{C["primary_soft"]}" stroke="{C["primary"]}" '
          f'stroke-width="1.6"/>')
    g.px_text((470, 344), "calls spent at a node = rank of the first action "
              "that works", size=16, fill=C["ink"], weight="700")
    g.px_text((470, 364), "that rank is the only quantity a learned layer "
              "changes", size=13, fill=C["primary"], weight="600")
    g.px_text((470, 396), "how often &#923; rejects an admissible action is "
              "also unmeasured &#8212; calibrate it before trusting any of "
              "these counts", size=13, fill=C["hot"])
    return g.save(path, "What a call is spent on")
