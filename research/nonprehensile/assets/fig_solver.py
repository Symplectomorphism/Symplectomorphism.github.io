"""Figures for realization: the local solver, the compiled cost, the gates."""

from __future__ import annotations

import math

import sim_release
from svgkit import (A, C, H, PIVOT, R_DISK, Svg, box_feature, disk_on_face,
                    draw_box, draw_disk, draw_environment, pivot_pose, rot,
                    scene_at)


MU_S = 0.4      # Coulomb coefficient used to draw the riser's cone


def _cone(g, y, nu, mu, r=0.56, col=None, label=None, lab_dy=0):
    """Filled friction cone at contact point `y` about the inward normal."""
    col = col or C["gold"]
    half = math.atan(mu)
    a = math.atan2(nu[1], nu[0])
    pts = [y]
    for k in range(13):
        t = a - half + 2 * half * k / 12
        pts.append((y[0] + r * math.cos(t), y[1] + r * math.sin(t)))
    g.poly(pts, fill=col, stroke="none", sw=0, opacity=0.30)
    g.poly(pts, fill="none", stroke=col, sw=1.6)
    g.arrow((y[0], y[1]), (y[0] + r * 0.80 * nu[0], y[1] + r * 0.80 * nu[1]),
            stroke=col, sw=2.4, head=7.0)
    if label:
        # sit the caption just outside the upper ray, clear of the wedge
        t = a - half
        g.text((y[0] + r * 1.18 * math.cos(t), y[1] + r * 1.18 * math.sin(t)),
               label, size=12, fill=col, weight="700", dy=lab_dy)


_RELEASE_CACHE = {}


def _release_sim(pm, th):
    """Simulated release, cached: the same motion serves both figures."""
    key = (round(pm[0], 6), round(pm[1], 6), round(th, 6))
    if key not in _RELEASE_CACHE:
        _RELEASE_CACHE[key] = sim_release.release_frames(
            pm, th, n=44, a=A, h=H, t_end=0.95)
    return _RELEASE_CACHE[key]


def _release_failure(g, pm, th, dur):
    """What actually happens when the manipulator lets go.

    The poses are integrated, not keyframed by hand: a penalty-contact model
    of the box against the step, released from rest (`sim_release`).  It
    slides on the step corner for about a fifth of a second, loses that
    contact, keeps rotating, and lands flat on the lower floor well clear of
    the riser -- so contact s is genuinely gone, exactly as the statics say.
    """
    frames, lost, traj = _release_sim(pm, th)
    t_end = frames[-1][0]
    # Contact is broken the instant the manipulator STARTS to move away, so
    # the box must start falling at that same instant -- not when the
    # withdrawal finishes.  Both begin at T0.
    T0, T1, THOLD = 0.10, 0.74, 0.94

    def loop_t(t_sim):
        return T0 + (T1 - T0) * (t_sim / t_end)

    kt, tr, rt = [0.0, T0], [], []
    for _ in range(2):
        tr.append("0,0")
        rt.append(f"0 {g.P(*pm)[0]:.1f} {g.P(*pm)[1]:.1f}")
    for t_sim, p, thi in frames[1:]:
        kt.append(loop_t(t_sim))
        tr.append(f"{g.dx(p[0] - pm[0]):.1f},{-(p[1] - pm[1]) * g.s:.1f}")
        rt.append(f"{g.rot_deg(math.degrees(thi - th)):.1f} "
                  f"{g.P(*pm)[0]:.1f} {g.P(*pm)[1]:.1f}")
    kt += [THOLD, 1.0]
    tr += [tr[-1], tr[1]]
    rt += [rt[-1], rt[1]]
    kts = ";".join(f"{v:.4f}" for v in kt)

    g.add(f'<g><animateTransform attributeName="transform" type="translate" '
          f'values="{";".join(tr)}" keyTimes="{kts}" dur="{dur}" '
          f'repeatCount="indefinite"/>')
    g.add(f'<g><animateTransform attributeName="transform" type="rotate" '
          f'values="{";".join(rt)}" keyTimes="{kts}" dur="{dur}" '
          f'repeatCount="indefinite"/>')
    draw_box(g, pm, th)
    g.add("</g>")
    # gravity rides the centre of mass but never rotates: it is world-fixed
    g.arrow(pm, (pm[0], pm[1] - 0.52), stroke=C["hot"], sw=2.6, head=8)
    g.text((pm[0], pm[1] - 0.52), "mg", size=13.5, fill=C["hot"], dy=16,
           dx=12, weight="700", family="'Latin Modern Math',Georgia,serif")
    g.add("</g>")

    # everything the riser can supply, until the contact is gone
    t_lost = loop_t(lost) if lost else T1
    g.add(f'<g opacity="1"><animate attributeName="opacity" '
          f'values="1;1;0;0;1" keyTimes="0;{t_lost:.3f};'
          f'{min(t_lost+0.03,1):.3f};{THOLD:.2f};1" dur="{dur}" '
          f'repeatCount="indefinite"/>')
    _cone(g, PIVOT, (-1.0, 0.0), MU_S, label="friction cone at s", lab_dy=-8)
    g.add("</g>")
    g.dot(PIVOT, r_px=5.0)

    # the manipulator withdraws along the outward normal of the face it held
    n = rot(th, (-1, 0))
    d0 = disk_on_face(pm, th, "-x", 0.74)
    away = (g.dx(0.62 * n[0]), -0.62 * n[1] * g.s)
    g.add(f'<g><animateTransform attributeName="transform" type="translate" '
          f'values="0,0;0,0;{away[0]:.1f},{away[1]:.1f};'
          f'{away[0]:.1f},{away[1]:.1f};0,0" '
          f'keyTimes="0;{T0:.2f};{T0+0.13:.2f};{THOLD:.2f};1" dur="{dur}" '
          f'calcMode="spline" '
          f'keySplines="0 0 1 1;0.35 0 0.7 1;0 0 1 1;0 0 1 1" '
          f'repeatCount="indefinite"/>')
    draw_disk(g, d0)
    g.add("</g>")


def solver(path):
    """One action, compiled to a cost, handed to a short-horizon optimizer."""
    g = Svg(940, 372, ox=0, oy=0, scale=1, cls="fig")
    g.header("One action, compiled into a cost, handed to an optimizer",
             y=32, crop=50, size=18)

    g.add(f'<rect x="26" y="58" width="190" height="172" rx="8" '
          f'fill="{C["primary_soft"]}" stroke="{C["primary"]}" '
          f'stroke-width="1.8"/>')
    g.px_text((121, 84), "action", size=13.5, fill=C["primary"], weight="700")
    g.px_text((121, 116), "make(s)", size=20, fill=C["ink"], weight="600",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((121, 148), "keep &#160;{ m&#8339;, g }", size=15,
              fill=C["muted"], family="'Latin Modern Math',Georgia,serif")
    g.px_text((121, 172), "add &#160;&#160;{ s }", size=15, fill=C["muted"],
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((121, 206), "from pose &#967;", size=14, fill=C["muted"],
              family="'Latin Modern Math',Georgia,serif")

    g.add(f'<line x1="224" y1="144" x2="272" y2="144" stroke="{C["ink"]}" '
          f'stroke-width="2"/>'
          f'<polygon points="282,144 270,138 270,150" fill="{C["ink"]}"/>')

    g.add(f'<rect x="286" y="58" width="284" height="172" rx="8" '
          f'fill="#ffffff" stroke="{C["rule"]}" stroke-width="1.8"/>')
    g.px_text((428, 84), "compiled running cost", size=13.5, fill=C["muted"],
              weight="700")
    for i, (t, col) in enumerate([
            ("keep every held gap in band", C["primary"]),
            ("drive the added gap to zero", C["primary"]),
            ("(or the dropped gap out of band)", C["muted"]),
            ("penalize control effort", C["muted"])]):
        g.add(f'<circle cx="308" cy="{110 + i*26}" r="3.4" fill="{col}"/>')
        g.px_text((320, 115 + i * 26), t, size=13.5, fill=C["ink"],
                  anchor="start")
    g.px_text((428, 216), "terminal test on the executed record", size=12.5,
              fill=C["faint"])

    g.add(f'<line x1="578" y1="144" x2="626" y2="144" stroke="{C["ink"]}" '
          f'stroke-width="2"/>'
          f'<polygon points="636,144 624,138 624,150" fill="{C["ink"]}"/>')

    g.add(f'<rect x="640" y="58" width="274" height="172" rx="8" '
          f'fill="#ffffff" stroke="{C["ink"]}" stroke-width="1.8"/>')
    g.px_text((777, 84), "&#923; &#160; local solver", size=16, fill=C["ink"],
              weight="700", family="'Latin Modern Math',Georgia,serif")
    g.px_text((777, 104), "sampling MPC over the simulator", size=12.5,
              fill=C["muted"])
    # rollout fan: drawn in full so a still frame is still legible
    for k in range(9):
        sp = (k - 4) * 8
        g.add(f'<path d="M 678 202 Q 744 {180 + sp} 812 {172 + sp*1.8}" '
              f'fill="none" stroke="{C["faint"]}" stroke-width="1.3" '
              f'opacity="0.5"><animate attributeName="stroke-dasharray" '
              f'values="0 300;300 0" dur="2.6s" begin="{k*0.06}s" '
              f'repeatCount="indefinite"/></path>')
    g.add(f'<path d="M 678 202 Q 746 176 816 158" fill="none" '
          f'stroke="{C["primary"]}" stroke-width="2.8">'
          f'<animate attributeName="stroke-dasharray" values="0 300;300 0" '
          f'dur="2.6s" begin="0.5s" repeatCount="indefinite"/></path>')
    g.add(f'<circle cx="678" cy="202" r="4.5" fill="{C["ink"]}"/>')
    g.px_text((777, 218), "one call = one budget unit", size=12.5,
              fill=C["faint"])

    g.add(f'<rect x="286" y="262" width="200" height="48" rx="8" '
          f'fill="#f2f8f4" stroke="{C["primary"]}" stroke-width="1.6"/>')
    g.px_text((386, 292), "accept &#8594; trajectory segment", size=13.5,
              fill=C["primary"], weight="600")
    g.add(f'<rect x="506" y="262" width="200" height="48" rx="8" '
          f'fill="{C["hot_soft"]}" stroke="{C["hot"]}" stroke-width="1.6"/>')
    g.px_text((606, 292), "reject &#8594; try the next action", size=13.5,
              fill=C["hot"], weight="600")
    g.add(f'<path d="M 777 232 L 777 248 L 606 248 L 606 258" fill="none" '
          f'stroke="{C["faint"]}" stroke-width="1.4"/>'
          f'<path d="M 777 232 L 777 248 L 386 248 L 386 258" fill="none" '
          f'stroke="{C["faint"]}" stroke-width="1.4"/>')

    g.px_text((470, 348), "the cost of a plan is the number of &#923; calls "
              "&#8212; the one number we report", size=15, fill=C["ink"],
              weight="700")
    return g.save(path, "The local solver")


def gates(path):
    """The two cheap tests applied before the solver is ever called."""
    g = Svg(940, 400, ox=0, oy=0, scale=1, cls="fig")
    g.header("Two cheap tests run before the solver is ever called", y=32,
             crop=50)

    # --- gate 1
    g.px_text((235, 66), "1 &#160; gap gate", size=17, fill=C["ink"],
              weight="700")
    g.px_text((235, 88), "a distant contact cannot be closed in one horizon",
              size=13, fill=C["muted"])
    scene_at(g, cx=214, floor_y=234, scale=70, x_lo=-1.9, x_hi=2.0)
    p = (-1.18, A / 2)
    draw_box(g, p, 0.0)
    draw_disk(g, disk_on_face(p, 0.0, "-x", 0.55))
    g.line((0, 0), (0, H), stroke=C["violet"], sw=3.6)
    g.line((0, H), (2 * A, H), stroke=C["faint"], sw=3.0, dash="3 4")
    g.line((-0.68, 0.16), (0.0, 0.16), stroke=C["hot"], sw=1.5, dash="4 3")
    g.text((-0.34, 0.16), "&#966;&#8347;", size=14, fill=C["hot"], dy=-8,
           weight="700", family="'Latin Modern Math',Georgia,serif")
    g.px_text((235, 288), "&#966;&#8347; is inside the gate &#160;&#10003;"
              "&#160; make(s) may be tried", size=14, fill=C["primary"],
              weight="600", family="'Latin Modern Math',Georgia,serif")
    g.px_text((235, 312), "&#966;&#8348; is not &#160;&#10007;&#160; make(t) "
              "is withheld for now", size=14, fill=C["faint"],
              family="'Latin Modern Math',Georgia,serif")

    g.add(f'<line x1="470" y1="56" x2="470" y2="340" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')

    # --- gate 2
    g.px_text((705, 66), "2 &#160; support test", size=17, fill=C["ink"],
              weight="700")
    g.px_text((705, 88), "you may not release your last support", size=13,
              fill=C["muted"])
    th = math.radians(38)
    pm = pivot_pose(th)
    scene_at(g, cx=684, floor_y=234, scale=70, x_lo=-1.9, x_hi=2.0)
    _release_failure(g, pm, th, "6s")
    g.text(PIVOT, "s", size=14, fill=C["hot"], dx=-16, dy=-7, weight="700",
           family="'Latin Modern Math',Georgia,serif")
    g.px_text((705, 288), "keep-set after break(m&#8339;) is { s }", size=14,
              fill=C["ink"], family="'Latin Modern Math',Georgia,serif")
    g.px_text((705, 312), "the cone about a horizontal normal holds no "
              "vertical force", size=13.5, fill=C["hot"], weight="600")

    return g.save(path, "The gap gate and the support test")


def support_theorem(path):
    """Mid-topple release fails; post-landing release succeeds."""
    g = Svg(940, 406, ox=0, oy=0, scale=1, cls="fig")
    g.header("The support test decides when the manipulator may let go",
             y=32, crop=50)

    th = math.radians(38)
    pm = pivot_pose(th)
    g.px_text((235, 68), "mid-topple &#160; &#963; = { m&#8339;, s }", size=16,
              fill=C["ink"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    scene_at(g, cx=214, floor_y=250, scale=84, x_lo=-1.9, x_hi=2.0)
    _release_failure(g, pm, th, "6.5s")
    g.px_text((235, 312), "break(m&#8339;) &#160; REFUSED", size=17,
              fill=C["hot"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((235, 336), "no equilibrium: the box tumbles off the riser and "
              "drops to the floor", size=13.5, fill=C["muted"])

    g.add(f'<line x1="470" y1="56" x2="470" y2="352" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')

    pl = pivot_pose(math.pi / 2)
    g.px_text((705, 68), "landed &#160; &#963; = { m&#8339;, s, t }", size=16,
              fill=C["ink"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    scene_at(g, cx=684, floor_y=250, scale=84, x_lo=-1.9, x_hi=2.0)
    draw_box(g, pl, math.pi / 2)
    # the manipulator lifts straight off the top face, as on the left panel
    nl = rot(math.pi / 2, (-1, 0))
    dl = disk_on_face(pl, math.pi / 2, "-x", 0.5)
    away_l = (g.dx(0.55 * nl[0]), -0.55 * nl[1] * g.s)
    g.add(f'<g><animateTransform attributeName="transform" type="translate" '
          f'values="0,0;0,0;{away_l[0]:.1f},{away_l[1]:.1f};'
          f'{away_l[0]:.1f},{away_l[1]:.1f};0,0" '
          f'keyTimes="0;0.10;0.32;0.96;1" dur="6.5s" calcMode="spline" '
          f'keySplines="0 0 1 1;0.35 0 0.7 1;0 0 1 1;0 0 1 1" '
          f'repeatCount="indefinite"/>')
    draw_disk(g, dl)
    g.add("</g>")
    f = box_feature(pl, math.pi / 2, "+x")
    g.line((0.0, H), (max(f[0][0], f[1][0]), H), stroke=C["hot"], sw=3.8)
    g.dot(PIVOT, r_px=5.2)
    # here a supporting force does exist: the platform's cone is vertical and
    # the centre of mass projects inside the contact patch
    _cone(g, (pl[0], H), (0.0, 1.0), MU_S, r=0.52, col=C["primary"],
          label="friction cone at t", lab_dy=-6)
    g.arrow(pl, (pl[0], pl[1] - 0.30), stroke=C["hot"], sw=2.6, head=8)
    g.text((pl[0], pl[1] - 0.30), "mg", size=13.5, fill=C["hot"], dy=15,
           dx=14, weight="700", family="'Latin Modern Math',Georgia,serif")
    g.px_text((705, 312), "break(m&#8339;) &#160; ADMITTED", size=17,
              fill=C["primary"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((705, 336), "the platform's cone does, and the centre of mass "
              "lies over the patch", size=13.5, fill=C["muted"])

    g.px_text((470, 380), "the refusal is decided by statics; the motion is "
              "integrated from the released state, not drawn by hand",
              size=13.5, fill=C["faint"])
    return g.save(path, "When the manipulator may let go")


def normals_m3(path):
    """Why make(m_z) must precede break(s)."""
    g = Svg(940, 406, ox=0, oy=0, scale=1, cls="fig")
    g.header("Geometry fixes the order of the last two actions", y=32,
             crop=50)

    pl = pivot_pose(math.pi / 2)
    th = math.pi / 2
    for col, face, head, sub, verdict, cc in [
            (0, "-x", "disk on &#8722;x&#7495;, now the top face",
             "contact normal is vertical",
             "can only press the box into the platform", C["hot"]),
            (1, "-z", "disk on &#8722;z&#7495;, now the back face",
             "contact normal is horizontal",
             "can push the box toward the goal", C["primary"])]:
        cx = 235 + col * 470
        g.px_text((cx, 70), head, size=15.5, fill=C["ink"], weight="700",
                  family="'Latin Modern Math',Georgia,serif")
        g.px_text((cx, 92), sub, size=13, fill=C["muted"])
        scene_at(g, cx=cx - 30, floor_y=282, scale=84, x_lo=-1.4, x_hi=1.9)
        draw_box(g, pl, th)
        d = disk_on_face(pl, th, face, 0.5)
        draw_disk(g, d)
        fp = box_feature(pl, th, face)
        g.line(fp[0], fp[1], stroke=cc, sw=4.0)
        n = rot(th, {"-x": (-1, 0), "-z": (0, -1)}[face])
        mid = ((fp[0][0] + fp[1][0]) / 2, (fp[0][1] + fp[1][1]) / 2)
        # slide the normal arrow along the face so it misses the disk
        t_hat = (-n[1], n[0])
        base = (mid[0] + 0.40 * t_hat[0], mid[1] + 0.40 * t_hat[1])
        g.arrow((base[0] + 0.46 * n[0], base[1] + 0.46 * n[1]), base,
                stroke=cc, sw=2.4)
        g.px_text((cx, 330), verdict, size=14, fill=cc, weight="600")

    g.add(f'<line x1="470" y1="56" x2="470" y2="344" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')
    g.px_text((470, 372), "so make(m&#7764;) must precede break(s)", size=16,
              fill=C["primary"], weight="700",
              family="'Latin Modern Math',Georgia,serif")
    g.px_text((470, 394), "a fact about this pose, which no fixed ordering of "
              "actions can encode", size=13, fill=C["faint"])
    return g.save(path, "Contact normals after landing")
