"""Hero figures: the running task, and the gap in what prior work searches."""

from __future__ import annotations

import math

from svgkit import (A, C, H, PIVOT, R_DISK, Svg, box_feature, disk_on_face,
                    draw_box, draw_disk, draw_environment, pivot_pose,
                    scene_at)

DUR = 16.0
#    push  dwell pivot  land  reloc  goal  hold
KT = [0.00, 0.15, 0.21, 0.53, 0.61, 0.73, 0.92, 1.00]


def task_animation(path):
    """The whole task in one looping SVG, with the marking read out beneath."""
    g = Svg(940, 476, ox=0, oy=0, scale=1, cls="fig fig-task")
    g.header("One task, nine discrete decisions",
             "the planner is given the goal, and invents the contacts",
             y=36, crop=76, size=21)

    scene_at(g, cx=486, floor_y=344, scale=150, x_lo=-2.78, x_hi=2.10)
    s = g.s

    # goal band on the platform
    gx = 1.0
    g.poly([(gx - 0.52, H), (gx + 0.52, H), (gx + 0.52, H + A), (gx - 0.52, H + A)],
           fill=C["primary"], stroke="none", opacity=0.08)
    g.line((gx - 0.52, H + A), (gx + 0.52, H + A), stroke=C["primary"], sw=1.3,
           dash="6 5", opacity=0.55)
    g.text((gx, H + A), "goal set &#119970;(q)", size=14, fill=C["primary"],
           dy=-11, weight="600")

    # reference pose: box flush against the riser at theta = 0
    p_ref = (-A / 2, A / 2)
    dx_in = g.dx(-1.32)          # start this far back along -x_W
    dx_goal = g.dx(gx - A / 8)   # after the topple the centre sits at a/8
    rot90 = g.rot_deg(90)
    pc = f"{g.P(*PIVOT)[0]:.1f} {g.P(*PIVOT)[1]:.1f}"
    kt = ";".join(f"{v:.3f}" for v in KT)

    tr = (f"{dx_in:.1f},0; 0,0; 0,0; 0,0; 0,0; 0,0; "
          f"{dx_goal:.1f},0; {dx_goal:.1f},0")
    rv = ";".join([f"0 {pc}"] * 3 + [f"{rot90:.0f} {pc}"] * 5)
    g.add(f'<g transform="translate({dx_in:.1f},0)">'
          f'<animateTransform attributeName="transform" type="translate" '
          f'values="{tr}" keyTimes="{kt}" dur="{DUR}s" calcMode="spline" '
          f'keySplines="0.4 0 0.25 1;0 0 1 1;0 0 1 1;0 0 1 1;0 0 1 1;'
          f'0.4 0 0.25 1;0 0 1 1" repeatCount="indefinite"/>')
    g.add(f'<g><animateTransform attributeName="transform" type="rotate" '
          f'values="{rv}" keyTimes="{kt}" dur="{DUR}s" calcMode="spline" '
          f'keySplines="0 0 1 1;0 0 1 1;0.45 0 0.3 1;0 0 1 1;0 0 1 1;'
          f'0 0 1 1;0 0 1 1" repeatCount="indefinite"/>')

    draw_box(g, p_ref, 0.0)

    # disk rides the -x face, then walks around the v-- corner onto -z
    d0 = disk_on_face(p_ref, 0.0, "-x", alpha=0.74)
    via = (d0[0], -R_DISK)
    d1 = disk_on_face(p_ref, 0.0, "-z", alpha=0.5)
    v2 = (g.dx(via[0] - d0[0]), -(via[1] - d0[1]) * s)
    v3 = (g.dx(d1[0] - d0[0]), -(d1[1] - d0[1]) * s)
    dv = (f"0,0; 0,0; 0,0; 0,0; 0,0; {v2[0]:.1f},{v2[1]:.1f}; "
          f"{v3[0]:.1f},{v3[1]:.1f}; {v3[0]:.1f},{v3[1]:.1f}")
    ktd = (f"{KT[0]:.3f};{KT[1]:.3f};{KT[2]:.3f};{KT[3]:.3f};{KT[4]:.3f};"
           f"{KT[4]+0.05:.3f};{KT[5]:.3f};{KT[7]:.3f}")
    g.add(f'<g><animateTransform attributeName="transform" type="translate" '
          f'values="{dv}" keyTimes="{ktd}" dur="{DUR}s" '
          f'repeatCount="indefinite"/>')
    draw_disk(g, d0)
    g.add("</g></g></g>")

    # marking read-out, one line per phase
    labels = [
        (KT[0], KT[1], "&#963;&#8320; = { m&#8339;, g }", "push along the floor"),
        (KT[1], KT[2], "&#963;&#8321; = { m&#8339;, g, s }", "riser reached"),
        (KT[2], KT[3], "&#963;&#8323; = { m&#8339;, s, g&#7525; }",
         "pivot about the riser top"),
        (KT[3], KT[4], "&#963;&#8325; = { m&#8339;, s, t }",
         "landed on the platform"),
        (KT[4], KT[5], "&#963;&#8327; = { m&#7764;, s, t }",
         "disk relocated to the new back face"),
        (KT[5], KT[7], "&#963;&#8328; = { m&#7764;, t }", "transport to the goal"),
    ]
    for t0, t1, setlab, phase in labels:
        e = 0.006
        k = f"0;{max(t0-e,0):.3f};{t0:.3f};{t1:.3f};{min(t1+e,1):.3f};1"
        op = "1" if t0 == 0 else "0"
        g.add(f'<g opacity="{op}"><animate attributeName="opacity" '
              f'values="0;0;1;1;0;0" keyTimes="{k}" dur="{DUR}s" '
              f'repeatCount="indefinite"/>')
        g.px_text((470, 424), setlab, size=22, fill=C["primary"], weight="600",
                  family="'Latin Modern Math','STIX Two Math',Georgia,serif")
        g.px_text((470, 450), phase, size=15, fill=C["muted"])
        g.add("</g>")
    return g.save(path, "Box over a step: the running task")


def panel_strip(path):
    """Five static snapshots with the asserted contacts marked."""
    W = 1180
    g = Svg(W, 306, ox=0, oy=0, scale=1, cls="fig fig-strip")
    cell = W / 5
    poses = [
        ("M0", (-1.02, A / 2), 0.0, "&#963;&#8320; = { m&#8339;, g }"),
        ("M1", (-A / 2, A / 2), 0.0, "&#963;&#8321; = { m&#8339;, g, s }"),
        ("M2", pivot_pose(math.radians(34)), math.radians(34),
         "&#963;&#8323; = { m&#8339;, s, g&#7525; }"),
        ("M3", pivot_pose(math.pi / 2), math.pi / 2,
         "&#963;&#8325; = { m&#8339;, s, t }"),
        ("M4", (1.0, H + A / 2), math.pi / 2, "&#963;&#8328; = { m&#7764;, t }"),
    ]
    for i, (name, p, th, cap) in enumerate(poses):
        cx = cell * i + cell / 2
        scene_at(g, cx=cx + 6, floor_y=214, scale=57, x_lo=-1.95, x_hi=1.9)
        draw_box(g, p, th)
        if name == "M0":
            d = disk_on_face(p, th, "-x", 0.5)
        elif name in ("M1", "M2"):
            d = disk_on_face(p, th, "-x", 0.74)
        elif name == "M3":
            d = disk_on_face(p, th, "-x", 0.5)
        else:
            d = disk_on_face(p, th, "-z", 0.5)
        draw_disk(g, d)

        if name == "M0":
            g.line(*box_feature(p, th, "-z"), stroke=C["hot"], sw=3.0)
            g.dot(disk_on_face(p, th, "-x", 0.5, r=0.0), r_px=3.6)
        elif name == "M1":
            g.line(*box_feature(p, th, "-z"), stroke=C["hot"], sw=3.0)
            g.line((0, 0), (0, H), stroke=C["hot"], sw=3.0)
            g.dot(disk_on_face(p, th, "-x", 0.74, r=0.0), r_px=3.6)
        elif name == "M2":
            g.dot(box_feature(p, th, "v+-"), r_px=4.2)
            g.dot(PIVOT, r_px=4.2)
            g.dot(disk_on_face(p, th, "-x", 0.74, r=0.0), r_px=3.6)
            g.arc_arrow((0, H), 0.72, 152, 112, stroke=C["man"], sw=1.8)
        elif name == "M3":
            f = box_feature(p, th, "+x")
            g.line((0.0, H), (max(f[0][0], f[1][0]), H), stroke=C["hot"], sw=3.0)
            g.dot(PIVOT, r_px=4.2)
            g.dot(disk_on_face(p, th, "-x", 0.5, r=0.0), r_px=3.6)
        else:
            g.line(*box_feature(p, th, "+x"), stroke=C["hot"], sw=3.0)
            g.dot(disk_on_face(p, th, "-z", 0.5, r=0.0), r_px=3.6)

        g.px_text((cx, 36), name, size=16, weight="700", fill=C["ink"])
        g.px_text((cx, 262), cap, size=15, fill=C["primary"], weight="600",
                  family="'Latin Modern Math','STIX Two Math',Georgia,serif")
    g.px_text((W / 2, 290), "red = asserted contact &#160;·&#160; consecutive "
              "markings differ by one make or break", size=13.5,
              fill=C["muted"])
    return g.save(path, "Five snapshots of the running task")


def prior_work(path):
    """What velocity-mode planners can see, versus what the task needs."""
    g = Svg(940, 372, ox=0, oy=0, scale=1, cls="fig")
    for col, (title, sub) in enumerate([
            ("Velocity modes", "enumerated from contacts that already touch"),
            ("Contact catalog", "every named feature pair, touching or not")]):
        cx = 235 + col * 470
        g.px_text((cx, 38), title, size=19, weight="700", fill=C["ink"])
        g.px_text((cx, 60), sub, size=14, fill=C["muted"])

    p = (-A / 2, A / 2)
    for col in (0, 1):
        scene_at(g, cx=228 + col * 470, floor_y=232, scale=88,
                 x_lo=-1.7, x_hi=2.0)
        draw_box(g, p, 0.0)
        d = disk_on_face(p, 0.0, "-x", alpha=0.6)
        draw_disk(g, d)
        g.line(*box_feature(p, 0.0, "-z"), stroke=C["hot"], sw=3.2)
        g.dot((d[0] + R_DISK, d[1]), r_px=4.0)
        if col == 1:
            g.line((0, 0), (0, H), stroke=C["violet"], sw=3.4, dash="5 4")
            g.line((0, H), (2 * A, H), stroke=C["violet"], sw=3.4, dash="5 4")
            g.dot(box_feature(p, 0.0, "v+-"), r_px=4.4, fill=C["violet"])

    g.px_text((235, 296), "only what is touching is in the state", size=14.5,
              fill=C["hot"], weight="600")
    g.px_text((235, 318), "the riser and the platform are invisible", size=13,
              fill=C["muted"])
    g.px_text((705, 296), "&#8220;go establish contact s&#8221; is an ordinary "
              "action", size=14.5, fill=C["primary"], weight="600")
    g.px_text((705, 318), "unmade contacts are already in the state", size=13,
              fill=C["muted"])
    g.add(f'<line x1="470" y1="26" x2="470" y2="334" stroke="{C["rule"]}" '
          f'stroke-width="1.4"/>')
    g.px_text((470, 358), "the difference that matters is which of these a "
              "planner may decide", size=13.5, fill=C["faint"])
    return g.save(path, "Existing contacts versus the full catalog")
