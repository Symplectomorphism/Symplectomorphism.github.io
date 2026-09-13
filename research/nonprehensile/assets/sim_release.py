"""Integrate what the box actually does when the manipulator lets go.

A slide that claims "no equilibrium exists" should be able to show the motion
that follows, not assert it.  This is a small penalty-contact simulator for the
planar box against the step, used only to generate figure frames.

Conventions (identical to svgkit)
---------------------------------
    R(t) v = (v_x cos t + v_z sin t,  -v_x sin t + v_z cos t)

so positive theta carries +x_B toward -z_W: the counterclockwise topple as it
is drawn, with +x_W to the left.  Differentiating, a body point at world offset
r from the centre of mass moves with

    v_pt = v_cm + omega * ( r_z , -r_x )

and the generalized force on theta from a point force f applied there is

    T = r_z f_x - r_x f_z .

Contacts are compliant -- a spring-damper along the normal with regularized
Coulomb friction along the tangent.  That needs no case analysis for sticking,
sliding, or separation: they fall out of the normal force reaching zero.  Two
families cover this scene:

  (a) each corner of the box against the step solid
      {x <= 0, z <= 0} union {x >= 0, z <= h};
  (b) the step's convex corner (0, h) against the interior of the box, which
      is the contact the box leans on at mid-topple.

Nothing here is used for planning; it only decides where to draw the box.
"""

from __future__ import annotations

import math

G = 9.81


def rot(t, v):
    c, s = math.cos(t), math.sin(t)
    return (v[0] * c + v[1] * s, -v[0] * s + v[1] * c)


def corners(p, th, a=1.0):
    q = a / 2
    return [(p[0] + rot(th, v)[0], p[1] + rot(th, v)[1])
            for v in ((+q, +q), (-q, +q), (-q, -q), (+q, -q))]


def step_depth(x, z, h):
    """Depth of a point inside the step solid, with the outward normal."""
    inside = (x <= 0.0 and z <= 0.0) or (x >= 0.0 and z <= h)
    if not inside:
        return 0.0, (0.0, 1.0)
    cands = []
    if x <= 0.0:
        cands.append((-z, (0.0, 1.0)))
    if x >= 0.0:
        cands.append((h - z, (0.0, 1.0)))
        if 0.0 <= z <= h:
            cands.append((x, (-1.0, 0.0)))
    d, n = min(cands, key=lambda c: c[0])
    return max(d, 0.0), n


def box_depth(q, p, th, a=1.0):
    """Depth of a world point inside the box, with the box's outward normal."""
    loc = rot(-th, (q[0] - p[0], q[1] - p[1]))
    half = a / 2
    dx, dz = half - abs(loc[0]), half - abs(loc[1])
    if dx <= 0.0 or dz <= 0.0:
        return 0.0, (0.0, 1.0)
    if dx < dz:
        return dx, rot(th, (math.copysign(1.0, loc[0]), 0.0))
    return dz, rot(th, (0.0, math.copysign(1.0, loc[1])))


def simulate(p0, th0, a=1.0, h=0.375, m=1.0, mu=0.4, kn=2.5e4, zeta=0.35,
             dt=1.0e-5, t_end=1.0, v_eps=0.03, sample=900):
    """Release from rest at (p0, th0).  Returns [(t, (x, z), theta), ...]."""
    I = m * a * a / 6.0
    x, z, th = p0[0], p0[1], th0
    vx = vz = om = 0.0
    cn = zeta * 2.0 * math.sqrt(kn * m)
    out = []
    n_steps = int(t_end / dt)
    every = max(1, n_steps // sample)

    for step in range(n_steps):
        Fx, Fz, T = 0.0, -m * G, 0.0

        def add(rx, rz, f):
            nonlocal Fx, Fz, T
            Fx += f[0]
            Fz += f[1]
            T += rz * f[0] - rx * f[1]

        # (a) the box's corners against the step
        for c in corners((x, z), th, a):
            d, n = step_depth(c[0], c[1], h)
            if d <= 0.0:
                continue
            rx, rz = c[0] - x, c[1] - z
            vpt = (vx + om * rz, vz - om * rx)
            vn = vpt[0] * n[0] + vpt[1] * n[1]
            fn = kn * d - cn * vn
            if fn <= 0.0:
                continue
            t_hat = (-n[1], n[0])
            vt = vpt[0] * t_hat[0] + vpt[1] * t_hat[1]
            ft = -mu * fn * math.tanh(vt / v_eps)
            add(rx, rz, (n[0] * fn + t_hat[0] * ft, n[1] * fn + t_hat[1] * ft))

        # (b) the step's corner against the box.  The force ejects the corner,
        # so it acts on the box along -n_out.
        d, n = box_depth((0.0, h), (x, z), th, a)
        if d > 0.0:
            rx, rz = 0.0 - x, h - z
            vpt = (vx + om * rz, vz - om * rx)
            vn = vpt[0] * n[0] + vpt[1] * n[1]      # >0 drives it deeper
            fn = kn * d + cn * vn
            if fn > 0.0:
                t_hat = (-n[1], n[0])
                vt = vpt[0] * t_hat[0] + vpt[1] * t_hat[1]
                ft = -mu * fn * math.tanh(vt / v_eps)
                add(rx, rz, (-n[0] * fn + t_hat[0] * ft,
                             -n[1] * fn + t_hat[1] * ft))

        vx += dt * Fx / m
        vz += dt * Fz / m
        om += dt * T / I
        x += dt * vx
        z += dt * vz
        th += dt * om
        if step % every == 0:
            out.append((step * dt, (x, z), th))
    return out


def release_frames(p0, th0, n=48, a=1.0, h=0.375, **kw):
    """Evenly sampled poses, and the time the step-corner contact is lost."""
    traj = simulate(p0, th0, a=a, h=h, **kw)
    lost = None
    for t, p, th in traj:
        d, _ = box_depth((0.0, h), p, th, a)
        if t > 1e-3 and d <= 0.0:
            lost = t
            break
    idx = [round(i * (len(traj) - 1) / (n - 1)) for i in range(n)]
    return [traj[i] for i in idx], lost, traj
