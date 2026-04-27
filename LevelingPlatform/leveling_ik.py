"""
3-RRS Leveling Platform — standalone inverse kinematics module.

Given a target 3D point (x, y, z) in world coordinates, returns the three
motor angles required to aim the platform normal toward that point.

Hardware model
--------------
- 3 motors on a base circle of radius Rb at 120 deg spacing.
- Crank of length La on each motor (revolute at B, axis tangential).
- Coupler of length Lc connects crank tip A to plate joint P.
  * Joint at A: revolute (tangential axis) -- coupler stays in the motor's
    r_hat-z vertical plane.
  * Joint at P: spherical (RC ball joint) with max deflection BALL_MAX_DEG
    from the plate's body +z axis.
- Top plate attaches on a circle of radius Rp = Rb - La.
- Home pose (theta = 0 on every motor): crank horizontal pointing inward,
  coupler vertical. Plate center is at (0, 0, H0 = Lc).

Motor angle convention
----------------------
theta = 0 at home. Positive theta raises the crank tip.
    A_i = B_i - La * (cos(theta) r_hat + sin(theta) z_hat)

3-RRS center offset
-------------------
When the plate tilts, its center slides horizontally so that every P_i
stays in its motor's r_hat-z plane. plate_center_offset(R) returns this
shift analytically.

Dependencies
------------
Only numpy. No matplotlib / display needed. Runs headless on Raspberry Pi 5.

CLI usage
---------
    python3 leveling_ik.py --target 0.10 0.00 3.0
"""

from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Hardware parameters (edit to match your build)
# ─────────────────────────────────────────────────────────────────────
Rb = 0.10                       # base pivot radius [m]
La = 0.04                       # crank length [m]
Lc = 0.12                       # coupler length [m]
Rp = Rb - La                    # plate joint radius [m] (forced by home)
H0 = Lc                         # nominal plate center height [m]

PHI = np.deg2rad([0.0, 120.0, 240.0])   # motor azimuths

MOTOR_STEPS = 4096                          # encoder counts / revolution
MOTOR_STEP_RAD = 2.0 * np.pi / MOTOR_STEPS  # [rad / step]

BALL_MAX_DEG = 30.0             # P-side ball joint deflection limit


# ─────────────────────────────────────────────────────────────────────
# Rotation helper (shortest-arc, yaw-free)
# ─────────────────────────────────────────────────────────────────────
def rot_from_normal(n):
    """Rotation matrix mapping +z to unit vector n (shortest arc)."""
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, n))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    vx = np.array([[0.0, -v[2],  v[1]],
                   [v[2],  0.0, -v[0]],
                   [-v[1], v[0],  0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


# ─────────────────────────────────────────────────────────────────────
# 3-RRS plate center offset (closed form)
# ─────────────────────────────────────────────────────────────────────
def plate_center_offset(R):
    """
    For a yaw-free rotation R, return (cx, cy) such that every
    P_i = (cx, cy, *) + R @ p_body_i lies in its motor's r_hat-z plane.
    """
    a = np.zeros(3)
    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        t_hat = np.array([-np.sin(phi), np.cos(phi), 0.0])
        a[i] = Rp * float(np.dot(R @ r_hat, t_hat))
    cx =  (2.0 / 3.0) * float(np.sum(np.sin(PHI) * a))
    cy = -(2.0 / 3.0) * float(np.sum(np.cos(PHI) * a))
    return cx, cy


# ─────────────────────────────────────────────────────────────────────
# Inverse kinematics: platform normal + height -> motor angles
# ─────────────────────────────────────────────────────────────────────
def inverse_kinematics(normal, height=H0, ball_max_deg=BALL_MAX_DEG,
                       quantize=True):
    """
    Core IK. Returns (thetas_rad, ok, ball_angles_deg).
      thetas_rad       (3,) motor angles [rad]  (NaN where unreachable)
      ok               True iff all legs reachable AND ball limit satisfied
      ball_angles_deg  (3,) P-side deflection [deg] per leg (NaN if length
                       infeasible)
    """
    R = rot_from_normal(normal)
    cx, cy = plate_center_offset(R)
    c = np.array([cx, cy, height])
    z_hat = np.array([0.0, 0.0, 1.0])
    plate_up = R @ z_hat

    thetas = np.full(3, np.nan)
    ball = np.full(3, np.nan)
    ok = True

    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        B_i = Rb * r_hat
        p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
        P_i = c + R @ p_body

        d = P_i - B_i
        u = float(np.dot(d, r_hat))
        v = float(d[2])
        k = (float(np.dot(d, d)) + La * La - Lc * Lc) / (2.0 * La)
        rho = np.hypot(u, v)
        if rho < 1e-12 or abs(k) > rho:
            ok = False
            continue

        # '-' branch -> home = 0
        th = np.arctan2(v, u) - np.arccos(-k / rho)
        th = (th + np.pi) % (2.0 * np.pi) - np.pi
        if quantize:
            th = round(th / MOTOR_STEP_RAD) * MOTOR_STEP_RAD

        A_i = B_i - La * (np.cos(th) * r_hat + np.sin(th) * z_hat)
        coupler_dir = (P_i - A_i) / Lc
        cos_P = float(np.clip(np.dot(coupler_dir, plate_up), -1.0, 1.0))
        ang_P = float(np.rad2deg(np.arccos(cos_P)))

        thetas[i] = th
        ball[i] = ang_P
        if ang_P > ball_max_deg:
            ok = False

    return thetas, ok, ball


# ─────────────────────────────────────────────────────────────────────
# Public API: aim the platform at a 3D target point
# ─────────────────────────────────────────────────────────────────────
def aim_at(target, height=H0, ball_max_deg=BALL_MAX_DEG, quantize=True):
    """
    Compute motor angles to point the plate normal from the platform
    center (0, 0, height) toward the target.

    Returns a dict:
      angles_deg    list[float]|None   motor angles [deg] (None if unreachable)
      angles_rad    list[float]|None   motor angles [rad]
      angles_steps  list[int]|None     encoder counts (nearest step)
      ok            bool               True iff feasible AND ball within limit
      ball_deg      list[float|None]   P-side ball deflection per leg [deg]
      c_shift_m     tuple[float,float] 3-RRS plate center horizontal shift [m]
      normal        list[float]        commanded plate normal (unit)
    """
    T = np.asarray(target, dtype=float)
    v = T - np.array([0.0, 0.0, height])
    nv = np.linalg.norm(v)
    n = np.array([0.0, 0.0, 1.0]) if nv < 1e-9 else v / nv

    thetas, ok, ball = inverse_kinematics(n, height, ball_max_deg, quantize)
    cx, cy = plate_center_offset(rot_from_normal(n))

    reachable = not np.any(np.isnan(thetas))
    return {
        'angles_deg':   [float(np.rad2deg(t)) for t in thetas] if reachable else None,
        'angles_rad':   [float(t) for t in thetas] if reachable else None,
        'angles_steps': [int(round(t / MOTOR_STEP_RAD)) for t in thetas] if reachable else None,
        'ok': bool(ok),
        'ball_deg':  [float(b) if not np.isnan(b) else None for b in ball],
        'c_shift_m': (float(cx), float(cy)),
        'normal':    [float(n[0]), float(n[1]), float(n[2])],
    }


def aim_normal(normal, height=H0, ball_max_deg=BALL_MAX_DEG, quantize=True):
    """Same as aim_at but takes the plate normal directly (unit vector)."""
    thetas, ok, ball = inverse_kinematics(normal, height, ball_max_deg, quantize)
    cx, cy = plate_center_offset(rot_from_normal(normal))
    reachable = not np.any(np.isnan(thetas))
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    return {
        'angles_deg':   [float(np.rad2deg(t)) for t in thetas] if reachable else None,
        'angles_rad':   [float(t) for t in thetas] if reachable else None,
        'angles_steps': [int(round(t / MOTOR_STEP_RAD)) for t in thetas] if reachable else None,
        'ok': bool(ok),
        'ball_deg':  [float(b) if not np.isnan(b) else None for b in ball],
        'c_shift_m': (float(cx), float(cy)),
        'normal':    [float(n[0]), float(n[1]), float(n[2])],
    }


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description='3-RRS leveling platform inverse kinematics')
    ap.add_argument('--target', nargs=3, type=float, required=True,
                    metavar=('X', 'Y', 'Z'),
                    help='target 3D point in world frame [m]')
    ap.add_argument('--height', type=float, default=H0,
                    help=f'plate center height [m] (default {H0})')
    ap.add_argument('--ball_max', type=float, default=BALL_MAX_DEG,
                    help=f'P-side ball joint limit [deg] (default {BALL_MAX_DEG})')
    ap.add_argument('--no-quantize', action='store_true',
                    help='skip encoder step quantization')
    args = ap.parse_args()

    r = aim_at(tuple(args.target), height=args.height,
               ball_max_deg=args.ball_max, quantize=not args.no_quantize)

    if r['angles_deg'] is None:
        print('UNREACHABLE (length constraint violated on at least one leg)')
        sys.exit(2)

    a = r['angles_deg']
    s = r['angles_steps']
    b = r['ball_deg']
    cx, cy = r['c_shift_m']
    print(f"target        : ({args.target[0]:+.4f}, {args.target[1]:+.4f}, {args.target[2]:+.4f}) m")
    print(f"normal        : ({r['normal'][0]:+.5f}, {r['normal'][1]:+.5f}, {r['normal'][2]:+.5f})")
    print(f"motor angles  : {a[0]:+8.4f}   {a[1]:+8.4f}   {a[2]:+8.4f}   [deg]")
    print(f"encoder steps : {s[0]:+8d}   {s[1]:+8d}   {s[2]:+8d}   (0..{MOTOR_STEPS-1})")
    print(f"ball P defl.  : {b[0]:8.4f}   {b[1]:8.4f}   {b[2]:8.4f}   [deg] (lim={args.ball_max})")
    print(f"center shift  : ({cx*1000:+.3f}, {cy*1000:+.3f}) mm")
    print(f"feasible      : {r['ok']}")
    sys.exit(0 if r['ok'] else 1)
