"""
3-DOF Leveling Platform Simulator (3-RSS parallel mechanism)

Geometry
--------
- 3 motors placed on a base circle (radius Rb) at 120 deg spacing.
- Each motor rotates a crank arm of length La in the vertical plane
  that contains the base radius direction (axis of rotation is tangential).
- A coupler rod of length Lc connects the crank tip to the top plate
  via ball joints at both ends (so the coupler is an S-S link and only
  its length constraint matters).
- The top plate has 3 attachment points on a circle of radius Rp, at
  the same 120 deg angles as the base (in the plate body frame).

Given a desired platform orientation (unit normal vector n) and a
commanded center height h, this script solves the inverse kinematics
for the three motor angles and renders the mechanism in 3D.

Closed-form IK per leg
----------------------
    d_i = P_i - B_i
    u   = d_i . r_hat_i        (radial component)
    v   = d_i_z                (vertical component)
    k   = (|d_i|^2 + La^2 - Lc^2) / (2 La)
    u cos th + v sin th = k
 => th_i = atan2(v, u) - acos( k / sqrt(u^2 + v^2) )
    (the '-' branch picks the "elbow-down / knee-out" assembly)

Run
---
    python leveling_sim.py
Drag the sliders to tilt the platform. The title shows the solved
motor angles in degrees. If a pose is unreachable (|k| > sqrt(u^2+v^2))
the title turns red.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -------------------- Geometry (edit to match your hardware) --------------------
Rb = 0.10   # base pivot radius [m]
La = 0.04   # crank (motor arm) length [m]
Lc = 0.12   # coupler rod length [m]
# Home pose assumption: crank points radially inward (horizontal) and
# coupler points straight up. This forces:
Rp = Rb - La   # top plate joint radius [m]  -> 0.06
H0 = Lc        # nominal platform center height [m] -> 0.12

PHI = np.deg2rad([0.0, 120.0, 240.0])   # motor angular positions

# Motor encoder resolution: 4096 steps per full revolution (2*pi)
MOTOR_STEPS = 4096
MOTOR_STEP_RAD = 2.0 * np.pi / MOTOR_STEPS


# -------------------- Math helpers --------------------
def rot_from_normal(n):
    """Rotation matrix mapping +z to the unit vector n (shortest arc)."""
    n = np.asarray(n, dtype=float)
    n /= np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, n))
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:
        # 180 deg flip around x
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(z, n)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]],
                   [v[2], 0, -v[0]],
                   [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def inverse_kinematics(normal, height):
    """
    Returns:
        thetas : (3,) motor angles [rad] (NaN where unreachable)
        A      : (3,3) crank tip positions
        P      : (3,3) top joint positions
        B      : (3,3) base pivot positions
        ok     : bool - all legs reachable
    """
    R = rot_from_normal(normal)
    c = np.array([0.0, 0.0, height])

    B = np.zeros((3, 3))
    P = np.zeros((3, 3))
    A = np.zeros((3, 3))
    thetas = np.full(3, np.nan)
    ok = True

    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        B[i] = Rb * r_hat
        # top joint in plate body frame, then rotated & translated
        p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
        P[i] = c + R @ p_body

        d = P[i] - B[i]
        u = float(np.dot(d, r_hat))
        v = float(d[2])
        k = (float(np.dot(d, d)) + La * La - Lc * Lc) / (2.0 * La)
        rho = np.hypot(u, v)
        if rho < 1e-12 or abs(k) > rho:
            ok = False
            continue
        # '+ acos' branch -> home pose is theta = 180 deg
        # (crank folded inward toward center, coupler vertical)
        th = np.arctan2(v, u) + np.arccos(k / rho)
        # Quantize to nearest encoder step (4096 counts / revolution)
        th = np.round(th / MOTOR_STEP_RAD) * MOTOR_STEP_RAD
        thetas[i] = th
        A[i] = B[i] + La * (np.cos(th) * r_hat + np.sin(th) * np.array([0, 0, 1.0]))

    return thetas, A, P, B, ok


def _platform_joints(nx, ny, zc):
    """Given pose state (nx, ny, zc), return P_i (3x3) and the normal n."""
    s2 = nx * nx + ny * ny
    nz = np.sqrt(max(1.0 - s2, 0.0))
    n = np.array([nx, ny, nz])
    R = rot_from_normal(n)
    c = np.array([0.0, 0.0, zc])
    P = np.zeros((3, 3))
    for i, phi in enumerate(PHI):
        p_body = np.array([Rp * np.cos(phi), Rp * np.sin(phi), 0.0])
        P[i] = c + R @ p_body
    return P, n


def forward_kinematics(thetas_q, guess_n, guess_zc):
    """
    Given the three (quantized) motor angles, solve for the platform pose
    via Newton's method. Returns (n_actual, zc_actual, ok).
    """
    # crank tips from quantized angles
    A = np.zeros((3, 3))
    for i, phi in enumerate(PHI):
        r_hat = np.array([np.cos(phi), np.sin(phi), 0.0])
        A[i] = Rb * r_hat + La * (np.cos(thetas_q[i]) * r_hat
                                  + np.sin(thetas_q[i]) * np.array([0, 0, 1.0]))

    def residual(state):
        P, _ = _platform_joints(state[0], state[1], state[2])
        return np.array([np.sum((P[i] - A[i])**2) - Lc * Lc for i in range(3)])

    state = np.array([guess_n[0], guess_n[1], guess_zc], dtype=float)
    eps = 1e-7
    for _ in range(25):
        r = residual(state)
        if np.linalg.norm(r) < 1e-12:
            break
        J = np.zeros((3, 3))
        for k in range(3):
            s2 = state.copy(); s2[k] += eps
            J[:, k] = (residual(s2) - r) / eps
        try:
            dx = np.linalg.solve(J, -r)
        except np.linalg.LinAlgError:
            return guess_n, guess_zc, False
        state += dx
        if np.linalg.norm(dx) < 1e-12:
            break
    _, n_act = _platform_joints(state[0], state[1], state[2])
    ok = np.linalg.norm(residual(state)) < 1e-8
    return n_act, float(state[2]), ok


# =====================================================================
#  LAYOUT — all axes positioned manually for a clean, non-overlapping UI
#
#  +-----------------------------------------+------------------+----+
#  |                                         |   XY picker      | Z  |
#  |           3D view                       |   (square)       |bar |
#  |                                         |                  |    |
#  +-----------------------------------------+------------------+----+
#  |           (3D continues)                | sliders + info   |    |
#  +-----------------------------------------+------------------+----+
# =====================================================================
fig = plt.figure(figsize=(14, 9))

# ---- 3D view (left) ----
ax = fig.add_axes([0.02, 0.06, 0.52, 0.90], projection='3d')

# ---- XY picker (right-top, square) ----
# To keep it square: fig is 14 x 9 in, so frac_w 0.28 => 3.92 in
# frac_h = 3.92 / 9 = 0.436
_xy_l, _xy_b, _xy_w, _xy_h = 0.58, 0.50, 0.28, 0.44
ax_xy = fig.add_axes([_xy_l, _xy_b, _xy_w, _xy_h])

# ---- Z bar (right of XY, same vertical span) ----
ax_z = fig.add_axes([_xy_l + _xy_w + 0.03, _xy_b, 0.02, _xy_h])

# ---- Sliders (below pickers) ----
_sl_l, _sl_w = 0.64, 0.26
ax_h  = fig.add_axes([_sl_l, 0.38, _sl_w, 0.022])
ax_Rb = fig.add_axes([_sl_l, 0.32, _sl_w, 0.022])
ax_La = fig.add_axes([_sl_l, 0.26, _sl_w, 0.022])
ax_Lc = fig.add_axes([_sl_l, 0.20, _sl_w, 0.022])
s_h  = Slider(ax_h,  'height', 0.04, 0.30, valinit=H0, valfmt='%.3f m')
s_Rb = Slider(ax_Rb, 'Rb',     0.04, 0.25, valinit=Rb, valfmt='%.3f m')
s_La = Slider(ax_La, 'La',     0.01, 0.12, valinit=La, valfmt='%.3f m')
s_Lc = Slider(ax_Lc, 'Lc',     0.04, 0.30, valinit=Lc, valfmt='%.3f m')

err_text = ax_xy.text(0.02, 0.98, '', transform=ax_xy.transAxes,
                      va='top', ha='left', fontsize=8, family='monospace',
                      bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85),
                      zorder=10)

# ---- 3D artists ----
base_poly,  = ax.plot([], [], [], 'k-', lw=2)
plate_poly, = ax.plot([], [], [], 'b-', lw=2)
crank_lines   = [ax.plot([], [], [], 'r-', lw=3)[0] for _ in range(3)]
coupler_lines = [ax.plot([], [], [], 'g-', lw=2)[0] for _ in range(3)]
pivot_pts,   = ax.plot([], [], [], 'ko', ms=5)
joint_pts,   = ax.plot([], [], [], 'bo', ms=5)
tip_pts,     = ax.plot([], [], [], 'rs', ms=5)
normal_line, = ax.plot([], [], [], 'm-', lw=2)
aim_line,    = ax.plot([], [], [], 'm--', lw=1.2)
actual_line, = ax.plot([], [], [], 'c-', lw=1.2)

lim = max(Rb, Rp) * 3.0
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
ax.set_box_aspect((1, 1, 1))
ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
title = ax.set_title('', pad=10)

# ---- XY picker setup ----
XY_LIM = 0.4
Z_LIM  = (2.5, 3.5)
target_state = {'x': 0.10, 'y': 0.0, 'z': 3.0}

ax_xy.set_xlim(-XY_LIM, XY_LIM); ax_xy.set_ylim(-XY_LIM, XY_LIM)
ax_xy.set_aspect('equal')
ax_xy.set_title('Target X, Y  (click / drag)', fontsize=10, pad=6)
ax_xy.set_xlabel('X [m]', fontsize=9); ax_xy.set_ylabel('Y [m]', fontsize=9)
ax_xy.tick_params(labelsize=8)
ax_xy.grid(True, alpha=0.3)

WS_N = 60
_gx = np.linspace(-XY_LIM, XY_LIM, WS_N)
_gy = np.linspace(-XY_LIM, XY_LIM, WS_N)
ws_img = ax_xy.imshow(np.zeros((WS_N, WS_N)),
                      extent=[-XY_LIM, XY_LIM, -XY_LIM, XY_LIM],
                      origin='lower', cmap='Greens', vmin=0, vmax=1,
                      alpha=0.35, zorder=0)
_th = np.linspace(0, 2 * np.pi, 64)
base_circle, = ax_xy.plot(Rb * np.cos(_th), Rb * np.sin(_th), 'k-', lw=1)
ax_xy.axhline(0, color='gray', lw=0.5); ax_xy.axvline(0, color='gray', lw=0.5)
xy_marker, = ax_xy.plot([target_state['x']], [target_state['y']],
                        'm*', ms=14, zorder=5)
xy_hit, = ax_xy.plot([], [], 'co', ms=7, zorder=5)

# ---- Z bar setup ----
ax_z.set_xlim(0, 1); ax_z.set_ylim(*Z_LIM)
ax_z.set_xticks([])
ax_z.set_title('Z [m]', fontsize=9, pad=6)
ax_z.tick_params(labelsize=8)
ax_z.yaxis.tick_right(); ax_z.yaxis.set_label_position('right')
z_marker, = ax_z.plot([0.5], [target_state['z']], 'm*', ms=14)


def close_loop(pts):
    return np.vstack([pts, pts[0]])


def update(_=None):
    h = s_h.val
    T = np.array([target_state['x'], target_state['y'], target_state['z']])
    c = np.array([0.0, 0.0, h])
    # aim platform normal from center toward target point
    v = T - c
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        n = np.array([0.0, 0.0, 1.0])
    else:
        n = v / nv

    thetas, A, P, B, ok = inverse_kinematics(n, h)

    bp = close_loop(B)
    base_poly.set_data(bp[:, 0], bp[:, 1]); base_poly.set_3d_properties(bp[:, 2])
    pp = close_loop(P)
    plate_poly.set_data(pp[:, 0], pp[:, 1]); plate_poly.set_3d_properties(pp[:, 2])

    for i in range(3):
        if np.isnan(thetas[i]):
            crank_lines[i].set_data([], []);  crank_lines[i].set_3d_properties([])
            coupler_lines[i].set_data([], []); coupler_lines[i].set_3d_properties([])
            continue
        xs = [B[i, 0], A[i, 0]]; ys = [B[i, 1], A[i, 1]]; zs = [B[i, 2], A[i, 2]]
        crank_lines[i].set_data(xs, ys); crank_lines[i].set_3d_properties(zs)
        xs = [A[i, 0], P[i, 0]]; ys = [A[i, 1], P[i, 1]]; zs = [A[i, 2], P[i, 2]]
        coupler_lines[i].set_data(xs, ys); coupler_lines[i].set_3d_properties(zs)

    pivot_pts.set_data(B[:, 0], B[:, 1]); pivot_pts.set_3d_properties(B[:, 2])
    joint_pts.set_data(P[:, 0], P[:, 1]); joint_pts.set_3d_properties(P[:, 2])
    valid = ~np.isnan(thetas)
    tip_pts.set_data(A[valid, 0], A[valid, 1]); tip_pts.set_3d_properties(A[valid, 2])

    nend = c + 0.06 * n
    normal_line.set_data([c[0], nend[0]], [c[1], nend[1]])
    normal_line.set_3d_properties([c[2], nend[2]])

    # commanded aim line from platform center to target (target is off-screen)
    aim_line.set_data([c[0], T[0]], [c[1], T[1]])
    aim_line.set_3d_properties([c[2], T[2]])

    # --- Forward kinematics with quantized motor angles -> actual aim ---
    hit = np.array([np.nan, np.nan, np.nan])
    err_xy = np.nan; err_ang = np.nan
    if ok and not np.any(np.isnan(thetas)):
        n_act, zc_act, fk_ok = forward_kinematics(thetas, n, h)
        if fk_ok and n_act[2] > 1e-6:
            c_act = np.array([0.0, 0.0, zc_act])
            # ray c_act + t*n_act  intersected with plane z = T[2]
            t = (T[2] - c_act[2]) / n_act[2]
            hit = c_act + t * n_act
            err_xy = float(np.linalg.norm(hit[:2] - T[:2]))
            cosang = float(np.clip(np.dot(n, n_act), -1, 1))
            err_ang = np.rad2deg(np.arccos(cosang))
            actual_line.set_data([c_act[0], hit[0]], [c_act[1], hit[1]])
            actual_line.set_3d_properties([c_act[2], hit[2]])
        else:
            actual_line.set_data([], []); actual_line.set_3d_properties([])
    else:
        actual_line.set_data([], []); actual_line.set_3d_properties([])

    if np.isnan(hit[0]):
        xy_hit.set_data([], [])
        err_text.set_text('  aim error: N/A')
    else:
        xy_hit.set_data([hit[0]], [hit[1]])
        err_text.set_text(
            f'  target  : ({T[0]:+.4f}, {T[1]:+.4f}, {T[2]:.2f})\n'
            f'  actual  : ({hit[0]:+.4f}, {hit[1]:+.4f}, {hit[2]:.2f})\n'
            f'  err XY  : {err_xy*1000:7.3f} mm\n'
            f'  err ang : {err_ang:7.4f} deg\n'
            f'  Rp(auto): {Rp:.4f} m'
        )

    deg = np.rad2deg(thetas)
    tilt_deg = np.rad2deg(np.arccos(np.clip(n[2], -1, 1)))
    txt = (f'theta1={deg[0]:+6.2f}  theta2={deg[1]:+6.2f}  theta3={deg[2]:+6.2f} [deg]'
           f'   (tilt={tilt_deg:.1f} deg)')
    title.set_text(txt)
    title.set_color('black' if ok else 'red')
    fig.canvas.draw_idle()


def recompute_workspace():
    """For current height + link lengths + target Z, compute which
    target (x,y) points are reachable, and update the shading."""
    h = s_h.val
    z = target_state['z']
    c = np.array([0.0, 0.0, h])
    mask = np.zeros((WS_N, WS_N), dtype=float)
    for iy, y in enumerate(_gy):
        for ix, x in enumerate(_gx):
            v = np.array([x, y, z]) - c
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                mask[iy, ix] = 1.0
                continue
            n = v / nv
            _, _, _, _, ok = inverse_kinematics(n, h)
            mask[iy, ix] = 1.0 if ok else 0.0
    ws_img.set_data(mask)


def on_params(_=None):
    """Link length / height slider handler: mutates globals, refreshes view."""
    global Rb, La, Lc, Rp, H0
    Rb = s_Rb.val
    La = s_La.val
    Lc = s_Lc.val
    Rp = max(Rb - La, 1e-4)
    H0 = Lc
    base_circle.set_data(Rb*np.cos(_th), Rb*np.sin(_th))
    recompute_workspace()
    update()


def _on_click(event):
    if event.inaxes is ax_xy and event.xdata is not None:
        target_state['x'] = float(np.clip(event.xdata, -XY_LIM, XY_LIM))
        target_state['y'] = float(np.clip(event.ydata, -XY_LIM, XY_LIM))
        xy_marker.set_data([target_state['x']], [target_state['y']])
        update()
    elif event.inaxes is ax_z and event.ydata is not None:
        target_state['z'] = float(np.clip(event.ydata, *Z_LIM))
        z_marker.set_data([0.5], [target_state['z']])
        recompute_workspace()
        update()

fig.canvas.mpl_connect('button_press_event', _on_click)
# drag support
def _on_motion(event):
    if event.button == 1:
        _on_click(event)
fig.canvas.mpl_connect('motion_notify_event', _on_motion)

s_h.on_changed(on_params)
s_Rb.on_changed(on_params)
s_La.on_changed(on_params)
s_Lc.on_changed(on_params)

recompute_workspace()
update()


# -------------------- Programmatic entry point --------------------
def solve(normal, height=H0):
    """Given a direction vector, return motor angles [deg]."""
    thetas, *_ , ok = inverse_kinematics(normal, height)
    return np.rad2deg(thetas), ok


def aim_at(target, height=H0):
    """Point the platform at a 3D world point. Returns motor angles [deg]."""
    T = np.asarray(target, dtype=float)
    c = np.array([0.0, 0.0, height])
    v = T - c
    n = v / np.linalg.norm(v)
    return solve(n, height)


if __name__ == '__main__':
    # quick sanity check
    ang, ok = solve([0, 0, 1], H0)
    print('neutral pose motor angles [deg]:', ang, 'reachable:', ok)
    plt.show()
