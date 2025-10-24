import time, sys, numpy as np, math

# ===================== ค่าพื้นฐาน =====================
pi = np.pi
d2r = pi/180     # องศา -> เรเดียน
r2d = 1/d2r      # เรเดียน -> องศา

# ===================== DH (ยืนยันแล้ว) =====================
# เปลี่ยนค่าในตาราง DH = เปลี่ยนคิเนเมติกส์ทั้งหมดของแขน
DH_Base = np.array([[0.0, 0.0, 0.0892, -pi/2]], dtype=float)  # [alpha, a, d, theta]
DH = np.array([
    [ 0.0   , 0.0    , 0.0    , 0.0],
    [ pi/2  , 0.0    , 0.0    , 0.0],
    [ 0.0   , 0.4251 , 0.0    , 0.0],
    [ 0.0   , 0.39215, 0.11   , 0.0],
    [ -pi/2 , 0.0    , 0.09475, 0.0],
    [  pi/2 , 0.0    , 0.0    , 0.0],
], dtype=float)
DH_EE = np.array([[0.0, 0.0, 0.26658, pi]], dtype=float)

# ออฟเซ็ตมุมศูนย์ของข้อต่อ (เปลี่ยน = เลื่อนตำแหน่งศูนย์แต่ละแกน)
th_offset = np.array([0.0, np.pi/2, 0.0, -np.pi/2, 0.0, 0.0], dtype=float)

# ===================== ขีดจำกัดข้อต่อ =====================
Q_MIN = np.array([-2*np.pi]*6, float)  # แคบลง = ปลอดภัยขึ้นแต่ IK หาได้น้อยลง
Q_MAX = np.array([ 2*np.pi]*6, float)
Q_NOM = np.zeros(6, float)             # จุดอ้างอิงสำหรับ nullspace

# ===================== คอนเวนชันทิศเข้าใกล้ของ tool =====================
# +1 = ให้ +Z ของ tool ชี้เข้าหาวัตถุ, -1 = ให้ -Z ชี้เข้า (สลับถ้าทิศ top-down กลับหัว)
APPROACH_TOOL_Z = +1

# ===================== Helpers =====================
def R_from_top_mode(mode, yaw):
    # เลือกเมทริกซ์หมุนตามโหมด TOP-DOWN/TOP-UP แล้วบิดด้วย yaw (รอบ Z_tool)
    if mode == 'TOP-DOWN':
        return make_R_topdown(yaw=yaw)
    elif mode == 'TOP-UP':
        return make_R_topup(yaw=yaw)
    else:
        raise ValueError("R_from_top_mode: mode ต้องเป็น 'TOP-DOWN' หรือ 'TOP-UP'")

def solve_with_yaw_sweep(q0, p_target, base_yaw, mode, solver_fn,
                         yaw_deg_list=(-180, -135, -90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 135)):
    # สแกน yaw หลายค่าเพื่อหลบลิมิต/ซิงกูลาร์และลดคอสต์ ep+0.35*eo
    best = None
    for yd in yaw_deg_list:
        yaw = base_yaw + yd*d2r
        R_try = R_from_top_mode(mode, yaw)
        q, it, ep, eo, ok = solver_fn(q0, p_target, R_try)
        cost = ep + 0.35*eo
        cand = dict(q=q, it=it, ep=ep, eo=eo, ok=ok, yaw=yaw, R=R_try, cost=cost)
        if best is None or (ok and not best['ok']) or (ok == best['ok'] and cost < best['cost']):
            best = cand
        if ok and cost < 2e-3:
            break
    return best

def clamp_q(q):
    # จำกัดมุมให้อยู่ในลิมิตข้อ
    return np.minimum(np.maximum(q, Q_MIN), Q_MAX)

def ang_wrap(x):
    # wrap มุมให้อยู่ในช่วง (-pi, pi]
    return (x + np.pi) % (2*np.pi) - np.pi

def ang_wrap_vec(q):
    # wrap ทุกองค์ประกอบของเวกเตอร์มุม
    q2 = np.array(q, float)
    for i in range(q2.shape[0]):
        q2[i] = ang_wrap(q2[i])
    return q2

def dh_transform_matrix(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    # เมทริกซ์ทรานส์ฟอร์มตาม DH
    ca = math.cos(alpha); sa = math.sin(alpha)
    ct = math.cos(theta); st = math.sin(theta)
    A = np.array([[   ct,   -st,    0,     a],
                  [st*ca, ct*ca, -sa , -d*sa],
                  [st*sa, ct*sa,  ca ,  d*ca],
                  [    0,     0,   0 ,     1]], dtype=float)
    return A

def _chain_from_DH(theta_list: list) -> tuple[list, np.ndarray]:
    # ต่อเชนทรานส์ฟอร์มตั้งแต่ฐานถึงปลายมือ และเก็บลิสต์ไว้ใช้คำนวณ J
    th = np.asarray(theta_list, dtype=float) + th_offset
    T_list = [np.eye(4, dtype=float)]
    T = np.eye(4, dtype=float)
    A_base = dh_transform_matrix(DH_Base[0,0], DH_Base[0,1], DH_Base[0,2], DH_Base[0,3])
    T = T @ A_base
    T_list.append(T)
    for i in range(6):
        alpha, a, d, _ = DH[i]
        A_i = dh_transform_matrix(alpha, a, d, th[i])
        T = T @ A_i
        T_list.append(T)
    A_ee = dh_transform_matrix(DH_EE[0,0], DH_EE[0,1], DH_EE[0,2], DH_EE[0,3])
    T = T @ A_ee
    T_list.append(T)
    return T_list, T

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    # แปลง R -> Euler XYZ (ระวังลำดับแกน)
    EPS = 1e-9
    r02 = max(-1.0, min(1.0, float(R[0, 2])))
    beta = math.asin(r02)
    if abs(r02) < 1.0 - EPS:
        alpha = math.atan2(-R[1, 2], R[2, 2])
        gamma = math.atan2(-R[0, 1], R[0, 0])
    else:
        alpha = 0.0
        gamma = math.atan2(R[1, 0], R[1, 1])
    return np.array([alpha, beta, gamma], dtype=float)

def ur5_forward_kinematrix(theta: list) -> tuple:
    # FK คืน (ตำแหน่ง, ยูเลอร์, ทรานส์ฟอร์มเต็ม)
    T_list, T_0E = _chain_from_DH(theta)
    position = T_0E[:3, 3]
    R = T_0E[:3, :3]
    euler_angles = rotation_matrix_to_euler(R)
    return position, euler_angles, T_0E

# ===================== Jacobian =====================
def ur5_geometric_jacobian_ref(theta: list) -> np.ndarray:
    # J เรขาคณิตอ้างอิงจากเฟรมแต่ละข้อ (ใช้เปรียบเทียบ/debug)
    T_list, T_0E = _chain_from_DH(theta)
    p_e = T_0E[:3, 3].copy()
    J = np.zeros((6, 6), dtype=float)
    for i in range(6):
        T_use = T_list[1] if i == 0 else T_list[i+2]
        z = T_use[:3, 2]
        p = T_use[:3, 3]
        if i == 0:
            J[:3, i] = np.cross(z, (p_e - p))
        else:
            J[:3, i] = np.cross((p_e - p), z)
        J[3:, i] = z
    return J

def match_reference_table(J: np.ndarray) -> np.ndarray:
    # ปรับสัญญาณบางแถวให้ตรงกับตารางอ้างอิงภายนอก
    J2 = J.copy()
    J2[1:3, 1:5] *= -1.0
    return J2

# ===================== SO(3) =====================
def log_SO3(R):
    # แปลง R -> เวกเตอร์แกน-มุม (log map)
    tr = np.clip((np.trace(R)-1.0)*0.5, -1.0, 1.0)
    theta = math.acos(tr)
    if abs(theta) < 1e-9:
        return np.zeros(3, float)
    w = (1.0/(2*math.sin(theta))) * np.array([R[2,1]-R[1,2],
                                              R[0,2]-R[2,0],
                                              R[1,0]-R[0,1]])
    return w*theta

# ===================== Euler / Rotation helpers =====================
def Rx(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1,0,0,0],[0,ca,-sa,0],[0,sa,ca,0],[0,0,0,1]], float)
def Ry(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,0,sa,0],[0,1,0,0],[-sa,0,ca,0],[0,0,0,1]], float)
def Rz(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ca,-sa,0,0],[sa,ca,0,0],[0,0,1,0],[0,0,0,1]], float)
def eulXYZ_to_R(eul_xyz):
    # Euler XYZ -> R (อย่าเปลี่ยนลำดับ)
    rx, ry, rz = eul_xyz
    return (Rx(rx) @ Ry(ry) @ Rz(rz))[:3,:3]

# ===================== Auto-approach =====================
Z_TOPDOWN = 0.60         # z ต่ำกว่า = โน้มไป TOP-DOWN
ELEV_DEG  = 35.0         # elevation แบนกว่า = TOP-DOWN
YAW_ALIGN_WITH_HEADING = True  # True = base_yaw = heading(x,y)

def rot_from_approach(approach_world, yaw=0.0):
    # ให้ Z_tool ชี้ตาม approach_world แล้วบิดรอบ Z_tool ด้วย yaw
    a = np.array(approach_world, float)
    a = a / (np.linalg.norm(a) + 1e-12)
    z_tool = APPROACH_TOOL_Z * a
    up_hint = np.array([0,0,1.0], float)
    if abs(np.dot(z_tool, up_hint)) > 0.99:
        up_hint = np.array([1,0,0], float)
    x_tool = np.cross(up_hint, z_tool); x_tool /= (np.linalg.norm(x_tool)+1e-12)
    y_tool = np.cross(z_tool, x_tool)
    R = np.column_stack([x_tool, y_tool, z_tool])
    c, s = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([[c,-s,0],[s,c,0],[0,0,1]], float)
    return R @ R_yaw

def make_R_topdown(yaw=0.0):
    # approach = -Z_world (จากด้านบน)
    return rot_from_approach([0,0,-1], yaw=yaw)

def make_R_topup(yaw=0.0):
    # approach = +Z_world (จากด้านล่าง)
    return rot_from_approach([0,0, 1], yaw=yaw)

def make_R_side_toward_xy(p_target, yaw=0.0):
    # approach ระนาบ XY ชี้จากฐานไปเป้า
    x, y = float(p_target[0]), float(p_target[1])
    ax = np.array([x, y, 0.0], float)
    n = np.linalg.norm(ax)
    if n < 1e-6: ax = np.array([1.0,0.0,0.0], float)
    else: ax /= n
    return rot_from_approach(ax, yaw=yaw)

# ย้ำค่า (ไม่มีผลเพิ่ม)
YAW_ALIGN_WITH_HEADING = True
ELEV_DEG  = 35.0
Z_TOPDOWN = 0.60

def pick_auto_approach(p_target, force_mode=None):
    # เลือกโหมด TOP-DOWN/SIDE อัตโนมัติจาก z และ elevation (TOP-UP ต้องบังคับ)
    x, y, z = float(p_target[0]), float(p_target[1]), float(p_target[2])
    rxy = math.hypot(x, y)
    elev = math.degrees(math.atan2(z, max(rxy,1e-12)))
    heading = math.atan2(y, x)
    yaw = heading if YAW_ALIGN_WITH_HEADING else 0.0

    mode = force_mode
    if mode is None:
        if (z < Z_TOPDOWN) or (abs(elev) < ELEV_DEG):
            mode = 'TOP-DOWN'
        else:
            mode = 'SIDE'

    if mode == 'TOP-DOWN':
        R = make_R_topdown(yaw=yaw)
    elif mode == 'TOP-UP':
        R = make_R_topup(yaw=yaw)
    elif mode == 'SIDE':
        R = make_R_side_toward_xy(p_target, yaw=0.0)
    else:
        R = make_R_side_toward_xy(p_target, yaw=0.0); mode = 'SIDE'
    return R, mode

def try_ik_with_fallback(q0, p_target, eul_deg_if_needed, solver_fn, force_mode=None):
    # ลองโหมดอัตโนมัติ; ถ้าเป็น TOP-* สแกน yaw; ถ้าไม่ได้ค่อย fallback ใช้ Euler ที่ป้อนมา
    eul_xyz = np.array(eul_deg_if_needed, float)*d2r
    R_euler = eulXYZ_to_R(eul_xyz)

    R_auto, mode_auto = pick_auto_approach(p_target, force_mode=force_mode)
    x, y = float(p_target[0]), float(p_target[1])
    heading = math.atan2(y, x)
    base_yaw = heading if YAW_ALIGN_WITH_HEADING else 0.0

    if mode_auto in ('TOP-DOWN', 'TOP-UP'):
        best = solve_with_yaw_sweep(q0, p_target, base_yaw, mode_auto, solver_fn)
        if best and (best['ok'] or best['cost'] < 1e-2):
            chosen = {"mode": mode_auto, "R": best['R'], "eul": rotation_matrix_to_euler(best['R']), "yaw": best['yaw']}
            return best['q'], best['it'], best['ep'], best['eo'], best['ok'], chosen
        q2, it2, ep2, eo2, ok2 = solver_fn(q0, p_target, R_euler)
        chosen = {"mode": "EULER-FALLBACK", "R": R_euler, "eul": rotation_matrix_to_euler(R_euler)}
        return q2, it2, ep2, eo2, ok2, chosen

    q1, it1, ep1, eo1, ok1 = solver_fn(q0, p_target, R_auto)
    if ok1:
        chosen = {"mode":mode_auto, "R":R_auto, "eul":rotation_matrix_to_euler(R_auto)}
        return q1, it1, ep1, eo1, ok1, chosen

    q2, it2, ep2, eo2, ok2 = solver_fn(q0, p_target, R_euler)
    chosen = {"mode":"EULER-FALLBACK", "R":R_euler, "eul":rotation_matrix_to_euler(R_euler)}
    return q2, it2, ep2, eo2, ok2, chosen

# ระยะก่อนเข้าหาเป้าตามแกน Z_tool (เพิ่ม = ถอยก่อนมากขึ้น)
PREAPPROACH_DIST = 0.10

def plan_pre_and_target(q_start, p_target, R_target, solver_fn):
    # คำนวณจุด PRE และ TARGET พร้อม q ที่ถึงจุดเหล่านั้น
    z_tool = R_target[:,2]
    p_pre  = p_target - PREAPPROACH_DIST * (APPROACH_TOOL_Z * z_tool)
    q_pre, *_ = solver_fn(q_start, p_pre, R_target)
    q_fin, it, ep, eo, ok = solver_fn(q_pre, p_target, R_target)
    return (p_pre, q_pre), (p_target, q_fin), ok

# ===================== จาคอบเบียนเชิงตัวเลข =====================
def jacobian_numeric(theta, h=1e-5):
    # central difference; h เล็กไปอาจมี noise, ใหญ่ไปจะหยาบ
    theta = np.array(theta, float)
    p0, _, T0 = ur5_forward_kinematrix(theta)
    R0 = T0[:3, :3]
    J = np.zeros((6, 6), float)
    for j in range(6):
        qp = theta.copy(); qp[j] += h
        qm = theta.copy(); qm[j] -= h
        pp, _, Tp = ur5_forward_kinematrix(qp)
        pm, _, Tm = ur5_forward_kinematrix(qm)
        Rp = Tp[:3,:3]; Rm = Tm[:3,:3]
        dp = (pp - pm) / (2*h)
        wp = log_SO3(Rp @ R0.T)
        wm = log_SO3(Rm @ R0.T)
        dw_body = (wp - wm) / (2*h)
        dw = R0 @ dw_body
        J[0:3, j] = dp
        J[3:6, j] = dw
    return J

# ===================== ตัวช่วย SVD / Nullspace =====================
def svd_damped_pinv(J, lam):
    # pseudoinverse แบบ DLS (lam สูงขึ้น = เสถียรขึ้นแต่ตอบช้าลง)
    U, s, Vt = np.linalg.svd(J, full_matrices=False)
    s_d = s / (s*s + lam*lam)
    return (Vt.T * s_d) @ U.T

def limit_avoidance_grad(q):
    # ดึงมุมเข้าใกล้ Q_NOM ใน nullspace
    w = 0.1
    return -w*(q - Q_NOM)

def clamp_step(dq, dq_max=10.0*d2r):
    # จำกัดก้าวต่อรอบ
    return np.clip(dq, -dq_max, dq_max)

# ===================== IK: Position-only + 6D DLS =====================
def ik_pos_only(theta0, p_target, max_iters=300, tol=2e-4, lam0=0.03, dq_max=10.0*d2r):
    # เริ่มต้นด้วยตำแหน่งล้วน (ช่วยทำให้เข้าใกล้ก่อน)
    th = ang_wrap_vec(theta0)
    for _ in range(max_iters):
        p_cur, _, _ = ur5_forward_kinematrix(th)
        ep = p_target - p_cur
        if np.linalg.norm(ep) < tol:
            break
        J = jacobian_numeric(th)
        Jp = J[:3,:]
        try:
            cond_p = np.linalg.cond(Jp)
        except:
            cond_p = 1e6
        lam = float(np.clip(lam0*(cond_p/50.0), 1e-3, 2e-1))
        dq = svd_damped_pinv(Jp, lam) @ ep
        dq += 0.03*limit_avoidance_grad(th)
        dq = np.clip(dq, -dq_max, dq_max)
        th = ang_wrap_vec(clamp_q(th + dq))
    return th

def dls_6d_weighted(J, e6, lam, w_pos=1.0, w_ori=1.0):
    # DLS 6D ถ่วงน้ำหนักตำแหน่ง/ทิศทาง
    W = np.diag([w_pos, w_pos, w_pos, w_ori, w_ori, w_ori])
    JW = W @ J
    eW = W @ e6
    JJt = JW @ JW.T
    dq = JW.T @ np.linalg.solve(JJt + (lam**2)*np.eye(6), eW)
    return dq

def ik_priority_dls(theta0, p_target, R_target,
                    max_iters=1000, tol_pos=1e-4, tol_ori=2e-3,
                    lam_pos0=0.02, lam_ori0=0.02, dq_max=12.0*d2r,
                    ko_base=0.06, w_null=0.08, warmup=True, do_polish=True):
    # เฟสตำแหน่ง -> โอเรียนเตชัน (nullspace) -> 6D รวม -> polish
    th = ang_wrap_vec(np.array(theta0, float))
    if warmup:
        th = ik_pos_only(th, p_target, max_iters=300, tol=2e-4, lam0=lam_pos0, dq_max=dq_max)
    alphas = [1.0, 0.7, 0.5, 0.3, 0.15, 0.08, 0.04]
    pos_phase_done = False
    for k in range(1, max_iters+1):
        p_cur, _, Tcur = ur5_forward_kinematrix(th)
        R_cur = Tcur[:3,:3]
        ep = p_target - p_cur
        ew_body = log_SO3(R_target @ R_cur.T)
        ew = R_cur @ ew_body
        epos = np.linalg.norm(ep)
        eori = np.linalg.norm(ew)
        if epos < tol_pos and eori < tol_ori:
            break
        J = jacobian_numeric(th)
        Jp = J[:3,:]
        Jo = J[3:,:]
        if (not pos_phase_done) or (epos > 3e-4):
            try: cond_p = np.linalg.cond(Jp)
            except: cond_p = 1e6
            lam_p = float(np.clip(lam_pos0 * (cond_p/50.0), 1e-3, 2e-1))
            Jp_pinv = svd_damped_pinv(Jp, lam_p)
            dq_pos = Jp_pinv @ ep
            Np = np.eye(6) - Jp_pinv @ Jp
            ko_now = ko_base + 0.6*(1.0 - math.exp(-epos/0.04))
            Jo_bar = Jo @ Np
            try: cond_o = np.linalg.cond(Jo_bar)
            except: cond_o = 1e6
            lam_o = float(np.clip(lam_ori0 * (cond_o/50.0), 1e-3, 2e-1))
            Jo_bar_pinv = svd_damped_pinv(Jo_bar, lam_o)
            dq_ori = Jo_bar_pinv @ (ko_now * ew)
            dq_ns = w_null * (Q_NOM - th)
            dq = dq_pos + Np @ dq_ori + Np @ dq_ns
            if epos < 3e-4:
                pos_phase_done = True
        else:
            try: cond6 = np.linalg.cond(J)
            except: cond6 = 1e6
            lam6 = float(np.clip(lam_ori0 * (cond6/50.0), 1e-3, 1.5e-1))
            w_pos = 8.0
            w_ori = 2.0 + 6.0*(1.0 - math.exp(-eori/0.08))
            e6 = np.hstack([ep, ew])
            dq = dls_6d_weighted(J, e6, lam6, w_pos=w_pos, w_ori=w_ori)
        dq = np.clip(dq, -dq_max, dq_max)
        p0 = p_cur; e0w = ew
        c0 = np.linalg.norm(p_target - p0) + 0.25*np.linalg.norm(e0w)
        improved = False
        th_best = th; c_best = c0
        for a in alphas:
            th_try = ang_wrap_vec(clamp_q(th + a*dq))
            p_t, _, T_t = ur5_forward_kinematrix(th_try)
            R_t = T_t[:3,:3]
            ep_t = p_target - p_t
            ew_t = R_t @ log_SO3(R_target @ R_t.T)
            c_t = np.linalg.norm(ep_t) + 0.25*np.linalg.norm(ew_t)
            if c_t < c_best:
                th_best = th_try; c_best = c_t; improved = True
                break
        th = th_best if improved else ang_wrap_vec(clamp_q(th + 0.05*dq))
    if do_polish:
        th = ik_pos_only(th, p_target, max_iters=120, tol=1.0e-4, lam0=0.02, dq_max=dq_max)
        for _ in range(60):
            p_cur, _, Tcur = ur5_forward_kinematrix(th)
            R_cur = Tcur[:3,:3]
            ep = p_target - p_cur
            if np.linalg.norm(ep) < 1.2e-4:
                break
            J = jacobian_numeric(th)
            Jp = J[:3,:]; Jo = J[3:,:]
            Jp_pinv = svd_damped_pinv(Jp, 0.02)
            Np = np.eye(6) - Jp_pinv @ Jp
            ew = R_cur @ log_SO3(R_target @ R_cur.T)
            dq = Jp_pinv @ ep + Np @ (0.02 * Jo.T @ ew)
            th = ang_wrap_vec(clamp_q(th + np.clip(dq, -8.0*d2r, 8.0*d2r)))
    p_cur, _, Tcur = ur5_forward_kinematrix(th)
    R_cur = Tcur[:3,:3]
    ep = p_target - p_cur
    ew = R_cur @ log_SO3(R_target @ R_cur.T)
    ok = (np.linalg.norm(ep) < tol_pos) and (np.linalg.norm(ew) < tol_ori)
    return th, k, float(np.linalg.norm(ep)), float(np.linalg.norm(ew)), ok

# ===================== ซีน/ข้อต่อ =====================
def get_or_create_target_cube(sim):
    # สร้างลูกบาศก์เป้า หากยังไม่มีในซีน
    try:
        return sim.getObject("/TargetCube")
    except:
        pass
    size = [0.05, 0.05, 0.05]
    h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 12)
    sim.setObjectAlias(h, "TargetCube", 0)
    sim.setObjectPosition(h, -1, [0.45, 0.0, 0.65])
    sim.setObjectOrientation(h, -1, [0.0, 0.0, 0.0])
    return h

def get_q(sim, hdl_j):
    # อ่านมุมข้อต่อทั้ง 6
    return np.array([sim.getJointPosition(hdl_j[i]) for i in range(6)], float)

def set_q(sim, hdl_j, q, use_target=True):
    # ส่งคำสั่งมุมไปยังคอนโทรลเลอร์หรือเซ็ตตรง
    qn = ang_wrap_vec(q)
    for i in range(6):
        if use_target:
            sim.setJointTargetPosition(hdl_j[i], float(qn[i]))
        else:
            sim.setJointPosition(hdl_j[i], float(qn[i]))

def seed_face_target_xy(p_t, q_now, shift_pi=False):
    # ตั้งมุมฐานหันเข้าหาเป้าในระนาบ XY (shift 180° ได้)
    q = q_now.copy()
    base = math.atan2(p_t[1], p_t[0])
    if shift_pi:
        base = ang_wrap(base + math.pi)
    q[0] = ang_wrap(base)
    return q

def print_log(sim, s):
    # พิมพ์ข้อความไปคอนโซลในซีน/STDOUT
    try: sim.auxiliaryConsolePrint(aux, s+"\n")
    except: sys.stdout.write(s+"\n")

# เวลาการเคลื่อนและเวลาค้างที่เป้า
MOVE_TIME = 5.0
DWELL_TIME = 2.0

def log_once(sim, aux, s):
    # log บรรทัดเดียวสั้น ๆ
    try:
        sim.auxiliaryConsolePrint(aux, s+"\n")
    except:
        sys.stdout.write(s+"\n")

def run_leg(sim, aux, hdl_j, hdl_end, q_from, q_to, target_pos, target_eul_rad, label):
    # เคลื่อนแขนจาก q_from -> q_to ภายใน MOVE_TIME พร้อม log ระหว่างทาง (ย้ายออกจาก sysCall_thread)
    t_start = sim.getSimulationTime()
    t_end   = t_start + MOVE_TIME
    dq_full = ang_wrap_vec(q_to - q_from)
    last_log = -1.0
    while True:
        t_now = sim.getSimulationTime()
        s = 0.0 if t_end == t_start else (t_now - t_start) / (t_end - t_start)
        if s < 0.0: s = 0.0
        if s > 1.0: s = 1.0
        q_cmd = ang_wrap_vec(q_from + s * dq_full)
        for i in range(6):
            sim.setJointTargetPosition(hdl_j[i], float(q_cmd[i]))
        if (last_log < 0) or (t_now - last_log > 0.2):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            msg = []
            msg.append(f"{label} s={s:5.2f} t={t_now:6.2f}s")
            msg.append("  target_pos:      " + ", ".join(f"{x:7.4f}" for x in target_pos))
            msg.append("  target_eul(deg): " + ", ".join(f"{x*r2d:7.2f}" for x in target_eul_rad))
            msg.append("  sim_pos:         " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul(deg):    " + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now
        if s >= 1.0:
            break
        sim.switchThread()
        
# ===================== Self-collision (coarse) =====================
# แทนแต่ละลิงก์เป็นทรงกลมที่เฟรมหลัก แล้วห้ามวงกลมที่ไม่ใช่เพื่อนบ้านชนกัน
SELF_SPHERE_IDX   = [3, 4, 5, 6, 7]                       # indices ใน T_list (หลัง joint2..joint6)
SELF_SPHERE_RADII = np.array([0.072, 0.072, 0.063, 0.054, 0.054], float) * 0.90  # เผื่อ safety
SELF_PAIR_SKIP_NEIGHBOR = True
BASE_RADIUS_APPROX = 0.0085      # รัศมีฐานโดยประมาณ (กันท่อนล่างไปชนฐาน)
CLEAR_MARGIN = 0.010            # margin เพิ่มเติม
PATH_SAMPLES = 5              # จำนวนจุดเช็ค self-collision ระหว่างทาง
GROUND_Z = 0.0          # ระดับพื้นโลก (CoppeliaSim ปกติคือ z=0)
GROUND_MARGIN = 0.010   # กันไว้ 10 มม. (ปรับได้)

# (ปรับช่วงมุมเพื่อลดท่าพับเกินจริง)
ELBOW_RANGE_DEG    = (-170.0, -5.0)   # q2
FOREARM_RANGE_DEG  = (-160.0, -5.0)   # q3

def _deg_ok(q, lohi_deg):
    lo, hi = np.deg2rad(lohi_deg[0]), np.deg2rad(lohi_deg[1])
    return (q >= lo) and (q <= hi)

def is_pose_self_clear(theta):
    # """ห้ามทรงกลม non-adjacent ชนกัน + ไม่ให้วงกลมไปเฉี่ยวฐาน + บังคับช่วงมุม q2,q3"""
    th = ang_wrap_vec(theta)
    q2, q3 = th[1], th[2]
    if (not _deg_ok(q2, ELBOW_RANGE_DEG)) or (not _deg_ok(q3, FOREARM_RANGE_DEG)):
        return False

    T_list, _ = _chain_from_DH(th)
    P = [T_list[i][:3, 3].copy() for i in SELF_SPHERE_IDX]
    R = SELF_SPHERE_RADII

    # กันฐาน
    for p, r in zip(P, R):
        rho = math.hypot(float(p[0]), float(p[1]))
        if rho < (BASE_RADIUS_APPROX + r - CLEAR_MARGIN):
            return False

    # ทรงกลมชนกันเอง (non-adjacent)
    for i in range(len(P)):
        for j in range(i+1, len(P)):
            if SELF_PAIR_SKIP_NEIGHBOR and (j == i+1):
                continue
            dij = float(np.linalg.norm(P[i] - P[j]))
            if dij < (R[i] + R[j] - CLEAR_MARGIN):
                return False
    return True

def is_path_self_clear(q0, qf, samples=PATH_SAMPLES):
    q0 = ang_wrap_vec(np.asarray(q0, float))
    qf = ang_wrap_vec(np.asarray(qf, float))
    for s in np.linspace(0.0, 1.0, samples):
        qt = ang_wrap_vec((1.0 - s) * q0 + s * qf)
        if not is_pose_self_clear(qt):
            return False
    return True

# ===================== สุ่มมุมเริ่ม–สุดท้ายในช่วงที่เหมาะสม =====================
# ช่วง "เหมาะสม" (แกนฐานปล่อยกว้าง, ไหล่/ศอกเน้นค่าลบให้แขนเหยียดออก, ข้อมือปล่อยหลวม)
JOINT_RANGE_DEG = [
    (-160, 160),   # q1 base
    (-110, -45),   # q2 shoulder (ไม่ให้ใกล้สุดพิสัย)
    (-130, -35),   # q3 elbow
    (-80,   80),   # q4 wrist1
    (-120, 120),   # q5 wrist2
    (-170, 170),   # q6 wrist3
]

def rand_joint_biased():
    q = np.zeros(6, float)
    for i, (lo, hi) in enumerate(JOINT_RANGE_DEG):
        q[i] = np.deg2rad(np.random.uniform(lo, hi))
    # เติม noise นิดหน่อย
    q += np.deg2rad(np.random.normal(0.0, 4.0, size=6))
    return ang_wrap_vec(clamp_q(q))

# ===================== Cubic polynomial trajectory =====================
def cubic_coeffs(q0, qf, T):
    # คิวบิก v(0)=v(T)=0 ต่อข้อ; คืนค่าสัมประสิทธิ์ a0..a3 รูปทรง (6,4)
    q0 = np.asarray(q0, float); qf = np.asarray(qf, float)
    dq = qf - q0
    a0 = q0
    a1 = np.zeros_like(q0)
    a2 = 3.0 * dq / (T*T)
    a3 = -2.0 * dq / (T*T*T)
    return np.column_stack([a0, a1, a2, a3])

def cubic_eval(coeffs, t):
    a0, a1, a2, a3 = coeffs[:,0], coeffs[:,1], coeffs[:,2], coeffs[:,3]
    q   = a0 + a1*t + a2*(t*t) + a3*(t*t*t)
    dq  = a1 + 2*a2*t + 3*a3*(t*t)
    ddq = 2*a2 + 6*a3*t
    return q, dq, ddq

def choose_T_with_accel(q0, qf, amax_each, T_hint):
    # เลือก T ให้ไม่เกิน amax ของแต่ละแกน: T >= max_i sqrt(6*|dq_i|/amax_i) และ T >= T_hint
    q0 = np.asarray(q0, float); qf = np.asarray(qf, float)
    amax_each = np.asarray(amax_each, float)
    dq = np.abs(ang_wrap_vec(qf - q0))
    T_req_per_joint = np.where(amax_each > 0, np.sqrt(np.where(dq > 0, 6.0*dq/amax_each, 0.0)), 0.0)
    T_req = float(np.max(T_req_per_joint))
    return max(T_req, float(T_hint))

SAFETY_MODE = 'warn'  # 'strict' = หยุดทันทีเมื่อเจอชน, 'warn' = แจ้งเตือนแล้วไปต่อ

def drive_q(sim, hdl_j, q):
    qn = ang_wrap_vec(q)
    for i in range(6):
        try:    sim.setJointTargetPosition(hdl_j[i], float(qn[i]))
        except: pass
        try:    sim.setJointPosition(hdl_j[i], float(qn[i]))
        except: pass

def run_leg_cubic(sim, aux, hdl_j, hdl_end, q_from, q_to, T_move, label):
    coeffs = cubic_coeffs(q_from, q_to, T_move)
    t0 = sim.getSimulationTime()
    last_log = -1.0
    log_once(sim, aux, f"{label} start cubic (T={T_move:5.2f}s)")
    warned = False
    while True:
        t_now = sim.getSimulationTime()
        t = t_now - t0
        if t < 0.0: t = 0.0
        if t > T_move: t = T_move
        q_cmd, dq_cmd, ddq_cmd = cubic_eval(coeffs, t)

        ok_self = is_pose_self_clear(q_cmd)
        if SAFETY_MODE != 'off':
            ok_self = is_pose_self_clear(q_cmd)
            if not ok_self:
                if SAFETY_MODE == 'strict':
                    log_once(sim, aux, f"{label} ABORT: self-collision risk at t={t_now:6.2f}s")
                    break
                elif not warned:
                    log_once(sim, aux, f"{label} WARN: self-collision check failed at t={t_now:6.2f}s (mode=warn)")
                    warned = True

        drive_q(sim, hdl_j, q_cmd)

        if (last_log < 0) or (t_now - last_log > 0.25):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            s = t/T_move if T_move>0 else 1.0
            msg = []
            msg.append(f"{label} s={s:5.2f} t={t_now:6.2f}s")
            msg.append("  q_cmd(deg):     " + ", ".join(f"{x*r2d:7.2f}" for x in q_cmd))
            msg.append("  dq_cmd(deg/s):  " + ", ".join(f"{x*r2d:7.2f}" for x in dq_cmd))
            msg.append("  ddq_cmd(deg/s2):" + ", ".join(f"{x*r2d:7.2f}" for x in ddq_cmd))
            msg.append("  sim_pos:        " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul(deg):   " + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now

        if t >= T_move: break
        sim.switchThread()

def scale_until_self_safe(q_from, q_to, max_halves=8, min_move_deg=10.0):
    # ย่อสโตก q_from→q_to ทีละครึ่งจนทางปลอดภัย; บังคับให้ขยับขั้นต่ำต่อแกน >= min_move_deg
    q_from = ang_wrap_vec(np.asarray(q_from, float))
    q_goal = ang_wrap_vec(np.asarray(q_to,   float))
    q_try  = q_goal.copy()
    ok_scaled = False
    for _ in range(max_halves):
        if is_path_self_clear(q_from, q_try):
            break
        q_try = ang_wrap_vec(q_from + 0.5*(q_try - q_from))
        ok_scaled = True
    dq = ang_wrap_vec(q_try - q_from)
    for i in range(6):
        if abs(dq[i])*r2d < min_move_deg:
            sgn = 1.0 if dq[i] >= 0.0 else -1.0
            dq[i] = sgn * (min_move_deg*d2r)
    q_try = ang_wrap_vec(clamp_q(q_from + dq))
    return q_try, ok_scaled

def sample_pair_no_self_collision(joint_min_deg=50.0, tries=800):
    # """พยายามหลายระดับ:
    #    L0: เคร่งสุด = ท่า A/B ผ่าน และ 'ทั้งทาง' ผ่าน
    #    L1: ผ่อนบางอย่าง (ลด PATH_SAMPLES, CLEAR_MARGIN เล็กน้อย) แล้วลองใหม่
    #    L2: หา A/B ที่ท่าผ่าน แล้ว 'ย่อสโตก' ให้ทางปลอดภัยแน่ ๆ """
    # L0: เคร่ง
    for _ in range(tries):
        qA = rand_joint_biased()
        if not is_pose_self_clear(qA): continue
        for _ in range(tries//3):
            qB = rand_joint_biased()
            if not is_pose_self_clear(qB): continue
            if float(np.linalg.norm(ang_wrap_vec(qB - qA)))*r2d < joint_min_deg: continue
            if is_path_self_clear(qA, qB):
                return qA, qB

    # L1: ผ่อนเล็กน้อยเฉพาะตอนตรวจทาง
    samples_backup = globals().get("PATH_SAMPLES", 11)
    margin_backup  = globals().get("CLEAR_MARGIN", 0.02)
    try:
        globals()["PATH_SAMPLES"] = max(7, samples_backup//2)
        globals()["CLEAR_MARGIN"] = max(0.012, margin_backup*0.7)
        for _ in range(tries):
            qA = rand_joint_biased()
            if not is_pose_self_clear(qA): continue
            for _ in range(tries//3):
                qB = rand_joint_biased()
                if not is_pose_self_clear(qB): continue
                if float(np.linalg.norm(ang_wrap_vec(qB - qA)))*r2d < (joint_min_deg-10): continue
                if is_path_self_clear(qA, qB):
                    return qA, qB
    finally:
        globals()["PATH_SAMPLES"] = samples_backup
        globals()["CLEAR_MARGIN"] = margin_backup

    # L2: หา A/B ที่ท่าผ่าน แล้ว 'ย่อสโตก' ให้ทางปลอดภัย
    for _ in range(tries):
        qA = rand_joint_biased()
        if not is_pose_self_clear(qA): continue
        for _ in range(tries//2):
            qB = rand_joint_biased()
            if not is_pose_self_clear(qB): continue
            if float(np.linalg.norm(ang_wrap_vec(qB - qA)))*r2d < (joint_min_deg-20): continue
            qB_safe, scaled = scale_until_self_safe(qA, qB, max_halves=8, min_move_deg=8.0)
            if is_path_self_clear(qA, qB_safe):
                return qA, qB_safe
    return None, None

def sample_two_poses_poses_only(joint_min_deg=25.0, tries=4000):
    # สุ่ม qA, qB ที่ 'แต่ละท่า' ผ่าน self-collision (ยังไม่บังคับทาง)
    for _ in range(tries):
        qA = rand_joint_biased()
        if not is_pose_self_clear(qA):
            continue
        for _ in range(tries//3):
            qB = rand_joint_biased()
            if not is_pose_self_clear(qB):
                continue
            if float(np.linalg.norm(ang_wrap_vec(qB - qA)))*r2d < joint_min_deg:
                continue
            return qA, qB
    return None, None

def pick_safe_pair_or_scale(joint_min_deg=20.0, max_tries=6000):
    # 1) พยายามแบบเข้มก่อน (ได้เร็วก็จบ)
    qA, qB = sample_pair_no_self_collision(joint_min_deg=max(joint_min_deg, 30.0), tries=max_tries)
    if (qA is not None) and (qB is not None):
        return qA, qB

    # 2) สุ่มหาท่า A ที่ "ปลอดภัยแน่ๆ" (เฉพาะโพส)
    qA = None
    for _ in range(max_tries):
        q = rand_joint_biased()
        if is_pose_self_clear(q):
            qA = q; break
    if qA is None:
        return None, None

    # 3) สร้าง B จาก A ด้วยเดลต้าแบบคุมปริมาณ แล้ว "ย่อสโตก" จนทางปลอดภัย
    for _ in range(max_tries):
        step_deg = np.array([40, 25, 25, 30, 40, 40], float)
        step = np.deg2rad(step_deg) * np.random.uniform(0.8, 1.2, size=6) * np.sign(np.random.randn(6))
        qB_try = ang_wrap_vec(clamp_q(qA + step))
        if not is_pose_self_clear(qB_try):
            continue
        qB_safe, _ = scale_until_self_safe(qA, qB_try, max_halves=8, min_move_deg=8.0)
        if is_path_self_clear(qA, qB_safe):
            return qA, qB_safe
    return None, None

def guaranteed_pair_from_mid():
    # """
    # สร้าง qA กลางช่วง + ทำ qB = qA + step คุมปริมาณ
    # แล้วบังคับย่อสโตกจนเส้นทางปลอดภัย — ต้องได้ผลเสมอถ้าขอบเขตไม่ขัดกัน
    # """
    mid = np.deg2rad(np.array([
        0.0,                      # q1
        (ELBOW_RANGE_DEG[0]+ELBOW_RANGE_DEG[1])/2.0,
        (FOREARM_RANGE_DEG[0]+FOREARM_RANGE_DEG[1])/2.0,
        0.0, 0.0, 0.0
    ], float))
    qA = ang_wrap_vec(clamp_q(mid))
    # ถ้าโพสกลางยังไม่ผ่าน ให้ขยับฐานเล็กน้อยวนหา
    for base_deg in [0, 20, -20, 45, -45, 70, -70, 110, -110, 150, -150]:
        q_try = qA.copy()
        q_try[0] = ang_wrap(q_try[0] + np.deg2rad(base_deg))
        if is_pose_self_clear(q_try):
            qA = q_try
            break
    if not is_pose_self_clear(qA):
        return None, None

    # กำหนดก้าวปลอดภัยสำหรับแต่ละแกน
    step_deg = np.array([35, 22, 22, 25, 35, 35], float)
    for _ in range(80):
        sgn = np.sign(np.random.randn(6)); sgn[sgn==0] = 1.0
        step = np.deg2rad(step_deg) * np.random.uniform(0.8, 1.2, size=6) * sgn
        qB_try = ang_wrap_vec(clamp_q(qA + step))
        if not is_pose_self_clear(qB_try):
            continue
        qB_safe, _ = scale_until_self_safe(qA, qB_try, max_halves=10, min_move_deg=6.0)
        if is_path_self_clear(qA, qB_safe):
            return qA, qB_safe
    # ทางตัน: ลดก้าวลงแล้วลองอีกรอบ
    step_deg_small = np.array([20, 15, 15, 18, 25, 25], float)
    for _ in range(120):
        sgn = np.sign(np.random.randn(6)); sgn[sgn==0] = 1.0
        step = np.deg2rad(step_deg_small) * np.random.uniform(0.9, 1.1, size=6) * sgn
        qB_try = ang_wrap_vec(clamp_q(qA + step))
        if not is_pose_self_clear(qB_try):
            continue
        qB_safe, _ = scale_until_self_safe(qA, qB_try, max_halves=12, min_move_deg=5.0)
        if is_path_self_clear(qA, qB_safe):
            return qA, qB_safe
    return None, None

# ===== ground-only safety (no self-collision) =====
def is_pose_self_clear(theta):
    th = ang_wrap_vec(theta)
    T_list, _ = _chain_from_DH(th)

    # ตรวจลิสต์จุดแทนลิงก์ + ปลายมือ (EE)
    check_ids = SELF_SPHERE_IDX + [len(T_list) - 1]  # EE index = ตัวสุดท้าย
    for idx in check_ids:
        z = float(T_list[idx][2, 3])  # world z
        if z < (GROUND_Z + GROUND_MARGIN):
            return False
    return True

def is_path_self_clear(q0, qf, samples=PATH_SAMPLES):
    q0 = ang_wrap_vec(np.asarray(q0, float))
    qf = ang_wrap_vec(np.asarray(qf, float))
    for s in np.linspace(0.0, 1.0, samples):
        qt = ang_wrap_vec((1.0 - s)*q0 + s*qf)
        if not is_pose_self_clear(qt):
            return False
    return True
# ==================================================

# ===================== Parabolic blend trajectory =====================
def parabolic_blend_coeffs(q0, qf, T, amax_each):
    # """
    # Linear interpolation with parabolic blend (LIPB)
    # q(0)=q0, q(T)=qf, dq(0)=dq(T)=0
    # แต่ละ joint ใช้ amax แต่เวลารวม T ร่วมกัน
    # คืน: (a, tb, v_const) สำหรับแต่ละ joint
    # """
    q0, qf, amax_each = np.asarray(q0,float), np.asarray(qf,float), np.asarray(amax_each,float)
    dq = qf - q0
    tb_each = np.sqrt(np.abs(dq)/amax_each)      # เวลาช่วงเร่ง/หน่วง ถ้าใช้ amax เต็ม
    tb_each = np.clip(tb_each, 1e-3, T/2.5)      # กันไม่ให้เกินครึ่งเวลา
    tb = np.min(tb_each)                         # ใช้ค่าเล็กสุดให้ทุกข้อไปพร้อมกัน
    v_const = dq / (T - tb)                      # ความเร็วคงที่แต่ละข้อ
    return amax_each, tb, v_const

def parabolic_blend_eval(q0, qf, amax_each, tb, v_const, t, T):
    # """
    # ประเมิน q,dq,ddq ที่เวลา t
    # """
    q0, qf = np.asarray(q0,float), np.asarray(qf,float)
    dq = qf - q0
    q = np.zeros_like(q0)
    dq_out = np.zeros_like(q0)
    ddq_out = np.zeros_like(q0)

    for i in range(6):
        a = amax_each[i]
        if t < tb:  # phase เร่ง
            q[i] = q0[i] + 0.5*a*t*t*np.sign(dq[i])
            dq_out[i] = a*t*np.sign(dq[i])
            ddq_out[i] = a*np.sign(dq[i])
        elif t < (T - tb):  # phase linear constant vel
            q[i] = q0[i] + (0.5*a*tb*tb + v_const[i]*(t - tb))
            dq_out[i] = v_const[i]
            ddq_out[i] = 0.0
        else:  # phase หน่วง
            td = T - t
            q[i] = qf[i] - 0.5*a*td*td*np.sign(dq[i])
            dq_out[i] = a*td*np.sign(dq[i])
            ddq_out[i] = -a*np.sign(dq[i])
    return q, dq_out, ddq_out

def run_leg_parabolic(sim, aux, hdl_j, hdl_end, q_from, q_to, T_move, amax_each, label):
    coeffs = parabolic_blend_coeffs(q_from, q_to, T_move, amax_each)
    t0 = sim.getSimulationTime()
    last_log = -1.0
    log_once(sim, aux, f"{label} start LIPB (T={T_move:.2f}s)")
    while True:
        t_now = sim.getSimulationTime()
        t = t_now - t0
        if t < 0: t = 0
        if t > T_move: t = T_move

        q_cmd, dq_cmd, ddq_cmd = parabolic_blend_eval(q_from, q_to, *coeffs, t, T_move)
        drive_q(sim, hdl_j, q_cmd)

        if (last_log < 0) or (t_now - last_log > 0.3):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            s = t/T_move
            msg = []
            msg.append(f"{label} s={s:5.2f} t={t_now:6.2f}s")
            msg.append("  q(deg):     " + ", ".join(f"{x*r2d:7.2f}" for x in q_cmd))
            msg.append("  dq(deg/s):  " + ", ".join(f"{x*r2d:7.2f}" for x in dq_cmd))
            msg.append("  ddq(deg/s2):" + ", ".join(f"{x*r2d:7.2f}" for x in ddq_cmd))
            msg.append("  sim_pos:    " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul:    " + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now

        if t >= T_move:
            break
        sim.switchThread()
        
def jacobian_resolved_rate_step(q_cur, p_des, R_des,
                                lam=0.02, w_pos=8.0, w_ori=2.0, dq_clip=10.0*d2r):
    # ใช้ numeric J ของคุณ + DLS 6D, ตามสูตรในไฟล์เดิม
    p_cur, _, Tcur = ur5_forward_kinematrix(q_cur)
    R_cur = Tcur[:3,:3]
    ep = np.asarray(p_des, float) - p_cur
    # error เชิงมุมใน world: ew = R_cur * log(R_des R_cur^T)
    ew_body = log_SO3(R_des @ R_cur.T)
    ew = R_cur @ ew_body
    e6 = np.hstack([ep, ew])

    J = jacobian_numeric(q_cur)
    dq = dls_6d_weighted(J, e6, lam, w_pos=w_pos, w_ori=w_ori)
    dq = np.clip(dq, -dq_clip, dq_clip)
    q_next = ang_wrap_vec(clamp_q(q_cur + dq))
    return q_next, ep, ew

def run_leg_task_cubic(sim, aux, hdl_j, hdl_end,
                       q_start, p_start, p_final, R_hold,
                       T_move, label):
    # เตรียม cubic ในตำแหน่ง
    a0, a2, a3 = cubic_pos_coeffs(p_start, p_final, T_move)

    # seed เป็น q_start
    q_cmd = ang_wrap_vec(q_start.copy())
    t0 = sim.getSimulationTime()
    last_log = -1.0
    log_once(sim, aux, f"{label} start TASK cubic (T={T_move:.2f}s)")

    while True:
        t_now = sim.getSimulationTime()
        t = t_now - t0
        if t < 0.0: t = 0.0
        if t > T_move: t = T_move

        p_des, v_des, a_des = cubic_pos_eval(a0, a2, a3, t)
        # ก้าว IK แบบ Jacobian 1 สเต็ปไปทาง p_des, R_hold
        q_cmd, ep, ew = jacobian_resolved_rate_step(q_cmd, p_des, R_hold,
                                                    lam=0.02, w_pos=10.0, w_ori=2.0,
                                                    dq_clip=12.0*d2r)
        drive_q(sim, hdl_j, q_cmd)

        if (last_log < 0) or (t_now - last_log > 0.25):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            s = t/T_move if T_move>0 else 1.0
            msg = []
            msg.append(f"{label} s={s:5.2f} t={t_now:6.2f}s")
            msg.append("  p_des:       " + ", ".join(f"{x:7.4f}" for x in p_des))
            msg.append("  v_des:       " + ", ".join(f"{x:7.4f}" for x in v_des))
            msg.append("  a_des:       " + ", ".join(f"{x:7.4f}" for x in a_des))
            msg.append("  |ep|,|ew|:   " + f"{np.linalg.norm(ep):.4e}, {np.linalg.norm(ew):.4e}")
            msg.append("  sim_pos:     " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul(deg):" + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now

        if t >= T_move:
            break
        sim.switchThread()

    return q_cmd  # คืน q สุดท้ายที่วิ่งถึงปลายทาง

def solve_pose_from_seed(q_seed, p_target, R_target, iters=220):
    # ใช้ไอเดียเดียวกับ ik_priority_dls (ซึ่งอาศัย Jacobian อยู่แล้ว)
    # แต่ขอ “light” ลูปสั้น ๆ เพื่อไปให้ถึง p/R
    q = ang_wrap_vec(q_seed.copy())
    for _ in range(iters):
        p_cur, _, Tcur = ur5_forward_kinematrix(q)
        R_cur = Tcur[:3,:3]
        ep = p_target - p_cur
        if np.linalg.norm(ep) < 1.5e-4:
            # orientation เก็บด้วย gain ต่ำ ๆ
            pass
        ew_body = log_SO3(R_target @ R_cur.T)
        ew = R_cur @ ew_body
        if (np.linalg.norm(ep) < 1.0e-4) and (np.linalg.norm(ew) < 2.0e-3):
            break
        J = jacobian_numeric(q)
        dq = dls_6d_weighted(J, np.hstack([ep, ew]), lam=0.02, w_pos=9.0, w_ori=2.0)
        q = ang_wrap_vec(clamp_q(q + np.clip(dq, -10.0*d2r, 10.0*d2r)))
    ok = (np.linalg.norm(ep) < 1.0e-4) and (np.linalg.norm(ew) < 2.0e-3)
    return q, ok

def sample_task_pair_with_fixed_R(tries=4000):
    # เลือก pA,pB แบบสุ่มให้ค่อนข้างสมเหตุผล + orientation คงเดิม (Top-down/Side ก็ได้)
    # แล้วหา qA,qB ด้วย Jacobian IK
    for _ in range(tries):
        pA = sample_task_position()
        pB = sample_task_position()
        # เลือก R ให้คงเดิมตาม pA (top-down/side โดยอัตโนมัติ)
        R_hold, mode = pick_auto_approach(pA, force_mode=None)

        # seed q1, q2
        q_seed = rand_joint_biased()
        qA, okA = solve_pose_from_seed(q_seed, pA, R_hold, iters=250)
        if not okA: continue
        qB, okB = solve_pose_from_seed(qA, pB, R_hold, iters=250)
        if not okB: continue

        # ตรวจ ground safety ระหว่างทางแบบหยาบ ๆ ใน joint-space
        if is_path_self_clear(qA, qB):
            return (pA, pB, R_hold, qA, qB)
    return None, None, None, None, None

# ===================== Task-space cubic (position only, orientation fixed) =====================
# จำกัดความเร่งเชิงเส้นสูงสุดต่อแกน XYZ (m/s^2)
ALIN_MAX = np.array([1.0, 1.0, 1.0], float)   # ปรับได้ตามที่ต้องการ

# ช่วงสุ่มตำแหน่ง EE (เมตร) ให้เหมาะกับ UR5
WS_X = (0.25, 0.65)
WS_Y = (-0.35, 0.35)
WS_Z = (0.25, 0.85)

def sample_task_position():
    x = np.random.uniform(*WS_X)
    y = np.random.uniform(*WS_Y)
    z = np.random.uniform(*WS_Z)
    return np.array([x, y, z], float)

def cubic_pos_coeffs(p0, pf, T):
    # p(t) = a0 + a2 t^2 + a3 t^3, v(0)=v(T)=0  ->  a2 = 3*d/T^2, a3 = -2*d/T^3
    p0 = np.asarray(p0, float); pf = np.asarray(pf, float)
    d = pf - p0
    a0 = p0
    a2 = 3.0 * d / (T*T)
    a3 = -2.0 * d / (T*T*T)
    return a0, a2, a3

def cubic_pos_eval(a0, a2, a3, t):
    p  = a0 + a2*(t*t) + a3*(t*t*t)
    v  =        2*a2*t  + 3*a3*(t*t)
    acc=        2*a2     + 6*a3*t
    return p, v, acc

def choose_T_task_with_accel(p0, pf, amax_lin, T_hint):
    # สำหรับ cubic v(0)=v(T)=0: |a|max ต่อแกน = 6|dp|/T^2  =>  T >= sqrt(6|dp|/amax)
    dp = np.abs(np.asarray(pf, float) - np.asarray(p0, float))
    T_req_axes = np.where(amax_lin > 0, np.sqrt(np.where(dp>0, 6.0*dp/amax_lin, 0.0)), 0.0)
    T_req = float(np.max(T_req_axes))
    return max(T_req, float(T_hint))

# ===================== Task-space Linear Interp. with Parabolic Blend (LIPB) =====================
# s(t) เป็นสเกลาร์วิ่งจาก 0→1 บนเส้นตรง p(t)=pA + s(t)*(pB-pA)
# ใช้ข้อจำกัดความเร่งเชิงเส้นสูงสุด per-axis: |a_i(t)| = |s_ddot * d_i| <= ALIN_MAX[i]
# => เลือก s_ddot_max = min_i (ALIN_MAX[i] / |d_i|) สำหรับแกนที่ d_i != 0

def _amax_s_from_disp(d, amax_lin, eps=1e-9):
    d = np.asarray(d, float); amax_lin = np.asarray(amax_lin, float)
    mask = np.abs(d) > eps
    if not np.any(mask):
        # pA≈pB: ระยะทางแทบเป็นศูนย์ — ใช้ค่า amax_s ปลอดภัยๆ
        return 1.0
    return float(np.min(amax_lin[mask] / np.abs(d[mask])))

def choose_T_task_with_blend(p0, pf, amax_lin, T_hint):
    # เงื่อนไขทราเปโซอิดสมมาตร (เริ่ม/จบหยุด):
    # 1 = amax_s * tb * (T - tb)  และต้องมี T >= 2/sqrt(amax_s) เพื่อให้มีคำตอบ tb จริง
    d = np.asarray(pf, float) - np.asarray(p0, float)
    amax_s = _amax_s_from_disp(d, amax_lin)
    T_req = 2.0 / math.sqrt(max(amax_s, 1e-12))
    return max(float(T_hint), float(T_req))

def lipb_scalar_params(p0, pf, amax_lin, T):
    # คืนพารามิเตอร์ของ s(t) = LIPB: (amax_s, tb, v_s_max, disp_vector)
    d = np.asarray(pf, float) - np.asarray(p0, float)
    amax_s = _amax_s_from_disp(d, amax_lin)
    # แก้ tb จาก: 1 = amax_s * tb * (T - tb)  => tb = (T - sqrt(T^2 - 4/amax_s))/2
    disc = max(0.0, T*T - 4.0/max(amax_s,1e-12))
    tb = 0.5 * (T - math.sqrt(disc))
    tb = float(np.clip(tb, 0.0, T/2.0))
    v_s = amax_s * tb
    return amax_s, tb, v_s, d

def lipb_scalar_eval(t, T, amax_s, tb):
    # piecewise: เร่ง (0→tb), คงที่ (tb→T-tb), หน่วง (T-tb→T)
    if t < tb:
        s    = 0.5*amax_s*t*t
        sd   = amax_s*t
        sdd  = amax_s
    elif t <= (T - tb):
        s    = 0.5*amax_s*tb*tb + (amax_s*tb)*(t - tb)
        sd   = amax_s*tb
        sdd  = 0.0
    else:
        td   = T - t
        s    = 1.0 - 0.5*amax_s*td*td
        sd   = amax_s*td
        sdd  = -amax_s
    return s, sd, sdd

def run_leg_task_lipb(sim, aux, hdl_j, hdl_end,
                      q_start, p_start, p_final, R_hold,
                      T_move, label):
    # เตรียมพารามิเตอร์ LIPB
    amax_s, tb, v_s, d = lipb_scalar_params(p_start, p_final, ALIN_MAX, T_move)

    q_cmd = ang_wrap_vec(q_start.copy())
    t0 = sim.getSimulationTime()
    last_log = -1.0
    log_once(sim, aux, f"{label} start TASK LIPB (T={T_move:.2f}s, tb={tb:.3f}s)")

    while True:
        t_now = sim.getSimulationTime()
        t = t_now - t0
        if t < 0.0: t = 0.0
        if t > T_move: t = T_move

        s, sd, sdd = lipb_scalar_eval(t, T_move, amax_s, tb)
        p_des  = p_start + s  * d
        v_des  = sd  * d
        a_des  = sdd * d

        # ไล่ตาม p_des / R_hold ด้วยก้าว Jacobian DLS
        q_cmd, ep, ew = jacobian_resolved_rate_step(
            q_cmd, p_des, R_hold,
            lam=0.02, w_pos=10.0, w_ori=2.0, dq_clip=12.0*d2r
        )
        drive_q(sim, hdl_j, q_cmd)

        if (last_log < 0) or (t_now - last_log > 0.25):
            sim_pos = sim.getObjectPosition(hdl_end, -1)
            sim_eul = sim.getObjectOrientation(hdl_end, -1)
            s_norm = t/T_move if T_move>0 else 1.0
            msg = []
            msg.append(f"{label} s={s_norm:5.2f} t={t_now:6.2f}s")
            msg.append("  p_des:       " + ", ".join(f"{x:7.4f}" for x in p_des))
            msg.append("  v_des:       " + ", ".join(f"{x:7.4f}" for x in v_des))
            msg.append("  a_des:       " + ", ".join(f"{x:7.4f}" for x in a_des))
            msg.append("  |ep|,|ew|:   " + f"{np.linalg.norm(ep):.4e}, {np.linalg.norm(ew):.4e}")
            msg.append("  sim_pos:     " + ", ".join(f"{x:7.4f}" for x in sim_pos))
            msg.append("  sim_eul(deg):" + ", ".join(f"{x*r2d:7.2f}" for x in sim_eul))
            log_once(sim, aux, "\n".join(msg))
            last_log = t_now

        if t >= T_move:
            break
        sim.switchThread()

    return q_cmd  # มุมข้อสุดท้ายที่ไปถึงจุดปลาย

# ===================== Hooks ของ CoppeliaSim =====================
def sysCall_init():
    sim = require("sim")

def sysCall_thread():
    sim = require("sim")

    # ---------- Handles ----------
    hdl_j = {}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_end = sim.getObject("/UR5/EndPoint")

    # ---------- Console ----------
    global aux
    aux = sim.auxiliaryConsoleOpen(
        'UR5 Joint-space Cubic (direct, self-safe)',
        3000, 0, [16,16], [980,800], [1,1,1], [0,0,0]
    )

    # --------- Main demo loop ----------
    T_HINT = 10.0        # วินาที (กำหนดได้เอง; ถ้าไม่พอจะถูกขยายโดย choose_T_task_with_blend)
    HOLD_START = 3.0
    HOLD_END   = 3.0
    amax_each_deg = np.array([80, 80, 80, 120, 160, 180], float)  # ยังเก็บไว้ใช้ clamp joint-step
    amax_each     = np.deg2rad(amax_each_deg)
    NUM_CASES = 3

    log_once(sim, aux, f"[DEMO] Task-space LIPB (pos-only, fixed R) — cases={NUM_CASES}")
    log_once(sim, aux, "  alin_max(m/s^2): " + ", ".join(f"{x:.2f}" for x in ALIN_MAX))
    log_once(sim, aux, "  amax_each(deg/s^2): " + ", ".join(f"{x:.0f}" for x in amax_each_deg))

    for case in range(1, NUM_CASES+1):
        pA, pB, R_hold, qA, qB = sample_task_pair_with_fixed_R(tries=5000)
        if pA is None:
            log_once(sim, aux, f"[Case {case}] ERROR: สุ่ม A/B ที่ IK ผ่านไม่สำเร็จ — ข้ามเคส")
            continue

        log_once(sim, aux, f"[Case {case}] pA: " + ", ".join(f"{x:7.3f}" for x in pA))
        log_once(sim, aux, f"[Case {case}] pB: " + ", ".join(f"{x:7.3f}" for x in pB))

        # เวลาเคลื่อนที่ที่ "แก้สมการได้" ตามข้อจำกัดความเร่งเชิงเส้น (หา T ให้มี tb จริง)
        T_AB = choose_T_task_with_blend(pA, pB, ALIN_MAX, T_HINT)
        log_once(sim, aux, f"[Case {case}] T_AB={T_AB:.2f}s")

        # เทเลพอร์ตนิ่ม ๆ ไปที่ท่าเริ่ม (qA)
        t0 = sim.getSimulationTime()
        while sim.getSimulationTime() - t0 < 0.7:
            drive_q(sim, hdl_j, qA)
            sim.switchThread()

        # รอที่จุดเริ่ม
        t_holdA = sim.getSimulationTime()
        while sim.getSimulationTime() - t_holdA < HOLD_START:
            drive_q(sim, hdl_j, qA)
            sim.switchThread()

        # วิ่ง A -> B ด้วย task-space LIPB (orientation คงเดิม = R_hold)
        q_end = run_leg_task_lipb(
            sim, aux, hdl_j, hdl_end,
            qA, pA, pB, R_hold, T_AB,
            f"[Case {case}] A→B (task-LIPB)"
        )

        # รอที่จุดปลาย
        t_holdB = sim.getSimulationTime()
        while sim.getSimulationTime() - t_holdB < HOLD_END:
            drive_q(sim, hdl_j, q_end)
            sim.switchThread()

        log_once(sim, aux, f"[Case {case}] done\n" + "-"*90)

    log_once(sim, aux, "[DEMO] all done.")

def sysCall_actuation():
    pass

def sysCall_sensing():
    pass

def sysCall_cleanup():
    pass
