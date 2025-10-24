#python

def sysCall_init():
    sim = require('sim')
    sim.setStepping(True)  # ใช้โหมด manual stepping


def sysCall_thread():
    sim = require('sim')

    conveyor = sim.getObject('/conveyor')
    box = sim.getObject('/TargetCube')
    
    # ---- หยุดสายพาน ----
    sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))

    SPEED = 0.1
    RUN_STEPS = 150
    WAIT_TIME = 1.0  # รอกล่องอยู่ในจุด start 1 วินาที

    # ---- ขอบเขตสำหรับจุด start ของสายพาน (แก้ได้ตามขนาดจริง) ----
    start_x_min, start_x_max =  -0.45, -0.3    # ตำแหน่ง X ด้านต้นสายพาน
    start_y_min, start_y_max = -0.10, 0.1   # ตำแหน่ง Y ด้านต้นสายพาน
    start_z_min, start_z_max = 0.02, 0.12     # ความสูงจากผิวสายพาน

    # ---- ฟังก์ชันตรวจสอบว่ากล่องอยู่บริเวณจุด start หรือไม่ ----
    def box_in_start_zone():
        pos = sim.getObjectPosition(box, conveyor)
        if pos is None:
            return False
        x, y, z = pos
        # print(x, y, z)
        return (start_x_min < x < start_x_max) and (start_y_min < y < start_y_max) and (start_z_min < z < start_z_max)

    # ---- รอจนกล่องอยู่ที่จุด start ต่อเนื่อง 1 วินาที ----
    t_detect = None
    while not sim.getSimulationStopping():
        if box_in_start_zone():
            if t_detect is None:
                t_detect = sim.getSimulationTime()
            elif sim.getSimulationTime() - t_detect >= WAIT_TIME:
                break  # อยู่ครบ 1 วินาที
        else:
            t_detect = None
        sim.step()

    # ---- เริ่มเดินสายพาน ----
    sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': SPEED}))
    sim.addLog(sim.verbosity_scriptinfos, f"[conveyor] box detected — running at {SPEED:.2f} m/s")

    # ---- เดินสายพาน 200 step ----
    for _ in range(RUN_STEPS):
        if sim.getSimulationStopping():
            break
        sim.step()

    # ---- หยุดสายพาน ----
    sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))
    sim.addLog(sim.verbosity_scriptinfos, f"[conveyor] stopped after {RUN_STEPS} steps")

    # ---- ให้ simulation เดินต่อปกติ ----
    while not sim.getSimulationStopping():
        sim.step()
