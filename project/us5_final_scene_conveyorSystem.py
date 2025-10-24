def sysCall_init():
    sim = require('sim')
    sim.setStepping(True)  # manual stepping


def sysCall_thread():
    sim = require('sim')

    conveyor = sim.getObject('/conveyorSystem')
    box = sim.getObject('/TargetCube')

    SPEED = 0.1
    RUN_STEPS = 493
    WAIT_TIME = 1.0     # ต้องเห็นกล่องในโซน start ต่อเนื่อง 1 วินาที
    PAUSE_AFTER = 20.0   # พักหลังหยุดสายพาน 2 วินาที

    # ---- ขอบเขตโซนเริ่มต้น ----
    start_x_min, start_x_max =  0.45, 0.55
    start_y_min, start_y_max = -0.50, -0.30
    start_z_min, start_z_max =  0.02, 0.12

    # ---- helper: ตรวจว่ากล่องอยู่โซน start ----
    def box_in_start_zone():
        pos = sim.getObjectPosition(box, conveyor)
        if pos is None:
            return False
        x, y, z = pos
        return (start_x_min < x < start_x_max) and \
               (start_y_min < y < start_y_max) and \
               (start_z_min < z < start_z_max)

    # ---- เริ่มลูปตลอดการทำงาน ----
    while not sim.getSimulationStopping():
        # 1) หยุดสายพานก่อนเริ่ม
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))

        # 2) รอกล่องอยู่ในโซน start ครบ WAIT_TIME
        sim.addLog(sim.verbosity_scriptinfos, "[conveyor] waiting for box at start zone...")
        t_detect = None
        while not sim.getSimulationStopping():
            if box_in_start_zone():
                if t_detect is None:
                    t_detect = sim.getSimulationTime()
                elif sim.getSimulationTime() - t_detect >= WAIT_TIME:
                    break  # กล่องอยู่ครบเวลา
            else:
                t_detect = None
            sim.step()

        if sim.getSimulationStopping():
            break

        # 3) เริ่มเดินสายพาน
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': SPEED}))
        sim.addLog(sim.verbosity_scriptinfos, f"[conveyor] box detected — running at {SPEED:.2f} m/s")

        # 4) วิ่งตามจำนวน step ที่กำหนด
        for _ in range(RUN_STEPS):
            if sim.getSimulationStopping():
                break
            sim.step()

        # 5) หยุดสายพาน
        sim.setBufferProperty(conveyor, 'customData.__ctrl__', sim.packTable({'vel': 0.0}))
        sim.addLog(sim.verbosity_scriptinfos, f"[conveyor] stopped after {RUN_STEPS} steps")

        # 6) พัก 2 วินาทีก่อนเริ่มรอบใหม่
        t0 = sim.getSimulationTime()
        while (sim.getSimulationTime() - t0 < PAUSE_AFTER) and not sim.getSimulationStopping():
            sim.step()
