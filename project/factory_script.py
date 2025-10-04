#luaExec wrapper='pythonWrapper' -- using the old wrapper for backw. compat.
# To switch to the new wrapper, simply remove above line, and add sim=require('sim')
# as the first instruction in sysCall_init() or sysCall_thread()
import time
import numpy as np

pi = np.pi
d2r = pi/180
r2d = 1/d2r

def sysCall_init():
    sim=require("sim")
def sysCall_thread():
    # define handles for axis
    hdl_j={}
    hdl_j[0] = sim.getObject("/UR5/joint")
    hdl_j[1] = sim.getObject("/UR5/joint/link/joint")
    hdl_j[2] = sim.getObject("/UR5/joint/link/joint/link/joint")
    hdl_j[3] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint")
    hdl_j[4] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint")
    hdl_j[5] = sim.getObject("/UR5/joint/link/joint/link/joint/link/joint/link/joint/link/joint")
    
    hdl_end = sim.getObject("/UR5/EndPoint")
    
    t = 0
    t1 = time.time()
#    th = np.array()
#    th = np.array([])
    th = {}
    
#    p={}
    while t<10:
        p = 45*pi/180*np.sin(0.2*pi*t);
#        p = np.pi/2
        
        for i in range(0,6):       
            sim.setJointTargetPosition(hdl_j[i],p)
         
        for i in range(0,6):
            th[i] = round(sim.getJointPosition(hdl_j[i])*r2d,2)
            # print (th[i])
            
        end_pos = sim.getObjectPosition(hdl_end,-1)
        # get Euler's angle X-Y-Z
        end_ori = sim.getObjectOrientation(hdl_end,-1)

        print("-----------------------")
        print("Joint Position: {}".format(th))
        print("End point position: {}".format(np.array(end_pos).round(4)))
        print("End point orientation: {}".format(np.array([x*r2d for x in end_ori]).round(2))) 

        # time
        t = time.time()-t1
        
        sim.switchThread() # resume in next simulation step
    pass

# See the user manual or the available code snippets for additional callback functions and details
