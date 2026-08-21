import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco
import numpy as np

xml = '''
<mujoco><worldbody>
  <light pos="0 0 3"/>
  <geom type="box" size="0.2 0.2 0.2" pos="0 0 0.2"/>
  <camera name="cam0" pos="1 0 0.5" xyaxes="0 1 0 -1 0 0"/>
</worldbody></mujoco>
'''
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
cid = m.camera('cam0').id
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
cam.fixedcamid = cid
scn = mujoco.MjvScene(m, 1000)
vopt = mujoco.MjvOption()
pert = mujoco.MjvPerturb()
mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
print('cam forward before:', cam.forward, 'up:', cam.up)
print('model cam_pos:', m.cam_pos[cid])
m.cam_pos[cid][0] += 0.5
print('after write, model cam_pos:', m.cam_pos[cid])
mujoco.mjv_updateScene(m, d, vopt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
print('cam forward after updateScene:', cam.forward, 'up:', cam.up)
