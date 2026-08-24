import os
os.environ['MUJOCO_GL'] = 'egl'
import mujoco

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
r = mujoco.Renderer(m, 64, 64)
r.update_scene(d, camera='cam0')
img = r.render()
print('mujoco render OK, shape', img.shape, 'mean', float(img.mean()))
