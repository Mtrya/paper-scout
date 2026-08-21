"""把 robosuite binding_utils 的 PyOpenGL-EGL 顶层 import 改为惰性(仅当用到 opengl/osmesa 渲染器时)。
LIBERO 评测用 renderer='mujoco'(mujoco 原生渲染器,不走 PyOpenGL),该补丁不影响评测路径。"""
import re
import sys

p = sys.argv[1]
src = open(p).read()
old = "from robosuite.renderers.context.egl_context import EGLGLContext as GLContext"
new = (
    "try:\n"
    "    from robosuite.renderers.context.egl_context import EGLGLContext as GLContext\n"
    "except Exception as e:\n"
    "    GLContext = None\n"
    "    print('[robosuite-patch] PyOpenGL-EGL unavailable, GLContext=None:', e)"
)
assert old in src, "pattern not found"
src = src.replace(old, new)
open(p, "w").write(src)
print("patched:", p)
