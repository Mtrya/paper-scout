# SPEAR (arXiv 2607.06701) — deep-dive thread packet

Paper: *SPEAR: A Simulator for Photorealistic Embodied AI Research* — Roberts, Wang, Zawar, ... Koltun
(Adobe Research / Intel Labs / Manycore Tech / NVIDIA / ETH Zurich / Imperial). ECCV 2026.
Repo: https://github.com/spear-sim/spear — cloned shallow to `code/spear-repo/` (workspace root, git-ignored), 2026-07-17, branch `main`.
Local paper text: `papers/robotics/spear-simulator-2607.06701.md`.

## What was attempted

1. Read the full paper (abstract through references).
2. Cloned the repo and statically traced the four claims: (a) Python-controls-any-UE-app via modular plugins;
   (b) 14K UE functions exposed; (c) one Python API across engine versions; (d) faster rendering than existing
   UE plugins. No UE build was attempted (per instructions) — all verification is static source tracing plus
   counting scripts run over the source tree.
3. Downloaded and verified 2 report figures from the arXiv HTML build (see bottom).

## Architecture as verified in code

**RPC layer.** Client and server are C++ over **rpclib (msgpack-rpc) on TCP/IP**, default port 30000
(`cpp/unreal_plugins/SpServices/Source/SpServices/RpcService.h:32`; rpclib is a git submodule in `third_party/`).
The server binds exactly one generic RPC method, `"rpc_server.call"`, which dispatches by function-name string
through a `FuncRegistry` to strongly typed C++ lambdas (`RpcServer.h:34-67`). The Python-facing client is a
**nanobind** extension (`python_ext/cpp/spear_ext.cpp`, `client.h`); at connect time it fetches every entry
point's type signature from the server (`client.h:46`) and packs args into typed msgpack arrays
(`client.h:76-110`). Python `spear` package wraps this (`python/spear/services/*.py`).

**Threading model.** The RPC server runs on its own thread inside the UE process ("server thread");
`EngineService` maintains `WorkQueue`s (boost::asio io_contexts, `WorkQueue.h`) that the game thread drains
at begin/end of frame. Entry points are bound either to the worker thread or the game thread
(`bindFuncToExecuteOnGameThread` / `...WorkerThread` throughout `SpServices/Source`). Every game-thread entry
point gets three auto-generated variants via template metaprogramming — `call_sync`, `call_async` (returns a
future), `send_async` (`EngineService.h:385-425`). Python-side `begin_frame`/`end_frame` context managers map
to `engine_service.begin_frame/execute_frame/end_frame` RPCs; deterministic stepping is
`end_frame(single_step=True)` (paper Fig. 4 implements AirSim/UnrealCV+/Habitat/CARLA/MuJoCo stepping idioms
in this model).

**The 14K functions: no code generator.** There is no UHT/binding codegen anywhere in the repo. The mechanism
is one generic reflection path: Python attribute access on a `UnrealObject` (`python/spear/unreal_object.py`)
resolves a UFunction by name string via `unreal_service.find_function_by_name`, then calls
`unreal_service.call_function` (`UnrealService.h:815`) → `UnrealUtils::callFunction`
(`cpp/unreal_plugins/SpCore/Source/SpCore/UnrealUtils.cpp:164`), which:
allocates a `ParmsSize`-byte buffer, `InitializeStruct`s it, walks the UFunction's `FProperty` params with
`TFieldIterator`, sets each argument from its string form (UE's JSON property import — this is why Python
dicts work as universal argument representation), calls **`uobject->ProcessEvent(ufunction, buf)`** (line 254 —
the same dispatch path Blueprints use), then re-reads params for out-args/return value. New C++ functions
become callable by adding `UFUNCTION`/`UPROPERTY` annotations anywhere — no SPEAR-side registration.

**Hand-crafted entry points.** Paper claims 193. My count (`code/count_entry_points.py` over
`SpServices/Source`): **208 unique (service, entry-point) pairs at this commit**, of which 162 are
`unreal_service` (i.e., ~78% exist to expose the reflection system itself, matching the paper's "roughly 75%"),
plus engine/engine_globals/input/enhanced_input/navigation/shared_memory/sp_func/world_registry/debug
services. The 193-vs-208 gap is consistent with version drift and/or excluding debug/editor-only entry points;
claim verified in substance. Editor-side module `SpServicesEditor` adds editor control on top.

**Bulk data path.** `SpFuncComponent` lets any actor register named "SpFunctions" taking named data arrays +
object-string maps + user string (`sp_func_service.call_function`, `SpFuncService.h`). The camera sensor
`USpSceneCaptureComponent2D` (`cpp/unreal_plugins/SpUnrealTypes/.../SpSceneCaptureComponent2D.{h,cpp}`)
registers `enqueue_copy`/`read_pixels` SpFuncs (.cpp:265,277), supports `ESpBufferingMode::{Single,Double,
TripleBuffered}` (.h:82,171) — this buffering is the paper's "rendering latency" knob — and uses
**boost interprocess shared memory** (`SharedMemoryService`, `SpCore/SharedMemory.h`) so rendered frames land
directly in a user-allocated NumPy array with no extra copy. This is the machinery behind the 73 fps number.

**Plugin architecture / any-UE-app claim.** 5 plugins (`SpCore`, `SpServices`, `SpServicesEditor`,
`SpUnrealTypes`, `SpModuleRules`, plus content-only `SpContent`) under `cpp/unreal_plugins/`.
`tools/install_plugins_in_external_project.py` integrates them into any UE project by adding an
`AdditionalPluginDirectories` entry to the .uproject JSON (this is the paper's "single-line declaration").
`SpearSim.uproject` has `EngineAssociation: 5.5` and enables `SpCore` plus stock Epic plugins.

**Engine versions.** `main` branch = **UE 5.5**; a `ue-58` branch targets **UE 5.8** (`docs/getting_started.md:13`).
The "single Python API spans versions" claim is structurally true: the Python package speaks only name-strings
and JSON-ish values over RPC, so nothing Python-side depends on engine version; per-version differences are
absorbed by recompiling the C++ plugins against each engine. Windows/macOS/Linux all supported; on Linux you
install Epic's prebuilt Linux UE 5.5.4 zip (no engine source build required).

## The 14K claim — honest scope assessment

- Table 1's 14,485 functions / 53,537 properties is a **runtime count over the loaded reflection registry**,
  produced by `examples/get_class_info/run.py`: iterate every UClass derived from UObject, sum
  `find_functions(..., IncludeDeprecated)` per class (excludes base-class double-counting), plus struct
  properties. It is *not* a count of hand-written bindings, and it varies with engine version and which
  plugins/projects are loaded.
- Against Table 1's baselines the order-of-magnitude claim is fair: AirSim 92 hand-crafted functions / 0 UE
  functions; CARLA 465 / 0; UnrealCV+ 56 commands exposing 747 UE functions (via console exec); SPEAR 193
  hand-crafted entry points exposing 14,485 UFUNCTIONs + 53,537 UPROPERTYs.
- What it buys in practice: anything Blueprint-reachable is Python-reachable with identical semantics —
  spawning, PCG graphs, Movie Render Queue, path tracer, Enhanced Input, editor scripting, console variables —
  without waiting for the simulator authors to wrap it. What it does *not* buy: type safety, docstrings,
  discoverability (you fish for names via `find_functions_by_name`), or non-reflectable C++ APIs (e.g., most
  rendering internals — those need the 193 hand-crafted entry points or new SpFuncs).
- Performance caveat visible in the code: the generic path stringifies all arguments through UE's JSON
  property serializer per call — fine for poses/actions, not for pixels; hence the separate SpFunc +
  shared-memory fast path.

## Install blockers / buildability for a robotics data-engine user

From `docs/getting_started.md` + pyproject:
- Requires UE 5.5 (or 5.8 branch) installed — Epic account, EULA; Linux gets prebuilt binaries, but the
  **SpearSim project and plugins still need a C++ build** (UE-bundled clang on Linux; VS2022 17.14 on Windows;
  Xcode 16/26 on macOS), plus building third-party boost/rpclib/yaml-cpp via `tools/build_third_party_libs.py`,
  plus `pip install -e python` (Python 3.11, conda recommended), plus `tools/install_python_extension.py` to
  build the nanobind client. Expect tens of GB and a long first build — heavier than `pip install maniskill`,
  lighter than building CARLA.
- Headless: `renderoffscreen` command-line flag in `python/spear/config/default_config.spear.yaml:61`;
  `LAUNCH_MODE` none/editor/game; VK_ICD_FILENAMES knob for NVIDIA Vulkan on Linux. No Docker image, no
  prebuilt wheels/binaries shipped in the repo.
- No gym/Gymnasium wrapper and no ROS bridge anywhere in the repo (checked: grep hits were spurious).
  RL-style control is roll-your-own on `begin_frame/end_frame` (paper Fig. 4 sketches Gym `step()` idioms).
  `navigation_service` (NavMesh random points/paths) is the only robotics-flavored service.
- Asset pipeline: `pipeline/` has free-space/visibility-graph/kinematic-tree/MuJoCo-scene generators; examples
  import Stanford/Mixamo/Humoto assets; interactive scenes come from a separate `spear-pipeline` content repo
  (perforce download tool included) plus Epic sample projects (CitySample, StackOBot, CropoutSample,
  GameAnimationSample, ElectricDreams, MetaHumans, HillsideSample) which carry their own Epic sample licenses.
  `examples/render_image_hypersim` reproduces Hypersim-style GT modalities (depth, normals, semantic/instance
  IDs, non-diffuse intrinsic decomposition, material IDs, PBR shading params).
- MuJoCo co-simulation (`examples/mujoco_interop`): UE state slaved to MuJoCo stepping with user substeps —
  the template for "MuJoCo/Isaac physics + SPEAR rendering" manipulation data generation.
- Extras: MCP server (`tools/run_mcp_server.py`, `mcp==1.28.1` dep) for natural-language scene editing;
  editor control via `SpServicesEditor`; path-tracer control example.

## Paper benchmarks (as reported; not reproduced — Windows/RTX 4090 test rig)

- Table 2 (1920x1080 end-to-end into Python): standalone UE 129.9 fps; standalone + offscreen-render overhead
  56.5 fps; UnrealCV+ 3.5 fps; SPEAR sync/no-shmem 24.7 fps → async+shmem 56.2 fps (0-frame latency);
  with 2-frame rendering latency **73.4 fps** → 9–21x faster than UnrealCV+.
- Table 3 (matched standalone render speed ~90 fps): AirSim 2.6 fps vs SPEAR 32.3 fps (12x, 0-frame latency);
  CARLA 32.7 vs SPEAR 37.1 fps (+10%, 2-frame latency). Also ">150 megapixels/sec" in conclusions.
- Demo tasks: no closed-loop task benchmarks — the demos are capability demonstrations: 6 agents (human, car,
  flying robot, game agents, parkour human, quadruped) across 5 Epic sample projects; PCG control in
  ElectricDreams; MetaHumans multi-view; MuJoCo co-sim; LLM scene editing. No manipulation benchmark, no
  trained policy results.

## Neighborhood triangulation (bounded)

- UnrealCV (2017) is effectively frozen (UE4-era); UnrealZoo/UnrealCV+ (ICCV 2025) is its maintained line and
  is the paper's main baseline (56 text commands; 3.5 fps at 1080p to Python — SPEAR is 9–21x faster
  like-for-like in the same project).
- AirSim: archived by Microsoft in 2022 (Colosseum fork continues); 2.6 fps to Python here; 92-function API.
- CARLA: custom UE fork, monolith, driving-specific; SPEAR beats it modestly (37.1 vs 32.7 fps) at matched
  latency while being engine- and domain-agnostic.
- Isaac Sim: closed-source, Omniverse/USD, strong robotics stack (ROS 2, articulated physics, GPU-parallel
  RL) but photorealism programmability goes through Omniverse Kit extensions — heavier and GPU-hungrier;
  SPEAR's pitch vs Isaac is open MIT code + state-of-the-art UE rendering + full reflection access, at the
  cost of no built-in robotics abstractions.

## Assessment for a manipulation data engine (vs ManiSkill3 / Isaac)

SPEAR is the most convincing "game engine as a *programmable* data engine" artifact to date: the reflection
hook is real, elegant, and small (~27-31K LOC), and the zero-copy camera path solves the actual bottleneck
that made UE simulators painful. For *photorealistic rendering of manipulation scenes authored elsewhere*
(e.g., MuJoCo/physics-driven co-sim, replayed teleop trajectories, BEHAVIOR-style scene scans → UE), it is
genuinely attractive: 56-73 fps at 1080p with Hypersim-grade GT modalities is 10x-class better than any
UE-based alternative and better than Isaac's practical throughput on a single GPU.
But as a *physics-first* manipulation trainer it does not compete with ManiSkill3/Isaac today: UE Chaos is
not GPU-batched, there is no parallel-env story (one instance per process, TCP RPC per instance), no
gym wrapper, no robot assets/articulation tooling out of the box, and UE's install footprint (account +
50GB-class toolchain + long C++ build) is a real adoption barrier. Realistic play: hybrid pipeline —
ManiSkill/Isaac/MuJoCo for dynamics + SPEAR as the photorealistic observation renderer (their own MuJoCo
co-sim example is the proof of concept) — or vision-data generation (navigation/embodied-LLM scene editing)
where physics fidelity is secondary. Watch: whether the ue-58 branch cadence keeps up, and whether anyone
ships a gym wrapper + parallel-instance story.

## Figures saved (verified valid JPEG)

- `runs/2026-07-17-gigaworld05-harness-spear/assets/fig1_control_sample_projects.jpg` (paper Fig. 1,
  2478x1047): 6 embodied agents across Epic sample projects.
- `runs/2026-07-17-gigaworld05-harness-spear/assets/fig2_hypersim_camera_modalities.jpg` (paper Fig. 2,
  8964x1849): camera sensor beauty + Hypersim-style GT modalities.
- (Not saved but available: `figures/timing_unrealcv_21.jpg`, `figures/timing_other_simulators_22.jpg` —
  the Table 2/3 scene images; `figures/mujoco.jpg`; `figures/electric_23.jpg` at the same arXiv HTML base.)

## How to rerun

```bash
# clone (scratch, git-ignored)
git clone --depth 1 https://github.com/spear-sim/spear code/spear-repo
# count hand-crafted entry points
python3 runs/2026-07-17-gigaworld05-harness-spear/spear-2607.06701/code/count_entry_points.py code/spear-repo
# the runtime 14K count requires a running SpearSim instance:
#   build per docs/getting_started.md, then: python examples/get_class_info/run.py
# figures:
curl -sL https://arxiv.org/html/2607.06701v1/figures/control_sample_projects.jpg -o <assets>/fig1_control_sample_projects.jpg
curl -sL https://arxiv.org/html/2607.06701v1/figures/hypersim.jpg -o <assets>/fig2_hypersim_camera_modalities.jpg
```

Key source references: `python_ext/cpp/client.h:46,110` · `SpServices/RpcServer.h:34-67` ·
`SpServices/RpcService.h:32` · `SpServices/EngineService.h:385-425` · `SpServices/UnrealService.h:815` ·
`SpCore/UnrealUtils.cpp:164-258` (ProcessEvent at :254) · `SpUnrealTypes/SpSceneCaptureComponent2D.{h:82,171,
cpp:265,277}` · `python/spear/unreal_object.py:13` · `examples/get_class_info/run.py` ·
`tools/install_plugins_in_external_project.py` · `docs/getting_started.md:13`.
