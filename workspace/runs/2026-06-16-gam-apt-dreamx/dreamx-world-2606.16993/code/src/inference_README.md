# Instructions for using the model

<!-- ## Parameters
```bash
# ======================== Model Path ========================
MODEL_NAME="./Wan2.2-TI2V-5B"    # Path to the folder containing the Wan2.2-5B-TI2V model weights.
CONFIG_PATH="./configs/wan2.2/wan_ti2v_5b.yaml" # Path to model config file.
TRANSFORMER_PATH="./Dreamx-5b/"  # Path to the folder containing the DreamX model weights.
# ====================== Basic settings ======================
INPUT_DIR="./configs/dreamx/eval.json"          # Json file of inputs, containing image, prompt, and camera control.
OUTPUT_DIR="./outputs/"          # Directory of saving output video.
SAMPLE_HEIGHT=704                # Height of the input image/output video.
SAMPLE_WIDTH=1280                # Width of the input image/output video.
VIDEO_LENGTH=121                 # Number of frames (must satisfy 1+4k pattern). 121 frames for 5-second 24fps video and 81 frames for 5-second 16fps video.
FPS=24                           # FPS of the output video. Supports 24 and 16 FPS.
GUIDANCE_SCALE=3.0               # CFG scale.
NUM_INFERENCE_STEPS=50           # Number of sampling steps.
SEED=42                          # Random seed for noise sampling.

# ====================== Camera Control ======================
CAM_METHOD="prope"               # camera control method
ADD_CONTROL_ADAPTER="--add_control_adapter"

# ======================== Multi-GPU ========================
WEIGHT_DTYPE="bfloat16"          # inference dtype.
ULYSSES_DEGREE=8                 # ulysses degree, 1 for no ulysses.
RING_DEGREE=1                    # ring degree, 1 for no ring.
CUDA_DEVICES="0,1,2,3,4,5,6,7"   # Specify GPUs, e.g., "4,5,6,7". Empty = use all available.
``` -->

### Camera Action Commands

| Action | Description |
|--------|-------------|
| `w` | Move forward |
| `s` | Move backward |
| `a` | Move left |
| `d` | Move right |
| `j` | Tilt down |
| `k` | Tilt up |
| `l` | Pan right |
| `h` | Pan left |

Actions can be composed (e.g., `wj` = move forward + tilt down, `dj` = move right + tilt down).


### Step1: Preparing Input Json File
Prepare your input JSON file (see `configs/dreamx/eval.json` for examples):

```json
{
  "image_path": "./demo/your_image.png",
  "caption": "Style: Photorealistic. A description of the scene...",
  "action_seq": ["w", "wj"],
  "action_speed_list": [4, 6]
}
```

### Step2: Run Inference

#### 1. DreamX-World-5B-Cam
- Generates 5-second videos at 24 FPS (121 frames) or 16 FPS (81 frames).
- Supports up to 7.5s (in 16FPS) video generation.


```bash
sh inference_dreamx_5b.sh
```

Parameters that you may want to change:
```bash
# ====================== Basic settings ======================
MODEL_NAME="./Wan2.2-TI2V-5B"    # Path to the folder containing the Wan2.2-5B-TI2V model weights.
TRANSFORMER_PATH="./DreamX-World-5B-Cam"  # Path to the folder containing the DreamX model weights.
INPUT_DIR="./configs/dreamx/eval.json"          # Json file of inputs, containing image, prompt, and camera control.
OUTPUT_DIR="./outputs/"          # Directory of saving output video.
SAMPLE_HEIGHT=704                # Height of the input image/output video.
SAMPLE_WIDTH=1280                # Width of the input image/output video.
VIDEO_LENGTH=121                 # Number of frames (must satisfy 1+4k pattern, e.g., 81, 121).
FPS=24                           # FPS of the output video.
GUIDANCE_SCALE=3.0               # CFG scale.
NUM_INFERENCE_STEPS=50           # Number of sampling steps.
SEED=42                          # Random seed for noise sampling.

# ======================== Multi-GPU ========================
WEIGHT_DTYPE="bfloat16"          # inference dtype.
ULYSSES_DEGREE=8                 # ulysses degree, 1 for no ulysses.
RING_DEGREE=1                    # ring degree, 1 for no ring.
CUDA_DEVICES="0,1,2,3,4,5,6,7"   # Specify GPUs, e.g., "4,5,6,7". Empty = use all available.
```

### Uncurated Videos (5s, 24 FPS): 
<table align="center">
  <tr>
    <td><video src="https://github.com/user-attachments/assets/77958ada-e840-46e8-8609-d0ef548a18ed" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/03cd69bb-9c3b-4336-ae73-d9e8eb7ad68b" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/47c80b08-151c-4efd-9590-91197e29a863" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/49a40e5a-7200-404c-8fa6-d31cffee4ab2" width="100%" autoplay muted loop playsinline></video></td>
  </tr>
  <tr>
    <td><video src="https://github.com/user-attachments/assets/ef35c650-043f-4050-a6ab-ad7101e92a8f" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/525ad5d6-f24b-4556-b12b-1e02735860df" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/89a946ff-f06b-4a49-ba3e-262a850b08e7" width="100%" autoplay muted loop playsinline></video></td>
    <td><video src="https://github.com/user-attachments/assets/8fbd9b81-9579-4fae-9447-d1e05c65319a" width="100%" autoplay muted loop playsinline></video></td>
  </tr>
</table>

You can reproduce the results by running the model with the provided json file: `configs/dreamx/eval.json`.

#### 2. DreamX-World-5B
- Autoregressive model, supports long-horizon video generation, up to 1-min, at 16 FPS.

```bash
sh inference_ar_forcing.sh
```

Parameters that you may want to change:
```bash
BASE_CHECKPOINT_PATH="${BASE_CHECKPOINT_PATH:-/path/to/baseline.pt}"  # Path to DreamX-World-5B model weights
DATA_PATH="${DATA_PATH:-configs/dreamx/eval.json}" # Path to input JSON file
OUTPUT_FOLDER="${OUTPUT_FOLDER:-./outputs_ar/}" # Path to save output video
NUM_OUTPUT_FRAMES=123             # Latent frames, shall be divisible by 3. Pixel frames = (N-1)*4+1. 21→81 pixels (5s@16fps), 63→249 pixels
SEED=42                          # Random seed

# ====================== Post-processing ======================
COLOR_CORRECTION_STRENGTH=1.0    # Lab color correction (0=off, 1=full). This could slightly mitigate the color shifting problem.
CUDA_DEVICES="${CUDA_DEVICES:-0}" # If more than 1 gpu is given, sequence parallel will be activated
```