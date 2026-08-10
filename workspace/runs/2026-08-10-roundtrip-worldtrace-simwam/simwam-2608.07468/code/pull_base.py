from modelscope import snapshot_download
snapshot_download("DiffSynth-Studio/Wan-Series-Converted-Safetensors",
  local_dir="checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors",
  allow_file_pattern=["models_t5_umt5-xxl-enc-bf16.safetensors","Wan2.2_VAE.safetensors"])
print("T5VAE_DONE")
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B",
  local_dir="checkpoints/Wan-AI/Wan2.1-T2V-1.3B",
  allow_file_pattern=["google/umt5-xxl/*"])
print("TOKENIZER_DONE")
