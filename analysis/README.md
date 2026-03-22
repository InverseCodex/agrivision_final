# Vegetation Analysis Module

This module now runs the production thesis checkpoints on the server side.

## What It Does

- Loads the final health-status checkpoint:
  - `models/health-status/rrl_health_mobilenet_rebuilt_best.pt`
- Loads the final growth-stage checkpoint:
  - `models/growth-stage/rrl_stage_mobilenet_realistic_best.pt`
- Rebuilds the hybrid MobileNetV3-Large + tabular classifier used in the thesis training scripts.
- Applies ImageNet normalization and the checkpoint-selected tabular features at inference time.
- Supports full-image analysis plus masked crop and segment analysis used by the website.

## Runtime Notes

- Model paths can be overridden with:
  - `AGRIVISION_HEALTH_MODEL_PATH`
  - `AGRIVISION_STAGE_MODEL_PATH`
  - `AGRIVISION_MODEL_DEVICE`
- The repo keeps copies of the production checkpoints in `models/` so local runs do not depend on the original thesis folder.

## Quick Verification

```powershell
.venv\Scripts\python.exe scripts\verify_ml_inference.py
```
