# Vegetation Analysis Module

This module provides a backend-only farm analysis workflow using vegetation indices.

## What It Does

- Supports RGB-only drone input (DJI Mini 4 Pro) using proxy indices:
  - `VARI`, `GLI`, `NGRDI`, `ExG`
- Also keeps a multispectral path (`NDVI`, `EVI`, `SAVI`, `GNDVI`, `NDRE`) if needed later.
- Uses an AI-style judge (`VegetationJudgeModel`) to produce:
  - health score
  - health band (`healthy`, `mature`, `watch`, `stressed`, `critical`)
  - confidence
  - key findings
- Converts model output into simple farmer-friendly language and action steps.

## Quick Run

```powershell
.venv\Scripts\python.exe scripts\run_field_analysis_demo.py
```

## Integration Idea (Backend)

Use this for DJI Mini 4 Pro RGB-based analysis:

```python
from analysis import RGBSnapshot, FieldContext, interpret_field_from_rgb

snapshot = RGBSnapshot(mean_red=108, mean_green=132, mean_blue=92, green_coverage=0.48)
context = FieldContext(crop_name="Rice", growth_stage="vegetative", rainfall_last_7d_mm=8, avg_temp_c=33)
report = interpret_field_from_rgb(snapshot, context)
```

Then persist `report` to your results table and show it in UI later.
