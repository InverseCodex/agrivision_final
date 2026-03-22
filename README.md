---
title: AgriVision Website
emoji: "🌾"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
fullWidth: true
---

# AgriVision Thesis Website

AgriVision is a Flask-based web application for thesis demonstration and evaluation. It allows a user to upload an RGB drone or field image, run the final thesis health-status and growth-stage ML checkpoints on the server, generate a heatzone visualization, review segment-level findings, save result personalization, and export a PDF report for presentation or documentation.

## What the system does

- Authenticates users through Supabase-backed login and registration.
- Accepts image uploads with file-type and size validation.
- Runs the final hybrid MobileNetV3-Large health-status and growth-stage models on the server and produces a heatzone image.
- Stores original images, generated outputs, and JSON analysis data in Supabase storage/tables.
- Shows a result gallery with filtering, sorting, deletion, and PDF export.
- Provides a detailed result page with crop-area rerun analysis, segment recommendations, saved notes, and advanced RGB metrics.
- Includes a dedicated tutorial page for guided onboarding during demos or defense.

## Tech stack

- Backend: Flask
- Storage and tables: Supabase
- Image processing and inference: Pillow, OpenCV, NumPy, Matplotlib, PyTorch, TorchVision
- Frontend: HTML, CSS, vanilla JavaScript
- Deployment-ready server dependency: Gunicorn

## Thesis-ready workflow

1. A user logs in or creates an account.
2. The user uploads a field image from the dashboard.
3. The backend validates the image, runs RGB analysis, and generates a heatzone image.
4. Results are saved to Supabase and shown in the `View Results` dashboard.
5. The user opens a result to inspect crop-level and segment-level findings.
6. The user can personalize the result, save field notes, and export a PDF report.
7. The tutorial page can be used during the defense to explain the complete workflow without needing live uploads.

## Project structure

- `app.py`: Main Flask application, routes, upload flow, Supabase integration, PDF export, rerun logic
- `analysis/`: model loading, RGB feature extraction, and inference helpers
- `models/`: production thesis checkpoints and their report metadata
- `templates/`: Flask HTML templates
- `static/`: CSS and JavaScript assets
- `tmp/`: Temporary working directory for image processing
- `.env.example`: Required environment variable template

## Environment setup

Create a `.env` file in the project root based on `.env.example`.

For Hugging Face Docker Spaces, `.env` is not copied into the container because it is excluded by `.dockerignore`. Add the same values in your Space Settings as Variables/Secrets instead.

Required variables:

- `FLASK_SECRET_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` or `SUPABASE_KEY` or `SUPABASE_ANON_KEY`
- `SUPABASE_TABLE`
- `SUPABASE_IMAGES_TABLE`
- `SUPABASE_RESULTS_TABLE`
- `SUPABASE_BUCKET_ORIGINAL`
- `SUPABASE_BUCKET_HEATZONE`
- `SUPABASE_BUCKET_REPORTS`
- `AGRIVISION_HEALTH_MODEL_PATH`
- `AGRIVISION_STAGE_MODEL_PATH`
- `AGRIVISION_MODEL_DEVICE` (optional)
- `AGRIVISION_RUNTIME_ROOT` (optional override for writable temp/cache storage)

Recommended for Spaces:

- Use `SUPABASE_SERVICE_ROLE_KEY` as a Space Secret for this server-side Flask app.
- Run `supabase/001_bootstrap_schema.sql` in the Supabase SQL Editor before first login so `user_info`, `uploaded_images`, and `analysis_results` exist with the expected columns.
- If you are testing login from the Hugging Face wrapper page and your browser blocks embedded cookies, also test the direct `https://<your-space-subdomain>.hf.space` URL.

## Local run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Hugging Face deployment notes

- The app auto-detects Hugging Face Spaces and switches Flask session cookies to `SameSite=None` with secure cookies so iframe-based logins have the best chance of working.
- If you override cookie settings manually, keep `SESSION_COOKIE_SECURE=true` whenever `SESSION_COOKIE_SAMESITE=None`.
- The app now stores temporary downloads and cached preview media in a writable runtime directory. On Spaces it prefers `/data/agrivision` when available, then falls back to the system temp directory. You can override this with `AGRIVISION_RUNTIME_ROOT`.
- Runtime startup logs now warn when Supabase credentials are missing or when the default Flask secret is still in use.

Inference smoke check:

```powershell
.venv\Scripts\python.exe scripts\verify_ml_inference.py
```

## Defense talking points

- The system is not just a static interface; it includes authentication, storage integration, analysis generation, saved result history, and report export.
- The result page supports deeper inspection through crop-area rerun analysis and segment-level recommendations, which improves interpretability for end users.
- The tutorial flow reduces training overhead and makes the platform easier to demonstrate to non-technical evaluators.
- The app has basic resilience features such as upload-size limits, unsupported-file rejection, and graceful fallbacks when result data is unavailable.

## Current limitations

- The primary health-status and growth-stage outputs now come from the final thesis hybrid MobileNetV3-Large checkpoints, but the surrounding vegetation metrics and heatzone remain RGB-derived support signals.
- Accuracy depends on image quality, lighting, altitude consistency, and scene composition.
- Live storage and authentication depend on valid Supabase configuration.
- Automated verification is still lightweight and currently focuses on inference smoke checks and targeted unit tests.

## Verification completed

The current project was checked with:

- `python -m py_compile app.py analysis\vegetation_damage_model.py analysis\__init__.py scripts\verify_ml_inference.py tests\test_vegetation_model.py`
- `.venv\Scripts\python.exe -m unittest tests.test_vegetation_model`
- `.venv\Scripts\python.exe scripts\verify_ml_inference.py`

## Recommended presentation framing

Describe AgriVision as a decision-support web platform for early crop-condition interpretation using RGB imagery. Emphasize usability, result explainability, and workflow completeness rather than claiming medical-style precision.
