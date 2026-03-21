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

AgriVision is a Flask-based web application for thesis demonstration and evaluation. It allows a user to upload an RGB drone or field image, generate a heatzone visualization, compute vegetation-related proxy metrics, review segment-level findings, save result personalization, and export a PDF report for presentation or documentation.

## What the system does

- Authenticates users through Supabase-backed login and registration.
- Accepts image uploads with file-type and size validation.
- Runs RGB-based vegetation analysis and produces a heatzone image.
- Stores original images, generated outputs, and JSON analysis data in Supabase storage/tables.
- Shows a result gallery with filtering, sorting, deletion, and PDF export.
- Provides a detailed result page with crop-area rerun analysis, segment recommendations, saved notes, and advanced RGB metrics.
- Includes a dedicated tutorial page for guided onboarding during demos or defense.

## Tech stack

- Backend: Flask
- Storage and tables: Supabase
- Image processing: Pillow, OpenCV, NumPy, Matplotlib
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
- `analysis/`: RGB interpretation and vegetation proxy logic
- `templates/`: Flask HTML templates
- `static/`: CSS and JavaScript assets
- `tmp/`: Temporary working directory for image processing
- `.env.example`: Required environment variable template

## Environment setup

Create a `.env` file in the project root based on `.env.example`.

Required variables:

- `FLASK_SECRET_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY` or `SUPABASE_ANON_KEY` or `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_TABLE`
- `SUPABASE_IMAGES_TABLE`
- `SUPABASE_RESULTS_TABLE`
- `SUPABASE_BUCKET_ORIGINAL`
- `SUPABASE_BUCKET_HEATZONE`
- `SUPABASE_BUCKET_REPORTS`

## Local run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Defense talking points

- The system is not just a static interface; it includes authentication, storage integration, analysis generation, saved result history, and report export.
- The result page supports deeper inspection through crop-area rerun analysis and segment-level recommendations, which improves interpretability for end users.
- The tutorial flow reduces training overhead and makes the platform easier to demonstrate to non-technical evaluators.
- The app has basic resilience features such as upload-size limits, unsupported-file rejection, and graceful fallbacks when result data is unavailable.

## Current limitations

- Analysis is based on RGB-derived proxy metrics, so it should be presented as a practical field-support tool rather than a laboratory-grade diagnostic system.
- Accuracy depends on image quality, lighting, altitude consistency, and scene composition.
- Live storage and authentication depend on valid Supabase configuration.
- There is not yet a formal automated test suite; smoke validation is currently done through route and compile checks.

## Verification completed

The current project was checked with:

- Python compile validation for `app.py` and `analysis/model.py`
- Flask route smoke checks for `/`, `/tutorial`, `/results`, `/results/<image_id>`, and `/results/<image_id>/report.pdf`

## Recommended presentation framing

Describe AgriVision as a decision-support web platform for early crop-condition interpretation using RGB imagery. Emphasize usability, result explainability, and workflow completeness rather than claiming medical-style precision.
