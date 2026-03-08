import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis import FieldContext, RGBSnapshot, interpret_field_from_rgb


def main() -> None:
    # Sample RGB statistics from a DJI Mini 4 Pro image.
    snapshot = RGBSnapshot(
        mean_red=108.0,
        mean_green=132.0,
        mean_blue=92.0,
        green_coverage=0.48,
    )
    context = FieldContext(
        crop_name="Rice",
        growth_stage="vegetative",
        rainfall_last_7d_mm=6.0,
        avg_temp_c=34.0,
    )

    report = interpret_field_from_rgb(snapshot, context)

    print("=== Farmer-Friendly Result ===")
    print(report.one_line_summary)
    print(report.simple_explanation)
    print("\nKey findings:")
    for finding in report.model_result.main_findings:
        print(f"- {finding}")
    print("\nRecommended actions:")
    for idx, rec in enumerate(report.recommendations, start=1):
        print(f"{idx}. {rec}")


if __name__ == "__main__":
    main()
