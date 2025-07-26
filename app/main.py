import argparse
import glob
import sys
from pathlib import Path

from app.model_loader import load_model
from app.pipeline import (
    extract_features_from_pdf,
    predict_and_build_json,
    save_json,
)

# Defaults that match your Docker run command
DEFAULT_INPUT_DIR = Path("/app/input")
DEFAULT_OUTPUT_DIR = Path("/app/output")
DEFAULT_MODEL_PATH = Path("/app/app/models/catboost_smote_model.joblib")


def parse_args():
    p = argparse.ArgumentParser(description="Process PDFs to JSON using CatBoost.")
    p.add_argument("--input", "-i", type=Path, default=DEFAULT_INPUT_DIR,
                   help="Input directory containing PDFs (default: /app/input)")
    p.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for JSON files (default: /app/output)")
    p.add_argument("--model", "-m", type=Path, default=DEFAULT_MODEL_PATH,
                   help="Path to the CatBoost model file (default: /app/app/models/catboost_smote_model.joblib)")
    return p.parse_args()


def process_pdf(pdf_path: Path, out_json_path: Path, model) -> None:
    print(f"→ {pdf_path.name}")
    blocks = extract_features_from_pdf(str(pdf_path))
    result = predict_and_build_json(model, blocks)
    save_json(result, str(out_json_path))
    print(f"   Saved: {out_json_path}")


def main():
    args = parse_args()

    # Ensure output dir exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from: {args.model}")
    model = load_model(str(args.model))
    print("Model loaded.\n")

    # Collect PDFs
    pdf_paths = sorted(Path(args.input).glob("*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in: {args.input}")
        return 0

    errors = 0
    for pdf in pdf_paths:
        out_path = args.output / (pdf.stem + ".json")
        try:
            process_pdf(pdf, out_path, model)
        except Exception as e:
            errors += 1
            print(f"✗ Failed on {pdf}: {e}", file=sys.stderr)

    print("\nDone.")
    if errors:
        print(f"{errors} file(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
