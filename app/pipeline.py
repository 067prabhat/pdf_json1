import json
import string
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
CLASS_NAMES = ['H1', 'H2', 'H3', 'None', 'Title']


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def smart_join_lines(lines: List[str]) -> str:
    """Intelligently joins lines with proper spacing."""
    joined = []
    for i, line in enumerate(lines):
        if i == 0:
            joined.append(line)
        else:
            if joined[-1].endswith(' ') or line.startswith(' '):
                joined.append(line)
            else:
                joined.append(' ' + line)
    return ''.join(joined).strip()


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to pure Python for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ---------------------------------------------------------------------
# Core PDF → features
# ---------------------------------------------------------------------
def extract_features_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Your full feature extractor, refactored to be offline (uses sklearn stopwords).
    Returns a list of dicts (one per text block) with engineered features.
    """
    doc = fitz.open(pdf_path)
    all_font_sizes_doc = []
    all_text_blocks_raw = []
    stop_words = set(ENGLISH_STOP_WORDS)

    # First pass: Extract raw text blocks and gather all font sizes
    for page_num, page in enumerate(doc):
        page_height = page.rect.height
        page_width = page.rect.width
        page_dict = page.get_text("dict")
        blocks = page_dict.get('blocks', [])

        # Calculate max font size for current page
        page_font_sizes = []
        for b in blocks:
            if b.get('type') == 0:  # Text block
                for line in b.get('lines', []):
                    for span in line.get('spans', []):
                        page_font_sizes.append(round(span.get('size', 1.0), 2))
        max_font_size_page = max(page_font_sizes) if page_font_sizes else 1.0

        # Store all blocks for space calculation
        page_blocks = []

        for b_idx, b in enumerate(blocks):
            if b.get('type') != 0:  # Skip non-text blocks
                continue

            # Heuristic to skip table-like blocks
            is_table_block = False
            if len(b.get('lines', [])) > 1:
                x_coords = [line['spans'][0]['origin'][0] for line in b['lines'] if line.get('spans')]
                if len(set(round(x, 1) for x in x_coords)) > 1:
                    is_table_block = True
            if is_table_block:
                continue

            # Merge spans into lines
            line_texts = []
            underline_present = False
            for line in b.get('lines', []):
                line_text = ""
                for span in line.get('spans', []):
                    line_text += span.get('text', '')
                    # underline flag 4
                    if span.get('flags', 0) & 4:
                        underline_present = True
                if line_text.strip():
                    line_texts.append(line_text)

            if not line_texts:
                continue

            combined_text = smart_join_lines(line_texts)

            first_span = b['lines'][0]['spans'][0]
            line_font_size = round(first_span.get('size', 1.0), 2)
            is_bold = bool(first_span.get('flags', 0) & 16)

            all_font_sizes_doc.append(line_font_size)

            block_data = {
                "text": combined_text,
                "font_size": line_font_size,
                "bbox": b.get('bbox', [0, 0, 0, 0]),
                "page_num": page_num,
                "page_height": page_height,
                "page_width": page_width,
                "max_font_size_page": max_font_size_page,
                "is_bold": is_bold,
                "is_underlined": underline_present,
                "block_index": b_idx,
            }

            all_text_blocks_raw.append(block_data)
            page_blocks.append(block_data)

        # Calculate space above and below for each block (within the page)
        for i, block in enumerate(page_blocks):
            if i == 0:
                space_above = block['bbox'][1]  # distance from top
            else:
                space_above = block['bbox'][1] - page_blocks[i - 1]['bbox'][3]

            if i == len(page_blocks) - 1:
                space_below = page_height - block['bbox'][3]
            else:
                space_below = page_blocks[i + 1]['bbox'][1] - block['bbox'][3]

            block['space_above'] = max(0, space_above)
            block['space_below'] = max(0, space_below)

    # document-level max font size
    max_font_size_pdf = max(all_font_sizes_doc) if all_font_sizes_doc else 1.0

    # Second pass: Compute relative features
    processed_data = []
    page_dimensions = {p_num: {'height': page.rect.height, 'width': page.rect.width}
                       for p_num, page in enumerate(doc)}

    for block_data in all_text_blocks_raw:
        text = block_data['text']
        words = text.strip().split()
        font_size = block_data['font_size']
        bbox = block_data['bbox']
        page_num = block_data['page_num']

        current_page_height = page_dimensions[page_num]['height']
        current_page_width = page_dimensions[page_num]['width']
        max_font_size_page = block_data['max_font_size_page']

        x0, y0, _, _ = bbox

        font_size_relative_to_max_pdf = font_size / max_font_size_pdf
        font_size_relative_to_max_page = (font_size / max_font_size_page) if max_font_size_page else 1.0
        is_bold = block_data['is_bold']
        num_words = len(words)

        last_char = text.strip()[-1] if text.strip() else ''
        punctuation_char = last_char if last_char in string.punctuation else "NULL"

        x_pos_relative = x0 / current_page_width if current_page_width else 0.0
        y_pos_relative = y0 / current_page_height if current_page_height else 0.0

        if num_words > 0:
            title_case_words = sum(1 for w in words if w and w[0].isupper())
            title_case_ratio = title_case_words / num_words
            stopword_count = sum(1 for w in words if w.lower() in stop_words)
            stopword_ratio = stopword_count / num_words
        else:
            title_case_ratio = 0.0
            stopword_ratio = 0.0

        space_above = block_data.get('space_above', 0) / current_page_height if current_page_height else 0.0
        space_below = block_data.get('space_below', 0) / current_page_height if current_page_height else 0.0
        is_underlined = block_data.get('is_underlined', False)

        processed_data.append({
            "text": text,
            "font_size_relative_to_max_pdf": font_size_relative_to_max_pdf,
            "font_size_relative_to_max_page": font_size_relative_to_max_page,
            "is_bold": is_bold,
            "num_words": num_words,
            "punctuation": punctuation_char,
            "x_pos_relative": x_pos_relative,
            "y_pos_relative": y_pos_relative,
            "page_no": page_num,
            "title_case_ratio": title_case_ratio,
            "stopword_ratio": stopword_ratio,
            "space_above": space_above,
            "space_below": space_below,
            "is_underlined": is_underlined,
        })

    doc.close()
    return processed_data


# ---------------------------------------------------------------------
# Features → model input
# ---------------------------------------------------------------------
def _align_with_model_features(X: pd.DataFrame, model) -> pd.DataFrame:
    """
    Make sure the feature matrix columns match what the model was trained on.
    Handles dummy cols for 'punctuation' and missing columns.
    """
    # Convert punctuation to one-hot (if present)
    if 'punctuation' in X.columns:
        X['punctuation'] = X['punctuation'].astype(str)
        X = pd.get_dummies(X, columns=['punctuation'], drop_first=False)

    # Find expected columns
    expected_cols = None
    if hasattr(model, 'feature_names_in_'):
        expected_cols = list(model.feature_names_in_)
    elif hasattr(model, 'feature_names_'):
        # CatBoost sometimes stores this attribute
        expected_cols = list(model.feature_names_)
    else:
        # If nothing is exposed, assume current X is fine
        expected_cols = list(X.columns)

    # Add missing cols
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0

    # Reorder
    X = X[expected_cols]

    # Impute NaNs
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X


# ---------------------------------------------------------------------
# Model inference → JSON
# ---------------------------------------------------------------------
def predict_and_build_json(model, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1) Turns your processed blocks into a DataFrame / model features
    2) Runs model.predict / predict_proba
    3) Builds the final JSON with:
       - "title"
       - "outline": list of H1/H2/H3 nodes
    """
    df = pd.DataFrame(input_data)
    if df.empty:
        return {"title": "", "outline": []}

    # Extract features (drop raw text)
    X = df.drop(columns=['text'], errors='ignore')

    X = _align_with_model_features(X, model)

    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X) if hasattr(model, 'predict_proba') else None

    # Map predictions to class names safely
    pred_levels = []
    for pred in predictions:
        # CatBoost may return np.ndarray of shape (1,)
        p = pred.item() if isinstance(pred, np.ndarray) else pred
        pred_levels.append(CLASS_NAMES[int(p)])

    # ====== TITLE LOGIC ======
    title_buffer: List[str] = []
    previous_page = None

    for idx, (level, item) in enumerate(zip(pred_levels, input_data)):
        text = item.get("text", "").strip()
        page = int(item.get("page_no", 0))

        if level == "Title":
            if previous_page is None or previous_page == page:
                title_buffer.append(text)
                previous_page = page
            else:
                break  # discontinuous page => break title chain
        elif title_buffer:
            break  # non-title encountered after some title lines

    title = " ".join(title_buffer).strip()

    # ====== HEADING OUTLINE LOGIC ======
    outline = []
    for idx, (item, level) in enumerate(zip(input_data, pred_levels)):
        if level in ["None", "Title"]:
            continue

        text = item.get("text", "").strip()
        page = int(item.get("page_no", 0))

        # Filters
        if len(text.split()) > 15:
            continue

        if probabilities is not None:
            none_prob = probabilities[idx][CLASS_NAMES.index("None")]
            heading_probs = [probabilities[idx][CLASS_NAMES.index(tag)] for tag in ['H1', 'H2', 'H3']]
            if none_prob >= 0.25 or max(heading_probs) <= 0.3:
                continue

        outline.append({
            "level": level,
            "text": text,
            "page": page
        })

    final_json = {
        "title": str(title),
        "outline": outline
    }

    return convert_numpy_types(final_json)


def save_json(data: Dict[str, Any], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(data), f, indent=2, ensure_ascii=False)
