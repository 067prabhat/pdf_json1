# PDF to JSON Extractor

## Overview
This project processes PDF files to generate structured JSON output with a document title and heading hierarchy (`H1`, `H2`, `H3`).  
It uses **PyMuPDF** for text extraction and a pre-trained **CatBoost model** to classify text segments into heading levels.

---

## Approach
1. **Feature Extraction**  
   - Each text block from the PDF is analyzed for font size, bold/underline styles, relative positions, and spacing.
   - Additional features like title-case ratio, stopword ratio, and punctuation are computed.
   - PyMuPDF (`fitz`) is used to parse PDFs efficiently.

2. **Prediction**  
   - A **CatBoost model** (stored as `catboost_smote_model.joblib`) predicts whether a text block is a title or a heading (`H1`, `H2`, `H3`).
   - Predictions are combined to build a structured JSON outline.

3. **Output**  
   - For every `filename.pdf` in `/app/input`, the output is written as `/app/output/filename.json`.

---

## Libraries and Model
- **PyMuPDF** (`pymupdf`) for PDF parsing
- **CatBoost** for heading classification
- **Pandas**, **NumPy**, **scikit-learn** for data processing
- **Joblib** for loading the model
- The model size is **< 200MB** and works fully offline.

---

## Expected Execution
The container is expected to:
- Automatically process all PDFs from `/app/input`
- Generate `<filename>.json` for each `<filename>.pdf` inside `/app/output`

---

## Build and Run Instructions

### **1. Build the Docker Image**
```bash
docker buildx build --platform linux/amd64 -t pdf2json:dev --load .
