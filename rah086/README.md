# AI-Powered Food Allergen Detection and Ingredient Risk Assistant Using NLP and OCR

**Student:** Rahimi Mohammad Mirwais
**Folder:** rah086
**Course:** Professional Training (YN3012170034)

## Overview
This application helps people with food allergies check a product quickly.
The user uploads a photo of an ingredient label (or pastes the text), selects
their allergies, and receives a risk report.

## How It Works
1. **OCR:** EasyOCR reads the text from the label image.
2. **NLP detection:** The text is tokenized and matched against a dictionary of
   14 major allergen groups. Fuzzy matching (Python `difflib`) tolerates OCR
   mistakes such as `WHEAL` instead of `WHEAT`.
3. **Risk assessment:** Text after phrases such as "may contain" is treated as a
   trace warning. Each allergy selected by the user is rated:
   - HIGH: the allergen is a direct ingredient
   - MEDIUM: the allergen appears only in a "may contain" warning
   - LOW: the allergen was not detected
4. **Interface:** A Gradio web app shows the report and the extracted text.

## Project Structure
```
rah086/
├── app.py              # Gradio web application
├── requirements.txt    # Python dependencies
├── src/
│   ├── ocr.py          # OCR module (EasyOCR)
│   ├── detector.py     # Allergen detection (keywords + fuzzy matching)
│   ├── risk.py         # Risk assessment logic
│   └── evaluate.py     # Evaluation on labelled test cases
├── data/
│   ├── README.md
│   └── samples/        # Sample label images
├── tests/
└── docs/
```

## Installation
Requires Python 3.10 or newer.

```bash
git clone https://github.com/rahimimohammadmirwais/PT22026.git
cd PT22026/rah086

python -m venv .venv
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the EasyOCR models (about 100 MB). They are stored
outside the project and are not uploaded to GitHub.

## Usage
**Web app:**
```bash
python app.py
```
Open http://127.0.0.1:7860, upload a label image (or paste text), select your
allergies and click **Analyze**.

**Command line:**
```bash
python -m src.ocr data/samples/label1.jpg                  # OCR only
python -m src.detector data/samples/label1.jpg             # OCR + allergens
python -m src.risk data/samples/label1.jpg "soy,fish"      # full risk report
python -m src.evaluate                                     # evaluation
```

## Evaluation
The detector was tested on 14 labelled ingredient texts, including cases with
simulated OCR errors.

| Metric    | Score |
|-----------|-------|
| Precision | 1.00  |
| Recall    | 0.96  |
| F1 score  | 0.98  |

13 of 14 cases were fully correct. The one failure is an OCR confusion where
"m" was read as "rn" (`sesarne` instead of `sesame`).

## Models and Methods
- **EasyOCR** (CRAFT text detector + CRNN recognizer) for text extraction.
- **Rule-based NLP** with keyword matching and fuzzy string similarity.
- No model training or fine-tuning was needed for this version.

## Limitations
- OCR accuracy drops on blurry, curved or very small text.
- Detection depends on the keyword dictionary, so uncommon allergen names can be missed.
- Only English labels are supported.
- This tool is an aid, **not medical advice**. Always read the original label.

## Acknowledgements
- EasyOCR: https://github.com/JaidedAI/EasyOCR
- Gradio: https://www.gradio.app
- Allergen list based on the major food allergens recognised in EU and Codex regulations.
- Sample images are used for educational purposes only.