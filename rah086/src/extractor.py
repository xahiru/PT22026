"""Ingredient-list extractor: uses the fine-tuned NER model to find the
ingredient list inside the raw OCR text of a whole food package."""

import re
from pathlib import Path

MODEL_DIR = Path("models/ingredient-ner")
CHUNK_WORDS = 100  # must match the chunk size used in training
MAX_LEN = 256      # must match the training setting

# A word is a run of characters that are not spaces or punctuation;
# every punctuation mark is its own word (same rule as the training data).
_PUNCT = r"!-/:-@\[-`{-~«»¡¿—–‘’“”„…•·"
WORD_PATTERN = re.compile(rf"[^\s{_PUNCT}]+|[{_PUNCT}]")


def split_words(text):
    """Split text into words and return a list of (word, start, end)."""
    return [(m.group(), m.start(), m.end()) for m in WORD_PATTERN.finditer(text)]


def spans_from_tags(words, tags, text):
    """Cut the ingredient list(s) out of the original text.

    Args:
        words: list of (word, start, end) from split_words.
        tags: one predicted tag per word (0 = outside, 1/2 = ingredient).
        text: the original text.

    Returns:
        list of text segments, one per run of ingredient words.
    """
    segments, first, last = [], None, None
    for i, tag in enumerate(tags):
        if tag > 0:
            if first is None:
                first = i
            last = i
        elif first is not None:
            segments.append(text[words[first][1]:words[last][2]])
            first = last = None
    if first is not None:
        segments.append(text[words[first][1]:words[last][2]])
    return segments


class IngredientExtractor:
    """Loads the trained model once and extracts ingredient lists."""

    def __init__(self, model_dir=MODEL_DIR):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
        self.model.eval()

    def predict_tags(self, words):
        """Predict one tag (0/1/2) for every word."""
        torch = self._torch
        tags = [0] * len(words)
        chunks = [list(range(i, min(i + CHUNK_WORDS, len(words))))
                  for i in range(0, len(words), CHUNK_WORDS)]
        for start in range(0, len(chunks), 16):  # small batches
            batch = chunks[start:start + 16]
            enc = self.tokenizer(
                [[words[i][0] for i in chunk] for chunk in batch],
                is_split_into_words=True, truncation=True,
                max_length=MAX_LEN, padding=True, return_tensors="pt")
            with torch.no_grad():
                guesses = self.model(**enc).logits.argmax(dim=-1)
            for b, chunk in enumerate(batch):
                previous = None
                for pos, wid in enumerate(enc.word_ids(batch_index=b)):
                    if wid is not None and wid != previous:
                        tags[chunk[wid]] = int(guesses[b, pos])
                    previous = wid
        return tags

    def extract(self, text):
        """Return the ingredient list(s) found in text as a single string."""
        words = split_words(text)
        if not words:
            return ""
        segments = spans_from_tags(words, self.predict_tags(words), text)
        return " ".join(segments).strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.extractor <image_path>")
        sys.exit(1)

    from src.ocr import extract_text

    ocr_text = extract_text(sys.argv[1])
    print("--- FULL OCR TEXT ---")
    print(ocr_text)
    print("\n--- INGREDIENT LIST FOUND BY THE MODEL ---")
    print(IngredientExtractor().extract(ocr_text) or "(nothing found)")