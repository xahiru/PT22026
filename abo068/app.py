"""
Sentiment Analysis App - Professional Training Midterm Project
Student: Ossama Abouaazza (abo068)

Model: distilbert-base-uncased-finetuned-sst-2-english (Hugging Face)
Task: classify English text as POSITIVE or NEGATIVE.
"""

import gradio as gr
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_CHARS = 2000  # limit input size to keep the app fast

# Load the model once, when the app starts.
try:
    classifier = pipeline("sentiment-analysis", model=MODEL_NAME, top_k=None)
except Exception as error:
    classifier = None
    print(f"Could not load the model: {error}")


def analyze(text):
    """Return the score of each label and a short summary sentence."""
    if classifier is None:
        raise gr.Error("The model is not loaded. Check your internet connection.")

    if not text or not text.strip():
        raise gr.Error("Please enter some text.")

    text = text.strip()
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text is too long (maximum {MAX_CHARS} characters).")

    try:
        results = classifier(text, truncation=True)
        # Some library versions wrap the result in an extra list.
        if results and isinstance(results[0], list):
            results = results[0]
    except Exception as error:
        raise gr.Error(f"Prediction failed: {error}")

    scores = {item["label"]: float(item["score"]) for item in results}
    best = max(scores, key=scores.get)
    summary = f"Prediction: {best} (confidence {scores[best]:.1%})"
    return scores, summary


def build_interface():
    """Create the Gradio user interface."""
    return gr.Interface(
        fn=analyze,
        inputs=gr.Textbox(lines=4, label="Enter your text",
                          placeholder="Type a sentence or a review here..."),
        outputs=[gr.Label(num_top_classes=2, label="Scores"),
                 gr.Textbox(label="Result")],
        title="Sentiment Analysis",
        description="Detects whether an English text is positive or negative "
                    "using a pre-trained DistilBERT model.",
        examples=[["I love this course, it is amazing!"],
                  ["The service was terrible and very slow."],
                  ["The movie was okay, nothing special."]],
        flagging_mode="never",
    )


if __name__ == "__main__":
    build_interface().launch()