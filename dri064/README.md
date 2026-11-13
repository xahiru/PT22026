# Text Summarization App

An interactive web app that summarizes long text using pre-trained
Transformer models from Hugging Face. Built with Python, Transformers and Gradio.

## Features
- Paste any text and get a short summary
- Choose between 3 models (DistilBART, T5-small, BART-large-CNN)
- Set minimum / maximum summary length
- Shows word counts and compression rate
- Evaluation script using ROUGE scores

## Models
| Model | Source |
|---|---|
| sshleifer/distilbart-cnn-12-6 | https://huggingface.co/sshleifer/distilbart-cnn-12-6 |
| t5-small | https://huggingface.co/t5-small |
| facebook/bart-large-cnn | https://huggingface.co/facebook/bart-large-cnn |

Models are downloaded automatically on first use (not stored in this repo).

## Installation
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
python app.py                 # opens the app at http://127.0.0.1:7860
python evaluate_model.py      # computes ROUGE scores -> results.txt
```

## Dataset
CNN/DailyMail (loaded automatically by `datasets`): https://huggingface.co/datasets/cnn_dailymail

## Limitations
- Input longer than the model limit (~1024 tokens) is truncated
- Models can produce facts that are not in the original text
- Trained on English news; weaker on other domains/languages

## Author
Student: <your name>, Student ID: <your ID>
