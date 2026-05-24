"""
Load the saved V3 RoBERTa model and predict sentiment for new movie reviews.

Usage:
    python predict.py "This movie was absolutely brilliant."
    python predict.py --file reviews.txt
"""

import argparse
import contextlib
import io
import re
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, logging as hf_logging

# suppress HuggingFace info messages
hf_logging.set_verbosity_error()

# paths to the fine-tuned RoBERTa artifacts saved by sentiment_analysis_v3.ipynb
MODEL_PATH     = "models/v3/roberta_sentiment"
TOKENIZER_PATH = "models/v3/roberta_tokenizer"

# truncate reviews to 512 tokens — matches the max_length used during training
MAX_LEN = 512

# use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_artifacts():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        # redirect stderr to suppress the safetensors loading bar
        with contextlib.redirect_stderr(io.StringIO()):
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()  # disable dropout for inference
    except (OSError, Exception):
        sys.exit(
            "Model files not found. Run sentiment_analysis_v3.ipynb on Kaggle first "
            "and download models_v3_export.zip into models/v3/."
        )
    return tokenizer, model


def clean_text(text: str) -> str:
    # remove HTML tags (e.g. <br />) — same minimal cleaning as V3 notebook
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict(review: str, tokenizer, model) -> tuple:
    cleaned = clean_text(review)

    # tokenize and move tensors to the same device as the model
    inputs = tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

    # run inference without computing gradients
    with torch.no_grad():
        outputs = model(**inputs)

    # convert logits to probabilities and pick the highest
    probs      = torch.nn.functional.softmax(outputs.logits, dim=-1)
    pred_class = probs.argmax().item()
    confidence = probs.max().item()

    label = "Positive" if pred_class == 1 else "Negative"
    return label, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Predict sentiment for a movie review using the V3 RoBERTa model."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("review", nargs="?", help="Review text in quotes")
    group.add_argument("--file", "-f", help="Path to a text file with one review per line")
    args = parser.parse_args()

    tokenizer, model = load_artifacts()

    if args.review:
        label, confidence = predict(args.review, tokenizer, model)
        print(f"Sentiment  : {label}")
        print(f"Confidence : {confidence:.4f}")
    else:
        with open(args.file, encoding="utf-8") as fh:
            reviews = [line.strip() for line in fh if line.strip()]
        print(f"Predicting {len(reviews)} reviews...\n")
        for i, review in enumerate(reviews, 1):
            label, confidence = predict(review, tokenizer, model)
            print(f"[{i}] {label} ({confidence:.4f})")
            print(f"    {review[:100]}")
            print()


if __name__ == "__main__":
    main()
