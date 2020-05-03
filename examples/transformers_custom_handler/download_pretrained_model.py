import argparse
import os
from transformers import AutoModelForSequenceClassification, AutoConfig, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-checkpoint-name",
        default="distilbert-base-uncased-finetuned-sst-2-english",
    )
    parser.add_argument(
        "--tokenizer-checkpoint-name", default="distilbert-base-uncased"
    )
    parser.add_argument(
        "--output-directory",
        default=os.path.join(os.path.dirname(__file__), "pretrained_model_checkpoint"),
    )
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint_name
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint_name)

    model.save_pretrained(args.output_directory)
    tokenizer.save_pretrained(args.output_directory)


if __name__ == "__main__":
    main()
