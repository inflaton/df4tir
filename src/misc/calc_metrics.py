import os
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from metrics import *


def print_usage():
    print(
        "Usage: python src/misc/calc_metrics.py <path_to_directory_containing_results> <platform>"
    )


if len(sys.argv) < 3:
    print("Error: No directory path and/or platform provided.")
    print_usage()
    sys.exit(1)

directory = sys.argv[1]
platform = sys.argv[2]
print(f"Calculating metrics for directory: {directory} for platform: {platform} ...")

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=True)

subfolders = [
    "llama3.2-vision_11b-qwen2.5_3b",
    "llama3.2-vision_11b-qwen2.5_7b",
    "llama3.2-vision_11b-functionary-small",
    "llama3.2-vision_11b-qwen2.5_14b",
    "llama3.2-vision_11b-qwen2.5_32b",
    "llama3.2-vision_11b-functionary-medium",
    "llama3.2-vision_11b-qwen2.5_72b",
    "llama3.2-vision_11b-cogito_3b",
    "llama3.2-vision_11b-cogito_8b",
    "llama3.2-vision_11b-cogito_14b",
    "llama3.2-vision_11b-cogito_32b",
    "llama3.2-vision_11b-cogito_70b",
    "gpt-4o-mini-gpt-4o-mini",
    "gpt-4o-gpt-4o",
    "gpt-4.1-nano-gpt-4.1-nano",
    "gpt-4.1-mini-gpt-4.1-mini",
    "gpt-4.1-gpt-4.1",
    "o4-mini-o4-mini",
    "claude-3-5-sonnet-20241022-claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219-claude-3-7-sonnet-20250219",
]


def total_power_in_watt(file_path):
    df = pd.read_csv(file_path)
    return df["CPU Package Power [W]"].mean() + df["GPU Power [W]"].mean()


# Initialize an empty DataFrame
metrics_df = pd.DataFrame()

for subfolder in subfolders:
    dir = f"{directory}/{subfolder}"
    if os.path.exists(dir):
        model = (
            subfolder.replace("gpt-4o-mini-", "")
            .replace("gpt-4o-gpt", "gpt")
            .replace("o4-mini-o4", "o4")
            .replace("gpt-4.1-nano-gpt-4.1-nano", "gpt-4.1-nano")
            .replace("gpt-4.1-mini-gpt-4.1-mini", "gpt-4.1-mini")
            .replace("gpt-4.1-gpt-4.1", "gpt-4.1")
            .replace("-20241022-claude-3-5-sonnet-20241022", "")
            .replace("-20250219-claude-3-7-sonnet-20250219", "")
            .replace("llama3.2-vision_11b-", "")
        )
        print(f"Calculating metrics for model: {model}...")

        db_filepath = f"{dir}/emails.db"
        metrics = calculate_metrics(db_filepath, including_bert_f1=False)
        df = metrics["df"]
        del metrics["df"]

        row_data = {"platform": platform, "model": model, "type": "overall", **metrics}
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([row_data])], ignore_index=True
        )

        df1 = df[df["attachments"] != ""].copy()
        metrics = get_metrics(df1, including_bert_f1=True)

        row_data = {"platform": platform, "model": model, "type": "vision", **metrics}
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([row_data])], ignore_index=True
        )

        df1 = df[df["attachments"] == ""].copy()
        metrics = get_metrics(df1, including_bert_f1=False)

        row_data = {"platform": platform, "model": model, "type": "non-vision", **metrics}
        metrics_df = pd.concat(
            [metrics_df, pd.DataFrame([row_data])], ignore_index=True
        )

metrics_df.to_csv(f"results/metrics_{platform}.csv", index=False)
print(f"Metrics saved as results/metrics_{platform}.csv")
