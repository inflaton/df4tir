import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import json
import traceback

df_ground_truth = pd.read_csv("dataset/ground_truth.csv")


def plot_value_distribution(df2, col="category", top_n=10):
    df2[col].value_counts()[:top_n].plot(kind="bar")
    plt.title(f"Distribution of {col}")

    # add the count on top of the bars
    for i in range(len(df2[col].value_counts()[:top_n])):
        count = df2[col].value_counts().values[i]
        plt.text(i, count, count, ha="center")

    plt.show()


def print_row_details(df, indices=[0], columns=None):
    if columns is None:
        columns = df.columns
    for index in indices:
        for col in columns:
            print("-" * 50)
            print(f"{col}: {df[col].iloc[index]}")
        print("=" * 50)


def get_invoice_id(row):
    return row["subject"].split("Invoice ID:")[-1].strip()


def verify_ocr_result(row, result):
    result = json.loads(result)["content"]
    invoice_id = get_invoice_id(row)

    df_ground_truth_row = df_ground_truth[
        df_ground_truth["invoice_id"] == int(invoice_id)
    ].iloc[0]

    element = df_ground_truth_row["amount"]
    amount = f"{element:.0f}"

    return invoice_id in result and amount in result


tool_call_configs = [
    {
        "name": "ocr_tool",
        "prefix": "OCR_TOOL",
        "verify_args": lambda row, args: (
            row["attachments"] == args["image_path"] if "image_path" in args else False
        ),
        "verify_result": lambda row, result: verify_ocr_result(row, result),
    },
    {
        "name": "invoice_db_query_tool",
        "prefix": "DB_QUERY_TOOL",
        "verify_args": lambda row, args: (
            get_invoice_id(row) == args["invoice_id"] if "invoice_id" in args else False
        ),
        "verify_result": lambda row, result: (
            get_invoice_id(row) == json.loads(result)["content"]["invoice_id"]
            if "invoice_id" in json.loads(result)["content"]
            else False
        ),
    },
    {
        "name": "invoice_db_update_tool",
        "prefix": "DB_UPDATE_TOOL",
        "verify_args": lambda row, args: (
            args["email_details"] is not None
            and get_invoice_id(row) == args["invoice_id"]
            if "invoice_id" in args and "email_details" in args
            else False
        ),
        "verify_result": lambda row, result: "DONE" == json.loads(result)["content"],
    },
]


def update_process_status(row):
    if row["process_status"] == "NOT_STARTED":
        return "NOT_STARTED"
    try:
        logs = json.loads(row["full_logs"])
        tool_call_logs = [[], [], []]
        starting = 0 if row["attachments"] else 1
        counter = starting

        for log in logs:
            if len(tool_call_logs[counter]) == 0:
                if (
                    "additional_kwargs" in log
                    and "tool_calls" in log["additional_kwargs"]
                ):
                    tool_calls = log["additional_kwargs"]["tool_calls"]
                    if len(tool_calls) > 0:
                        tool_call = tool_calls[0]
                        if tool_call["name"] == tool_call_configs[counter]["name"]:
                            tool_call_logs[counter] = tool_call
                            if not tool_call_configs[counter]["verify_args"](
                                row, tool_call["args"]
                            ):
                                return f"{tool_call_configs[counter]['prefix']}_ARGS_MISMATCH"
            elif log["name"] == tool_call_configs[counter]["name"]:
                if log["content"].lower().startswith("error"):
                    return f"{tool_call_configs[counter]['prefix']}_ERROR"

                if not tool_call_configs[counter]["verify_result"](row, log["content"]):
                    return f"{tool_call_configs[counter]['prefix']}_RESULT_MISMATCH"
                tool_call_logs[counter] = log
                counter += 1

        for i in range(starting, len(tool_call_configs)):
            if len(tool_call_logs[i]) == 0:
                return f"{tool_call_configs[i]['prefix']}_NOT_INITIALIZED"
            elif len(tool_call_logs[i]) == 1:
                return f"{tool_call_configs[i]['prefix']}_NOT_CALLED"

        return "SUCCESS"
    except Exception as e:
        # print(log["content"])
        print(f"Error processing row: {e}")
        traceback.print_exc()
        return "ERROR"


bertscore = None


def calc_bert_f1(row):
    global bertscore
    if bertscore is None:
        from evaluate import load

        bertscore = load("bertscore")

    try:
        x = row["full_logs"]
        if x and isinstance(x, str):
            logs = json.loads(row["full_logs"])
            for log in logs:
                if log["name"] == "ocr_tool":
                    content = log.get("content", "")
                    if not content or not isinstance(content, str):
                        continue
                    invoice_id = get_invoice_id(row)
                    result = json.loads(content)["content"]
                    df_ground_truth_row = df_ground_truth[
                        df_ground_truth["invoice_id"] == int(invoice_id)
                    ].iloc[0]
                    predictions = [result]
                    references = [df_ground_truth_row["expected_ocr_result"]]
                    results = bertscore.compute(
                        predictions=predictions, references=references, lang="en"
                    )
                    return results["f1"][0]
    except Exception as e:
        print(f"Error calculating BERT F1: {e}")
        traceback.print_exc()

    return 0


def get_metrics(df, debug=False, including_df=False, including_bert_f1=True):
    df["original_process_status"] = df["process_status"].copy()
    df["process_steps"] = df["full_logs"].apply(
        lambda x: len(json.loads(x)) if x and isinstance(x, str) else 0
    )

    if including_bert_f1:
        df["ocr_bert_f1"] = df.apply(
            lambda x: calc_bert_f1(x) if x["attachments"] else 0,
            axis=1,
        )

    df["process_status"] = df.apply(
        update_process_status,
        axis=1,
    )

    if debug:
        print(f"Total number of tasks:\t\t{len(df)}")
        print(df["process_status"].value_counts())

    vc = df["process_status"].value_counts()
    completed = 1 - vc["NOT_STARTED"] / len(df) if "NOT_STARTED" in vc else 1

    if debug:
        print(f"Task completion rate:\t\t{completed * 100:.2f}%")

    # Calculate success rate
    success_rate = vc["SUCCESS"] / len(df) / completed if "SUCCESS" in vc else 0
    if debug:
        print(f"Task success rate:\t\t{success_rate * 100:.2f}%")

    df["end_time"] = pd.to_datetime(df["end_time"])
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()

    # Convert duration to HH:MM:SS format without milliseconds
    df["duration_hms"] = pd.to_timedelta(df["duration"], unit="s").dt.floor("s")

    if debug:
        print(f"Total execution time:\t\t{df['duration_hms'].sum()}")

    metrics = {
        "task_completion_rate": completed,
        "task_success_rate": success_rate,
        "total_execution_time": df["duration"].sum(),
        "mean_execution_time": df["duration"].mean(),
        "std_execution_time": df["duration"].std(),
        "mean_process_steps": df["process_steps"].mean(),
        "std_process_steps": df["process_steps"].std(),
        "total_tasks": len(df),
        **df["process_status"].value_counts().to_dict(),
    }
    if including_bert_f1:
        metrics["mean_ocr_bert_f1"] = df["ocr_bert_f1"].mean()
        metrics["std_ocr_bert_f1"] = df["ocr_bert_f1"].std()

    if "NOT_INVOICE" in metrics:
        metrics["NO_INVOICE"] = metrics["NOT_INVOICE"]
        del metrics["NOT_INVOICE"]

    if including_df:
        metrics["df"] = df

    # Estimate remaining time
    if "NOT_STARTED" in vc:
        df2 = df[df["process_status"] != "NOT_STARTED"]
        avg_duration = df2["duration"].mean()

        total_eval_time = pd.to_timedelta(avg_duration * len(df2), unit="s").floor("s")
        metrics["elapsed_time"] = total_eval_time

        remaining_tasks = vc["NOT_STARTED"]
        estimated_remaining_time = pd.to_timedelta(
            avg_duration * remaining_tasks, unit="s"
        ).floor("s")
        if debug:
            print(f"Estimated remaining time:\t{estimated_remaining_time}")
        metrics["estimated_remaining_time"] = estimated_remaining_time

    return metrics


def run_sql_query(db_filepath, query, debug=False):
    conn = sqlite3.connect(db_filepath)

    # Read the query results into a pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close the database connection
    conn.close()

    return df


def calculate_metrics(
    db_filepath,
    vision_only=False,
    including_df=True,
    including_bert_f1=True,
    debug=False,
):
    conn = sqlite3.connect(db_filepath)

    # Write your SQL query
    query = "SELECT * FROM emails"

    # Read the query results into a pandas DataFrame
    df = pd.read_sql(query, conn)

    # Close the database connection
    conn.close()

    if vision_only:
        df = df[df["attachments"] != ""].copy()

    return get_metrics(
        df, debug=debug, including_bert_f1=including_bert_f1, including_df=including_df
    )


def print_metrics(msg, metrics):
    print(msg)
    for key, value in metrics.items():
        print(f"\t{key}: {value}")
    print()


if __name__ == "__main__":
    db_filepath = "results/H100/llama3.2-vision_11b-qwen2.5_32b/emails.db"
    metrics = calculate_metrics(db_filepath, including_df=True, debug=True)
    df = metrics["df"]
    del metrics["df"]
    print_metrics("Full metrics:", metrics)

    df1 = df[df["attachments"] != ""].copy()
    metrics = get_metrics(df1, debug=True)
    print_metrics("Vision metrtics:", metrics)

    df1 = df[df["attachments"] == ""].copy()
    metrics = get_metrics(df1, debug=True)
    print_metrics("Non-vision metrtics:", metrics)
