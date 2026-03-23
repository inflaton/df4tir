import sys
import os
from metrics import *


def print_usage():
    print("Usage: python status.py <path_to_directory_containing_db>")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: No directory path provided.")
        print_usage()
        sys.exit(1)

    directory = sys.argv[1]
    db_filepath = os.path.join(directory, "emails.db")
    print(f"Calculating metrics for {db_filepath}...\n")

    metrics = calculate_metrics(db_filepath, including_df=True, including_bert_f1=False)
    df = metrics.pop("df", None)
    print_metrics("Full metrics:", metrics)

    df1 = df[df["attachments"] != ""].copy()
    metrics = get_metrics(df1, including_bert_f1=False)
    print_metrics("Vision metrtics:", metrics)

    df1 = df[df["attachments"] == ""].copy()
    metrics = get_metrics(df1, including_bert_f1=False)
    print_metrics("Non-vision metrtics:", metrics)

    if len(sys.argv) == 3:
        print("========================================================")
        invoice_id = int(sys.argv[2])
        db_filepath = os.path.join(directory, "transactions.db")
        query = f"SELECT * FROM transactions WHERE invoice_id = {invoice_id}"
        df_txn = run_sql_query(db_filepath, query)
        print_row_details(df_txn)
