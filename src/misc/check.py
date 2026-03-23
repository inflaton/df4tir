######## This code is to get the excel outputs for dbs
import sqlite3
import pandas as pd

# Define the database URL and output Excel file path
DB_URL = "src/data/db/gpt-4o-gpt-4o/emails.db"
OUTPUT_EXCEL_FILE = "emails_output.xlsx"

# Connect to the SQLite database
conn = sqlite3.connect(DB_URL)

# Read the SQL query results into a pandas DataFrame
query = "SELECT * FROM emails"
df = pd.read_sql_query(query, conn)

# Write the DataFrame to an Excel file
df.to_excel(OUTPUT_EXCEL_FILE, index=False, engine="openpyxl")

print(f"Data exported successfully to {OUTPUT_EXCEL_FILE}")

# Close the database connection
conn.close()
