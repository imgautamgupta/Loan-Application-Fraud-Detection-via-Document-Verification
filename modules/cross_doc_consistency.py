# Cross-document consistency checks (salary slips vs. bank deposits)
import sqlite3
import pandas as pd

def cross_doc_consistency(salary_db_path, bank_db_path):
    # Example: compare salary slip amounts to bank deposits
    salary_conn = sqlite3.connect(salary_db_path)
    bank_conn = sqlite3.connect(bank_db_path)
    salary_df = pd.read_sql_query('SELECT * FROM salary_slips', salary_conn)
    bank_df = pd.read_sql_query('SELECT * FROM bank_deposits', bank_conn)
    # Simple check: match by employee_id and month
    merged = pd.merge(salary_df, bank_df, on=['employee_id', 'month'], suffixes=('_slip', '_bank'))
    merged['amount_match'] = merged['amount_slip'] == merged['amount_bank']
    return merged[['employee_id', 'month', 'amount_slip', 'amount_bank', 'amount_match']]

# Usage:
# cross_doc_consistency('salary.db', 'bank.db')
