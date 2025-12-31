import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 3306))
}

def get_connection():
    if not DB_CONFIG["host"]:
        return None
    return mysql.connector.connect(**DB_CONFIG)

def init_db():
    conn = get_connection()
    if not conn:
        print("⚠️ No database configured — skipping DB init")
        return

    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_name VARCHAR(255),
            csv_file LONGBLOB,
            rows_count INT,
            columns_count INT,
            target VARCHAR(255),
            features TEXT,
            accuracy FLOAT,
            trained_on DATETIME
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def save_training_record(file_name, csv_bytes, rows, columns, target, features, accuracy):
    conn = get_connection()
    if not conn:
        print("⚠️ No database configured — skipping save")
        return

    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO training_history
        (file_name, csv_file, rows_count, columns_count, target, features, accuracy, trained_on)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        file_name,
        csv_bytes,
        rows,
        columns,
        target,
        features,
        accuracy,
        datetime.now()
    ))
    conn.commit()
    cursor.close()
    conn.close()
