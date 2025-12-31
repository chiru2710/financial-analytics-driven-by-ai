import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "financial_ai")
}

def init_db():
    conn = mysql.connector.connect(**DB_CONFIG)
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
    conn = mysql.connector.connect(**DB_CONFIG)
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
