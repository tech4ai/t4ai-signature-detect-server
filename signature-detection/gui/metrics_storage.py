import os
import sqlite3

DATABASE_DIR = os.path.join(os.getcwd(), "signature-detection", "gui", "db")
DATABASE_PATH = os.path.join(DATABASE_DIR, "metrics.db")


class MetricsStorage:
    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path
        self.setup_database()

    def setup_database(self):
        """Initialize the SQLite database and create tables if they don't exist"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS inference_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inference_time REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.commit()

    def add_metric(self, inference_time):
        """Add a new inference time measurement to the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO inference_metrics (inference_time) VALUES (?)",
                (inference_time,),
            )
            conn.commit()

    def get_recent_metrics(self, limit=80):
        """Get the most recent metrics from the database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT inference_time FROM inference_metrics ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            )
            results = cursor.fetchall()
            return [r[0] for r in reversed(results)]

    def get_total_inferences(self):
        """Get the total number of inferences recorded"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM inference_metrics")
            return cursor.fetchone()[0]

    def get_average_time(self, limit=80):
        """Get the average inference time from the most recent entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT AVG(inference_time) FROM (SELECT inference_time FROM inference_metrics ORDER BY timestamp DESC LIMIT ?)",
                (limit,),
            )
            result = cursor.fetchone()[0]
            return result if result is not None else 0
