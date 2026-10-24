import json
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_DATABASE = Path(__file__).resolve().parent / "jobs.sqlite3"
DATABASE_PATH = Path(
    os.getenv("RAG_JOB_DATABASE", str(DEFAULT_DATABASE))
).expanduser()
JOB_TTL_SECONDS = 30 * 60
MAX_JOBS = 500
LEASE_SECONDS = 60 * 60


def _connect():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def initialize_job_store():
    with closing(_connect()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL,
                question TEXT NOT NULL,
                refresh INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                error_status INTEGER,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS jobs_status_created "
            "ON jobs(status, created_at)"
        )
        connection.commit()


def enqueue_job(job_id, video_id, question, refresh):
    now = time.time()
    with closing(_connect()) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                job_id, video_id, question, refresh, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?)
            """,
            (job_id, video_id, question, int(refresh), now, now),
        )
        _prune(connection, now)
        connection.commit()


def claim_next_job():
    now = time.time()
    with closing(_connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT job_id, video_id, question, refresh
            FROM jobs
            WHERE status = 'queued'
               OR (status = 'processing' AND updated_at < ?)
            ORDER BY created_at
            LIMIT 1
            """,
            (now - LEASE_SECONDS,),
        ).fetchone()
        if row is None:
            connection.commit()
            return None

        connection.execute(
            "UPDATE jobs SET status = 'processing', updated_at = ? WHERE job_id = ?",
            (now, row["job_id"]),
        )
        connection.commit()
        return {
            "job_id": row["job_id"],
            "video_id": row["video_id"],
            "question": row["question"],
            "refresh": bool(row["refresh"]),
        }


def complete_job(job_id, result):
    with closing(_connect()) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'completed', result_json = ?, error = NULL,
                error_status = NULL, updated_at = ?
            WHERE job_id = ?
            """,
            (json.dumps(result), time.time(), job_id),
        )
        connection.commit()


def fail_job(job_id, message, error_status):
    with closing(_connect()) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'failed', error = ?, error_status = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (message, error_status, time.time(), job_id),
        )
        connection.commit()


def get_job(job_id):
    with closing(_connect()) as connection:
        _prune(connection, time.time())
        connection.commit()
        row = connection.execute(
            """
            SELECT job_id, status, result_json, error, error_status
            FROM jobs
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()

    if row is None:
        return None
    job = {"job_id": row["job_id"], "status": row["status"]}
    if row["status"] == "completed":
        job["result"] = json.loads(row["result_json"])
    elif row["status"] == "failed":
        job["error"] = row["error"]
        job["error_status"] = row["error_status"]
    return job


def _prune(connection, now):
    connection.execute(
        """
        DELETE FROM jobs
        WHERE status IN ('completed', 'failed') AND created_at < ?
        """,
        (now - JOB_TTL_SECONDS,),
    )
    connection.execute(
        """
        DELETE FROM jobs
        WHERE job_id IN (
            SELECT job_id FROM jobs
            WHERE status IN ('completed', 'failed')
            ORDER BY created_at DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (MAX_JOBS,),
    )
