import unittest
import time
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from Server import job_store
from Server.main import app


class AskApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path_patch = patch.object(
            job_store,
            "DATABASE_PATH",
            Path(self.temp_dir.name) / "jobs.sqlite3",
        )
        self.path_patch.start()
        self.client_context = TestClient(app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_invalid_video_url_is_rejected(self):
        response = self.client.post(
            "/ask",
            json={"video_url": "https://example.com/not-youtube", "question": "What is it about?"},
        )

        self.assertEqual(response.status_code, 400)

    def test_request_returns_job_and_completed_result(self):
        expected_result = {
            "video_id": "RE2c_eH0Sq4",
            "answer": "It is a test [00:00:01].",
            "citations": [],
        }
        with patch("Server.controllers.handle_question", return_value=expected_result):
            response = self.client.post(
                "/ask",
                json={
                    "video_url": "https://youtu.be/RE2c_eH0Sq4",
                    "question": "What is this about?",
                },
            )
            self.assertEqual(response.status_code, 202)
            job = response.json()
            self.assertEqual(job["status"], "queued")

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status_response = self.client.get(job["status_url"])
                self.assertEqual(status_response.status_code, 200)
                job_status = status_response.json()
                if job_status["status"] == "completed":
                    break
                time.sleep(0.05)

            self.assertEqual(job_status["status"], "completed")
            self.assertEqual(job_status["result"], expected_result)


if __name__ == "__main__":
    unittest.main()
