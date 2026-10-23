import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from Server.main import app


class AskApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

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

        status_response = self.client.get(job["status_url"])
        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["status"], "completed")
        self.assertEqual(status_response.json()["result"], expected_result)


if __name__ == "__main__":
    unittest.main()
