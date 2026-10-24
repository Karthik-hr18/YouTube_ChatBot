import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Server import job_store


class JobStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_dir.name) / "jobs.sqlite3"
        self.path_patch = patch.object(
            job_store,
            "DATABASE_PATH",
            self.database_path,
        )
        self.path_patch.start()
        job_store.initialize_job_store()

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_job_is_persisted_claimed_and_completed(self):
        job_store.enqueue_job(
            "job-1",
            "RE2c_eH0Sq4",
            "What is this about?",
            False,
        )

        claimed = job_store.claim_next_job()
        self.assertEqual(claimed["job_id"], "job-1")
        self.assertEqual(job_store.get_job("job-1")["status"], "processing")

        result = {"answer": "A cited answer.", "citations": []}
        job_store.complete_job("job-1", result)
        self.assertEqual(job_store.get_job("job-1")["result"], result)
        self.assertIsNone(job_store.claim_next_job())

    def test_failed_jobs_return_error_information(self):
        job_store.enqueue_job("job-2", "RE2c_eH0Sq4", "Question?", False)
        job_store.claim_next_job()
        job_store.fail_job("job-2", "Transcript unavailable", 502)

        self.assertEqual(
            job_store.get_job("job-2"),
            {
                "job_id": "job-2",
                "status": "failed",
                "error": "Transcript unavailable",
                "error_status": 502,
            },
        )


if __name__ == "__main__":
    unittest.main()
