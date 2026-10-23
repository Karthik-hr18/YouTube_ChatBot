import unittest
from unittest.mock import patch

from RAG_Service.ingest_transcript import ingest_video


class FakeVectorStore:
    def __init__(self):
        self.records = {}
        self.deleted = []

    def get(self, where, include):
        records = [
            (chunk_id, document.metadata)
            for chunk_id, document in self.records.items()
            if document.metadata["video_id"] == where["video_id"]
        ]
        return {
            "ids": [chunk_id for chunk_id, _ in records],
            "metadatas": [metadata for _, metadata in records],
        }

    def add_documents(self, documents, ids):
        self.records.update(zip(ids, documents))

    def delete(self, ids):
        self.deleted.extend(ids)
        for chunk_id in ids:
            self.records.pop(chunk_id, None)


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.video_id = "RE2c_eH0Sq4"
        self.store = FakeVectorStore()
        self.transcripts = [
            "[00:00:01] First version.\n[00:00:04] More first version.",
            "[00:00:02] Updated version.\n[00:00:05] More updated version.",
        ]

    def test_new_video_is_indexed_and_repeat_is_skipped(self):
        with (
            patch(
                "RAG_Service.ingest_transcript.get_vector_store",
                return_value=self.store,
            ),
            patch(
                "RAG_Service.ingest_transcript.get_transcript",
                side_effect=self.transcripts,
            ) as get_transcript,
        ):
            first = ingest_video(self.video_id)
            second = ingest_video(self.video_id)

        self.assertTrue(first["indexed"])
        self.assertFalse(second["indexed"])
        self.assertEqual(get_transcript.call_count, 1)
        self.assertTrue(self.store.records)
        for document in self.store.records.values():
            self.assertIn("timestamp_start_seconds", document.metadata)
            self.assertIn("index_hash", document.metadata)

    def test_refresh_replaces_old_chunks_only_after_indexing(self):
        with (
            patch(
                "RAG_Service.ingest_transcript.get_vector_store",
                return_value=self.store,
            ),
            patch(
                "RAG_Service.ingest_transcript.get_transcript",
                side_effect=self.transcripts,
            ),
        ):
            ingest_video(self.video_id)
            old_ids = set(self.store.records)
            result = ingest_video(self.video_id, refresh=True)

        self.assertTrue(result["indexed"])
        self.assertTrue(self.store.deleted)
        self.assertFalse(old_ids.intersection(self.store.records))
        self.assertTrue(self.store.records)


if __name__ == "__main__":
    unittest.main()
