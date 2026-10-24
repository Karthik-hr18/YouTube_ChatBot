import unittest
from types import SimpleNamespace

from RAG_Service.lang_chain import (
    ANSWER_WITH_CITATIONS,
    attach_validated_citations,
)


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.documents = [
            SimpleNamespace(
                metadata={
                    "source": "https://www.youtube.com/watch?v=RE2c_eH0Sq4",
                    "timestamp_start": "00:00:04",
                    "timestamp_end": "00:00:09",
                    "timestamp_start_seconds": 4,
                }
            )
        ]

    def test_valid_chunk_reference_becomes_a_timestamp_link(self):
        answer, citations = attach_validated_citations(
            "The speaker introduces the topic [S1].",
            self.documents,
        )

        self.assertEqual(answer, "The speaker introduces the topic [00:00:04].")
        self.assertEqual(citations[0]["url"], "https://www.youtube.com/watch?v=RE2c_eH0Sq4&t=4s")

    def test_unknown_or_missing_references_are_rejected(self):
        for answer_text in (
            "Unsupported claim [S2].",
            "Uncited factual claim.",
            "Supported first claim [S1]. Unsupported second claim.",
        ):
            with self.subTest(answer=answer_text):
                answer, citations = attach_validated_citations(
                    answer_text,
                    self.documents,
                )
                self.assertEqual(answer, ANSWER_WITH_CITATIONS)
                self.assertEqual(citations, [])


if __name__ == "__main__":
    unittest.main()
