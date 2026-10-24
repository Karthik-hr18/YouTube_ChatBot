import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from langchain_core.runnables import RunnableLambda

from RAG_Service.lang_chain import (
    answer_question,
    get_video_overview_docs,
    is_broad_question,
    split_independent_questions,
)


class QuestionHandlingTests(unittest.TestCase):
    def test_splits_multiple_questions_and_coordinated_parts(self):
        self.assertEqual(
            split_independent_questions(
                "What is the video about? Who is speaking? And when?"
            ),
            ["What is the video about", "Who is speaking", "when"],
        )
        self.assertEqual(
            split_independent_questions(
                "What is the main idea and who explains it?"
            ),
            ["What is the main idea", "who explains it"],
        )

    def test_detects_broad_video_questions(self):
        self.assertTrue(is_broad_question("What is this video about?"))
        self.assertTrue(is_broad_question("Give me an overview of the video"))
        self.assertFalse(is_broad_question("When is the first example shown?"))

    def test_overview_samples_chunks_across_the_full_video(self):
        store = Mock()
        store.get.return_value = {
            "documents": [f"chunk {index}" for index in range(20)],
            "metadatas": [{"chunk_index": index} for index in range(20)],
        }

        documents = get_video_overview_docs(store, "video-id", limit=5)

        self.assertEqual(len(documents), 5)
        self.assertEqual(
            [doc.metadata["chunk_index"] for doc in documents],
            [0, 5, 10, 14, 19],
        )
        store.get.assert_called_once()

    def test_each_question_is_retrieved_independently(self):
        document = SimpleNamespace(
            page_content="[00:00:01] Evidence from the transcript.",
            metadata={
                "source": "https://www.youtube.com/watch?v=RE2c_eH0Sq4",
                "timestamp_start": "00:00:01",
                "timestamp_end": "00:00:02",
                "timestamp_start_seconds": 1,
                "chunk_index": 0,
            },
        )
        store = Mock()
        store.get.return_value = {"ids": ["chunk-id"]}
        store.similarity_search.return_value = [document]

        with (
            patch("RAG_Service.lang_chain.get_vector_store", return_value=store),
            patch(
                "RAG_Service.lang_chain.ChatGoogleGenerativeAI",
                return_value=RunnableLambda(lambda _: "Supported fact [S1]."),
            ),
        ):
            result = answer_question(
                "RE2c_eH0Sq4",
                "What is the topic? Who explains it?",
            )

        self.assertEqual(store.similarity_search.call_count, 2)
        self.assertEqual(len(result["answers"]), 2)
        self.assertIn("[00:00:01]", result["answer"])


if __name__ == "__main__":
    unittest.main()
