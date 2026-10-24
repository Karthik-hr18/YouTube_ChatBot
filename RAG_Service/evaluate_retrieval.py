import argparse
import json
from pathlib import Path

from RAG_Service.lang_chain import (
    get_vector_store,
    get_video_overview_docs,
    is_broad_question,
)


def evaluate_dataset(dataset_path):
    cases = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("The evaluation dataset has no cases.")

    vector_store = get_vector_store()
    hits = 0
    total_expected = 0
    total_found = 0

    for index, case in enumerate(cases, start=1):
        video_id = case["video_id"]
        question = case["question"]
        expected_timestamps = set(case["expected_timestamps"])
        if not expected_timestamps:
            raise ValueError(f"Evaluation case {index} has no expected timestamps.")

        if is_broad_question(question):
            documents = get_video_overview_docs(vector_store, video_id)
        else:
            documents = vector_store.similarity_search(
                question,
                k=4,
                filter={"video_id": video_id},
            )
        retrieved_text = "\n".join(document.page_content for document in documents)
        found = {
            timestamp
            for timestamp in expected_timestamps
            if f"[{timestamp}]" in retrieved_text
        }
        case_hit = bool(found)
        hits += int(case_hit)
        total_expected += len(expected_timestamps)
        total_found += len(found)

        print(
            f"{index}. {'HIT' if case_hit else 'MISS'} "
            f"video={video_id} expected={sorted(expected_timestamps)} "
            f"found={sorted(found)}"
        )

    hit_rate = hits / len(cases)
    recall_at_k = total_found / total_expected
    print(f"\nHit rate@4/overview: {hit_rate:.1%} ({hits}/{len(cases)})")
    print(f"Expected-timestamp recall: {recall_at_k:.1%} ({total_found}/{total_expected})")
    return {"hit_rate": hit_rate, "timestamp_recall": recall_at_k}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate whether retrieval finds expected transcript timestamps."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="tests/retrieval_eval.json",
        help="JSON dataset with video_id, question, expected_timestamps",
    )
    arguments = parser.parse_args()
    evaluate_dataset(arguments.dataset)


if __name__ == "__main__":
    main()
