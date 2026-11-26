
import os
import json
import csv
from pathlib import Path

from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
)

from src.rag_pipeline import answer_question_with_rag
from evals.groq_judge import GroqJudge


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GOLDEN_PATH = PROJECT_ROOT / "Data/annotations/questions.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs/evaluation"

THRESHOLD = 0.7
TOP_K = 5

DATASET_TYPE = os.getenv("RAG_DATASET_TYPE", "ece").lower()
JUDGE_MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD QUESTIONS
# --------------------------------------------------

def load_questions():
    questions = []

    with open(GOLDEN_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                item = json.loads(line)

                questions.append({
                    "question_id": item.get("question_id", f"Q{len(questions)+1}"),
                    "question": item.get(
                        "question",
                        item.get("query", item.get("input", ""))
                    ),
                    "category": item.get("category", "unknown"),
                })

    return questions


# --------------------------------------------------
# 3. SELECT COLLECTION
# --------------------------------------------------

def get_collection():
    from data_preprocessing.query import (
        legal_collection,
        ece_collection,
    )

    if DATASET_TYPE == "legal":
        return legal_collection

    if DATASET_TYPE == "ece":
        return ece_collection

    raise ValueError(
        f"Unknown RAG_DATASET_TYPE: {DATASET_TYPE}. Use ece or legal."
    )


# --------------------------------------------------
# 4. INITIALIZE GROQ JUDGE
# --------------------------------------------------

def get_metrics():
    judge = GroqJudge(
        model_name=JUDGE_MODEL_NAME,
        temperature=0,
    )

    return [
        ContextualRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=THRESHOLD,
            model=judge,
            include_reason=True,
        ),
    ]


# --------------------------------------------------
# 5. SAVE RESULTS
# --------------------------------------------------

def save_results(results):
    json_path = OUTPUT_DIR / "rag_pipeline_results.json"
    csv_path = OUTPUT_DIR / "rag_pipeline_results.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(results[0].keys())
            )
            writer.writeheader()
            writer.writerows(results)

    print(f"\nResults saved to: {json_path}")
    print(f"Results saved to: {csv_path}")


# --------------------------------------------------
# 6. PRINT SUMMARY
# --------------------------------------------------

def print_summary(results):
    print("\n" + "=" * 65)
    print("RAG EVALUATION SUMMARY")
    print("=" * 65)

    metric_names = [
        "contextual_relevancy",
        "faithfulness",
        "answer_relevancy",
    ]

    for metric_name in metric_names:
        scores = [
            row[f"{metric_name}_score"]
            for row in results
            if row.get(f"{metric_name}_score") is not None
        ]

        if not scores:
            print(f"\n{metric_name}: No successful scores")
            continue

        average_score = sum(scores) / len(scores)
        passed = sum(score >= THRESHOLD for score in scores)
        pass_rate = (passed / len(scores)) * 100

        print(f"\n{metric_name.replace('_', ' ').title()}")
        print(f"Average Score : {average_score:.4f}")
        print(f"Passed        : {passed}/{len(scores)}")
        print(f"Pass Rate     : {pass_rate:.2f}%")

    print("\n" + "=" * 65)


# --------------------------------------------------
# 7. MAIN EVALUATION FUNCTION
# --------------------------------------------------

def run_evaluation():
    questions = load_questions()

    # TEMPORARY: evaluate only one question.
    # Remove this line after the smoke test succeeds.
    questions = questions[:1]

    if not questions:
        raise ValueError("No questions found in the evaluation dataset.")

    collection = get_collection()
    metrics = get_metrics()

    results = []
    total = len(questions)

    print("\nStarting RAG evaluation...")
    print(f"Total questions: {total}")
    print(f"Top K: {TOP_K}")
    print("Generator: Ollama (llama3.2)")
    print(f"Judge: Groq ({JUDGE_MODEL_NAME})")

    for index, item in enumerate(questions, start=1):
        question = item["question"]

        print("\n" + "-" * 65)
        print(f"Question {index}/{total}")
        print(f"ID: {item['question_id']}")
        print(f"Question: {question}")

        rag_result = answer_question_with_rag(
            question=question,
            collection=collection,
            top_k=TOP_K,
        )

        answer = rag_result["answer"]
        context = rag_result.get("context")

        if context is None:
            raise KeyError(
                "RAG pipeline did not return context. "
                "Check src/rag_pipeline.py."
            )

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            retrieval_context=context,
        )

        row = {
            "question_id": item["question_id"],
            "question": question,
            "category": item["category"],
            "answer": answer,
            "retrieved_chunks": len(context),
        }

        for metric in metrics:
            metric_name = metric.__class__.__name__.replace("Metric", "")
            metric_name = "".join(
                "_" + char.lower() if char.isupper() else char
                for char in metric_name
            ).lstrip("_")

            print(f"\nEvaluating {metric_name} using Groq...")

            try:
                metric.measure(test_case)

                score = metric.score
                reason = metric.reason

                row[f"{metric_name}_score"] = score
                row[f"{metric_name}_reason"] = reason

                print(f"Score: {score}")
                print(f"Reason: {reason}")

            except Exception as exc:
                print(f"Evaluation failed: {exc}")
                row[f"{metric_name}_score"] = None
                row[f"{metric_name}_reason"] = str(exc)

        results.append(row)
        save_results(results)

    print_summary(results)
    return results


# --------------------------------------------------
# 8. EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    run_evaluation()
