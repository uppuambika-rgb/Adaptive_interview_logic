import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load .env from script directory or current working directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def extract_json_from_response(text):
    """Safely extracts JSON from model text even if wrapped in markdown code blocks."""
    text = text.strip()
    if "```" in text:
        first = text.find("```")
        last = text.rfind("```")
        if first != -1 and last != -1 and last > first:
            block = text[first:last]
            lines = block.splitlines()
            if lines and lines[0].startswith("```"):
                text = "\n".join(lines[1:]).strip()
            else:
                text = block.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx:end_idx + 1]
    return json.loads(text)


def evaluate_answer(input_json):
    """
    Evaluates candidate's answer in a structured format:
    1. Technical Correctness (0-100) assessed objectively against technical accuracy.
    2. Concept Completeness (0-100) calculated deterministically from matched concepts.
    3. Composite Question Score (0-100) combining correctness (60%) and completeness (40%).
    4. Deterministic status classification (correct, partially_correct, incorrect_or_unclear).
    """
    try:
        if client is None:
            raise ValueError("GROQ_API_KEY not found in .env file.")

        data = json.loads(input_json)

        question = data["question"]
        candidate_answer = data["candidate_answer"]
        expected_concepts = data["expected_concepts"]

        if not isinstance(candidate_answer, str):
            raise ValueError("Candidate answer must be text.")

        if not isinstance(expected_concepts, list) or not expected_concepts:
            raise ValueError("Expected concepts must be a non-empty list.")

        # Handle empty or whitespace answer
        if not candidate_answer.strip():
            return json.dumps({
                "status": "incorrect_or_unclear",
                "score": 0,
                "correctness_score": 0,
                "completeness_score": 0,
                "explanation": "No answer was provided by the candidate.",
                "expected_concepts": expected_concepts,
                "matched_concepts": [],
                "missing_concepts": expected_concepts,
                "evidence": []
            }, indent=4)

        prompt = f"""
You are an expert technical interviewer and evaluator.

Evaluate the candidate's answer fairly and objectively against the expected concepts.

Question:
{question}

Candidate's answer:
{candidate_answer}

Expected concepts:
{json.dumps(expected_concepts)}

Evaluation Instructions:
1. Correctness Assessment (0 to 100):
   Evaluate technical accuracy, reasoning validity, and factual correctness.
   - 90-100: Completely accurate and technically sound explanation without misconceptions.
   - 70-89: Mostly accurate with minor omissions or slight imprecision.
   - 40-69: Partially accurate, contains noticeable misconceptions or vague statements.
   - 0-39: Fundamentally incorrect, irrelevant, or unclear.

2. Concept Matching:
   - "matched_concepts": ONLY concepts from the expected concepts list that the candidate genuinely demonstrated in their answer.
   - "missing_concepts": concepts from the expected concepts list that the candidate omitted or failed to demonstrate.

3. Evidence:
   - Provide direct short quotes from the candidate's answer demonstrating their understanding.

4. Explanation:
   - A concise technical explanation of the evaluation.

Return ONLY valid JSON in this exact structure:
{{
    "correctness_score": 85,
    "matched_concepts": ["concept from expected concepts"],
    "missing_concepts": ["concept from expected concepts"],
    "evidence": ["direct quote from answer"],
    "explanation": "Concise justification."
}}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        result_raw = response.choices[0].message.content.strip()
        evaluation = extract_json_from_response(result_raw)

        # 1. Parse and validate correctness score
        correctness = evaluation.get("correctness_score")
        if not isinstance(correctness, (int, float)) or isinstance(correctness, bool):
            correctness = evaluation.get("score", 50)
        correctness = max(0, min(100, round(float(correctness))))

        # 2. Strict matching against expected concepts
        matched_raw = evaluation.get("matched_concepts", [])
        if not isinstance(matched_raw, list):
            matched_raw = []

        matched_keys = {
            item.strip().casefold()
            for item in matched_raw
            if isinstance(item, str)
        }

        matched = [
            concept for concept in expected_concepts
            if concept.strip().casefold() in matched_keys
        ]

        missing = [
            concept for concept in expected_concepts
            if concept.strip().casefold() not in matched_keys
        ]

        # 3. Calculate completeness score systematically from concept match ratio
        if len(expected_concepts) > 0:
            completeness = round((len(matched) / len(expected_concepts)) * 100)
        else:
            completeness = 0

        # 4. Calculate structured combined question score (60% correctness, 40% completeness)
        combined_score = round((0.60 * correctness) + (0.40 * completeness))

        # 5. Deterministic status classification
        if combined_score >= 75 and correctness >= 70:
            status = "correct"
        elif combined_score >= 40 or correctness >= 40 or completeness >= 40:
            status = "partially_correct"
        else:
            status = "incorrect_or_unclear"

        evidence = evaluation.get("evidence", [])
        if not isinstance(evidence, list):
            evidence = []

        explanation = evaluation.get("explanation", "").strip()
        if not explanation:
            explanation = (
                f"Candidate demonstrated {len(matched)} of {len(expected_concepts)} concepts "
                f"with {correctness}% technical correctness."
            )

        structured_eval = {
            "status": status,
            "score": combined_score,
            "correctness_score": correctness,
            "completeness_score": completeness,
            "explanation": explanation,
            "expected_concepts": expected_concepts,
            "matched_concepts": matched,
            "missing_concepts": missing,
            "evidence": [item for item in evidence if isinstance(item, str) and item.strip()]
        }

        return json.dumps(structured_eval, indent=4)

    except Exception as e:
        return json.dumps({
            "error": f"Answer evaluation failed: {e}"
        }, indent=4)


if __name__ == "__main__":
    test_input = {
        "question": "What is a Python function?",
        "candidate_answer": (
            "A function is a reusable block of code. "
            "It can accept parameters and return a value."
        ),
        "expected_concepts": [
            "Reusable block of code",
            "Parameters",
            "Return values"
        ]
    }

    print("Answer Evaluation:")
    print(evaluate_answer(json.dumps(test_input)))