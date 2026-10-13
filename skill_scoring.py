import json


def calculate_skill_score(input_json):
    """
    Calculates a structured skill score based on a comprehensive document/record
    of questions evaluated for a specific skill.

    Supports both:
    1. Multi-question document mode (primary):
       {"skill": "...", "questions": [ {question_data}, ... ]}
    2. Single-question / legacy mode:
       {"skill": "...", "expected_concepts": [...], "matched_concepts": [...], "evidence": [...]}
    """
    data = json.loads(input_json)

    skill = data.get("skill")
    if not isinstance(skill, str) or not skill.strip():
        raise ValueError("A valid skill name is required.")

    skill = skill.strip()
    questions = data.get("questions")

    # If questions list is not provided, convert legacy single-question format
    if questions is None:
        expected_concepts = data.get("expected_concepts")
        matched_concepts = data.get("matched_concepts", [])
        evidence = data.get("evidence", [])
        correctness_score = data.get("correctness_score")
        difficulty = data.get("difficulty", "easy")

        if not isinstance(expected_concepts, list) or not expected_concepts:
            raise ValueError("Expected concepts are required.")

        if not isinstance(matched_concepts, list):
            raise ValueError("Matched concepts must be a list.")

        # Determine completeness
        matched_lookup = {c.strip().casefold() for c in matched_concepts if isinstance(c, str)}
        matched = [c for c in expected_concepts if c.strip().casefold() in matched_lookup]
        missing = [c for c in expected_concepts if c.strip().casefold() not in matched_lookup]

        comp_score = round(len(matched) / len(expected_concepts) * 100) if expected_concepts else 0
        corr_score = correctness_score if isinstance(correctness_score, (int, float)) else comp_score
        q_score = round(0.60 * corr_score + 0.40 * comp_score)

        questions = [
            {
                "question": data.get("question", f"Question on {skill}"),
                "difficulty": difficulty,
                "status": "correct" if q_score >= 75 else ("partially_correct" if q_score >= 40 else "incorrect_or_unclear"),
                "correctness_score": corr_score,
                "completeness_score": comp_score,
                "score": q_score,
                "expected_concepts": expected_concepts,
                "matched_concepts": matched,
                "missing_concepts": missing,
                "evidence": evidence
            }
        ]

    if not isinstance(questions, list):
        raise ValueError("Questions must be a list of evaluated question objects.")

    if len(questions) == 0:
        return json.dumps({
            "skill": skill,
            "score": 0,
            "level": "Unassessed",
            "total_questions": 0,
            "difficulty_breakdown": {"easy": 0, "medium": 0, "hard": 0},
            "average_correctness": 0.0,
            "average_completeness": 0.0,
            "concept_coverage_percentage": 0.0,
            "explanation": f"No questions were attempted for {skill}.",
            "evidence": [],
            "matched_concepts": [],
            "missing_concepts": [],
            "questions_audit": []
        }, indent=4)

    # Difficulty weight multipliers
    difficulty_weights = {
        "easy": 1.0,
        "medium": 1.15,
        "hard": 1.30
    }

    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    weighted_score_sum = 0.0
    total_weights = 0.0

    total_correctness = 0.0
    total_completeness = 0.0

    all_expected = []
    all_matched = []
    all_evidence = []
    questions_audit = []

    for q in questions:
        diff = str(q.get("difficulty", "easy")).lower()
        if diff not in difficulty_weights:
            diff = "easy"
        difficulty_counts[diff] += 1

        weight = difficulty_weights[diff]
        score = float(q.get("score", 0))

        weighted_score_sum += score * weight
        total_weights += weight

        corr = float(q.get("correctness_score", score))
        comp = float(q.get("completeness_score", 0))

        total_correctness += corr
        total_completeness += comp

        exp_c = q.get("expected_concepts", [])
        mat_c = q.get("matched_concepts", [])
        mis_c = q.get("missing_concepts", [])
        ev = q.get("evidence", [])

        for c in exp_c:
            if isinstance(c, str) and c.strip() and c not in all_expected:
                all_expected.append(c)

        for c in mat_c:
            if isinstance(c, str) and c.strip() and c not in all_matched:
                all_matched.append(c)

        for e in ev:
            if isinstance(e, str) and e.strip() and e.strip() not in all_evidence:
                all_evidence.append(e.strip())

        questions_audit.append({
            "question": q.get("question", ""),
            "difficulty": diff,
            "status": q.get("status", "unknown"),
            "correctness_score": round(corr, 1),
            "completeness_score": round(comp, 1),
            "score": round(score, 1),
            "matched_concepts": mat_c,
            "missing_concepts": mis_c
        })

    num_questions = len(questions)
    avg_correctness = round(total_correctness / num_questions, 2)
    avg_completeness = round(total_completeness / num_questions, 2)

    # Base difficulty-weighted score
    final_score = round(min(100.0, weighted_score_sum / total_weights)) if total_weights > 0 else 0

    # Identified persistent gaps: concepts that were never demonstrated across any question
    matched_casefold = {c.casefold() for c in all_matched}
    unresolved_gaps = [c for c in all_expected if c.casefold() not in matched_casefold]

    # Concept coverage percentage
    concept_coverage = (
        round((len(all_matched) / len(all_expected)) * 100, 2)
        if all_expected else 0.0
    )

    # Level classification
    if final_score >= 85:
        level = "Advanced"
        rating_desc = "deep and comprehensive"
    elif final_score >= 70:
        level = "Proficient"
        rating_desc = "solid and proficient"
    elif final_score >= 50:
        level = "Average"
        rating_desc = "moderate with noticeable knowledge gaps"
    else:
        level = "Needs improvement"
        rating_desc = "limited and requires foundational study"

    explanation = (
        f"The candidate demonstrated a {rating_desc} understanding of {skill} "
        f"with an average correctness of {avg_correctness}%, concept completeness of "
        f"{avg_completeness}%, and concept coverage of {concept_coverage}% across "
        f"{num_questions} question(s)."
    )

    result = {
        "skill": skill,
        "score": final_score,
        "level": level,
        "total_questions": num_questions,
        "difficulty_breakdown": difficulty_counts,
        "average_correctness": avg_correctness,
        "average_completeness": avg_completeness,
        "concept_coverage_percentage": concept_coverage,
        "explanation": explanation,
        "evidence": all_evidence,
        "matched_concepts": all_matched,
        "missing_concepts": unresolved_gaps,
        "questions_audit": questions_audit
    }

    return json.dumps(result, indent=4)


if __name__ == "__main__":
    # Test 1: Legacy format
    legacy_input = {
        "skill": "Python Functions",
        "expected_concepts": [
            "Reusable block of code",
            "Parameters",
            "Return values"
        ],
        "matched_concepts": [
            "Reusable block of code",
            "Parameters"
        ],
        "evidence": [
            "A function is a reusable block of code.",
            "It can accept parameters."
        ]
    }
    print("--- Legacy Format Test ---")
    print(calculate_skill_score(json.dumps(legacy_input)))

    # Test 2: Structured Multi-Question Document format
    structured_input = {
        "skill": "Python Functions",
        "questions": [
            {
                "question": "What is a function in Python?",
                "difficulty": "easy",
                "status": "correct",
                "correctness_score": 95,
                "completeness_score": 100,
                "score": 97,
                "expected_concepts": ["Reusable block", "Parameters", "Return values"],
                "matched_concepts": ["Reusable block", "Parameters", "Return values"],
                "missing_concepts": [],
                "evidence": ["A function is a reusable block..."]
            },
            {
                "question": "Explain default mutable arguments in Python.",
                "difficulty": "medium",
                "status": "partially_correct",
                "correctness_score": 75,
                "completeness_score": 67,
                "score": 72,
                "expected_concepts": ["definition time binding", "shared list problem", "None sentinel"],
                "matched_concepts": ["definition time binding", "shared list problem"],
                "missing_concepts": ["None sentinel"],
                "evidence": ["Mutable defaults share state across calls..."]
            }
        ]
    }
    print("\n--- Structured Multi-Question Document Test ---")
    print(calculate_skill_score(json.dumps(structured_input)))