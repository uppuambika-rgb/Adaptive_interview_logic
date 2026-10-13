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


def select_next_question(input_json):
    """
    Intelligently determines the next interview question and difficulty.
    Balances:
    1. Adaptive Depth: When a candidate answers correctly, escalates difficulty
       or asks follow-ups to probe deeper understanding.
    2. Comprehensive Resume Coverage: Prevents staying trapped in a single skill;
       tracks per-skill progress and transitions to cover all main skills.
    """
    try:
        if client is None:
            raise ValueError("GROQ_API_KEY not found in .env file.")

        data = json.loads(input_json)

        job_role = data.get("job_role", "Software Engineer")
        current_question = data.get("current_question", "")
        candidate_answer = data.get("candidate_answer", "")
        status = data.get("status", "correct")
        current_difficulty = data.get("current_difficulty", "easy").lower()

        # Support both singular skill and full skills list
        skills = data.get("skills")
        if not skills:
            single_skill = data.get("skill")
            skills = [single_skill] if single_skill else ["General Technical"]

        current_skill = data.get("current_skill") or data.get("skill") or skills[0]

        skill_stats = data.get("skill_stats", {})
        round_number = data.get("round_number", 1)
        total_rounds = data.get("total_rounds", 10)

        difficulties = ["easy", "medium", "hard"]

        if status not in [
            "correct",
            "partially_correct",
            "incorrect_or_unclear"
        ]:
            raise ValueError("Invalid answer status.")

        if current_difficulty not in difficulties:
            current_difficulty = "easy"

        curr_diff_idx = difficulties.index(current_difficulty)

        # 1. Calculate next difficulty for the current skill
        if status == "correct":
            next_diff_idx_curr = min(curr_diff_idx + 1, 2)
        elif status == "partially_correct":
            next_diff_idx_curr = curr_diff_idx
        else:
            next_diff_idx_curr = max(curr_diff_idx - 1, 0)

        updated_curr_difficulty = difficulties[next_diff_idx_curr]

        # 2. Check skill coverage and depth constraints
        rounds_remaining = max(0, total_rounds - round_number)

        # Unvisited skills (zero questions asked)
        unvisited_skills = [
            s for s in skills
            if skill_stats.get(s, {}).get("questions_asked", 0) == 0 and s != current_skill
        ]

        curr_consecutive = skill_stats.get(current_skill, {}).get("consecutive", 1)

        # Max consecutive depth limit before switching skills
        # Allows 1 harder follow-up if correct (depth 2), then rotates to ensure coverage
        max_consecutive_allowed = 2 if len(skills) > 1 else 999

        must_switch_skill = False
        decision_reason = ""

        # Condition A: Must switch to cover unvisited skills before rounds run out
        if len(unvisited_skills) >= rounds_remaining and len(unvisited_skills) > 0:
            must_switch_skill = True
            decision_reason = (
                f"Transitioning to assess unvisited resume skill to ensure 100% skill coverage."
            )

        # Condition B: Max consecutive depth reached for this skill
        elif curr_consecutive >= max_consecutive_allowed and len(skills) > 1:
            must_switch_skill = True
            decision_reason = (
                f"Completed in-depth assessment of '{current_skill}'. Rotating to cover next resume skill."
            )

        # Condition C: Struggled on this skill and other skills need assessment
        elif status == "incorrect_or_unclear" and len(skills) > 1 and len(unvisited_skills) > 0:
            must_switch_skill = True
            decision_reason = (
                f"Candidate struggled on '{current_skill}'. Switching to assess other resume competencies."
            )

        # 3. Handle Skill Transition vs. Skill Depth
        if must_switch_skill:
            action = "switch_skill"
            # Pick next skill: prioritize unvisited skills, else skill with fewest questions
            if unvisited_skills:
                next_skill = unvisited_skills[0]
            else:
                other_skills = [s for s in skills if s != current_skill]
                next_skill = min(
                    other_skills,
                    key=lambda s: skill_stats.get(s, {}).get("questions_asked", 0)
                )

            next_difficulty = skill_stats.get(next_skill, {}).get("difficulty", "easy")

            prompt = f"""
You are an expert technical interviewer.

Job role: {job_role}
Skill to assess: {next_skill}
Difficulty level: {next_difficulty}

Instruction:
Generate ONE technical interview question for this candidate on '{next_skill}' at '{next_difficulty}' difficulty.
Also identify 3 to 5 important concepts expected in a good answer.

Return ONLY valid JSON in this format:
{{
    "next_question": "The interview question",
    "difficulty": "{next_difficulty}",
    "expected_concepts": [
        "concept 1",
        "concept 2",
        "concept 3"
    ]
}}
"""

        else:
            # Stay on the same skill to test adaptive depth or clarify gaps
            next_skill = current_skill
            next_difficulty = updated_curr_difficulty

            if status == "correct":
                action = "increase_difficulty"
                instruction = (
                    f"The candidate answered correctly. Ask a harder, more in-depth question "
                    f"at '{next_difficulty}' difficulty on '{current_skill}' to test advanced mastery."
                )
                decision_reason = (
                    f"Candidate answered correctly. Escalating difficulty to '{next_difficulty}' "
                    f"to test deeper mastery in '{current_skill}'."
                )
            elif status == "partially_correct":
                action = "ask_follow_up"
                instruction = (
                    f"The candidate gave a partially correct answer. Ask a targeted follow-up question "
                    f"on '{current_skill}' at '{next_difficulty}' difficulty to clarify their partial understanding."
                )
                decision_reason = (
                    f"Candidate had minor gaps. Asking a targeted follow-up on '{current_skill}'."
                )
            else:
                action = "decrease_difficulty"
                instruction = (
                    f"The candidate struggled with the previous question. Ask a simpler, foundational question "
                    f"at '{next_difficulty}' difficulty on '{current_skill}'."
                )
                decision_reason = (
                    f"Candidate struggled. Lowering difficulty to '{next_difficulty}' for '{current_skill}'."
                )

            prompt = f"""
You are an expert technical interviewer.

Job role: {job_role}
Skill: {current_skill}

Previous question:
{current_question}

Candidate answer:
{candidate_answer}

Answer evaluation status: {status}
Previous difficulty: {current_difficulty}
Next difficulty: {next_difficulty}

Instruction:
{instruction}

Generate exactly ONE new question and 3 to 5 expected concepts for evaluating its answer.

Return ONLY valid JSON in this format:
{{
    "next_question": "The new interview question",
    "difficulty": "{next_difficulty}",
    "expected_concepts": [
        "concept 1",
        "concept 2",
        "concept 3"
    ]
}}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        result_raw = response.choices[0].message.content.strip()
        next_data = extract_json_from_response(result_raw)

        if not isinstance(next_data.get("next_question"), str) or not next_data["next_question"].strip():
            raise ValueError("Invalid next question returned.")

        concepts = next_data.get("expected_concepts")
        if not isinstance(concepts, list) or len(concepts) < 2:
            raise ValueError("Invalid expected concepts.")

        next_data["skill"] = next_skill
        next_data["difficulty"] = next_difficulty
        next_data["action"] = action
        next_data["reason"] = decision_reason

        return json.dumps(next_data, indent=4)

    except Exception as e:
        return json.dumps({
            "error": f"Adaptive question generation failed: {e}"
        }, indent=4)


if __name__ == "__main__":
    # Test 1: Single-skill adaptive depth test
    test_input = {
        "status": "correct",
        "current_difficulty": "easy",
        "current_question": "What is a Python function?",
        "candidate_answer": (
            "A function is a reusable block of code "
            "that can accept parameters and return a value."
        ),
        "job_role": "Python Developer",
        "skill": "Python Functions"
    }

    print("--- Single Skill Adaptive Test ---")
    print(select_next_question(json.dumps(test_input)))

    # Test 2: Multi-skill rotation test (triggering skill transition)
    test_multi = {
        "job_role": "Python Developer",
        "skills": ["Python Functions", "Data Structures", "SQL"],
        "current_skill": "Python Functions",
        "current_difficulty": "medium",
        "current_question": "Explain mutable default arguments in Python.",
        "candidate_answer": "They are evaluated once at definition time.",
        "status": "correct",
        "skill_stats": {
            "Python Functions": {"questions_asked": 2, "consecutive": 2, "difficulty": "medium"},
            "Data Structures": {"questions_asked": 0, "consecutive": 0, "difficulty": "easy"},
            "SQL": {"questions_asked": 0, "consecutive": 0, "difficulty": "easy"}
        },
        "round_number": 2,
        "total_rounds": 6
    }

    print("\n--- Multi-Skill Transition Test ---")
    print(select_next_question(json.dumps(test_multi)))