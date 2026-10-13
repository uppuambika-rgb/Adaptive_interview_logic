import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load .env from script directory or current working directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=api_key)


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


def generate_question_request(input_json):
    data = json.loads(input_json)

    job_role = data.get("job_role", "")
    skills = data.get("skills")

    if not skills and "skill" in data:
        skills = [data["skill"]]

    difficulty = data.get("difficulty", "easy").lower()

    if not isinstance(skills, list) or not skills:
        raise ValueError("Skills must be a non-empty list.")

    if difficulty not in ["easy", "medium", "hard"]:
        raise ValueError("Difficulty must be easy, medium, or hard.")

    selected_skill = skills[0] if len(skills) == 1 else ", ".join(skills)

    prompt = f"""
You are an expert technical interviewer.

Generate ONE technical interview question.

Job role: {job_role}
Skill to assess: {selected_skill}
Difficulty: {difficulty}

Also identify 3 to 5 important concepts expected in a good answer to that question.

Return ONLY valid JSON in this format:
{{
    "job_role": "{job_role}",
    "skill": "{skills[0] if len(skills) == 1 else 'the selected skill'}",
    "difficulty": "{difficulty}",
    "question": "the interview question",
    "expected_concepts": [
        "concept 1",
        "concept 2",
        "concept 3"
    ]
}}

Rules:
- Select one skill from the provided skills.
- The question must match the requested difficulty.
- Expected concepts must be relevant to the question.
- Keep the question clear and specific.
- Return JSON only, without Markdown.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    result_raw = response.choices[0].message.content.strip()
    question_data = extract_json_from_response(result_raw)

    required_fields = [
        "job_role",
        "skill",
        "difficulty",
        "question",
        "expected_concepts"
    ]

    for field in required_fields:
        if field not in question_data:
            raise ValueError(f"Missing field: {field}")

    if not isinstance(question_data["expected_concepts"], list):
        raise ValueError("Expected concepts must be a list.")

    if not (2 <= len(question_data["expected_concepts"]) <= 6):
        raise ValueError("AI must generate 2 to 6 expected concepts.")

    return json.dumps(question_data, indent=4)


if __name__ == "__main__":
    test_input = {
        "job_role": "Python Developer",
        "skills": ["Python Functions", "Data Structures"],
        "difficulty": "easy"
    }

    result = generate_question_request(json.dumps(test_input))

    print("AI-generated interview question:")
    print(result)