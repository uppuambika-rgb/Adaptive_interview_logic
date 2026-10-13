import os
import json
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq

from question_generation import generate_question_request
from answer_evaluation import evaluate_answer
from adaptive_questions import select_next_question
from skill_scoring import calculate_skill_score

# Load .env
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None


def format_progress_bar(percentage, width=15):
    """Generates an ASCII progress bar for terminal display safe on Windows cp1252."""
    filled = int(round(width * (percentage / 100.0)))
    filled = max(0, min(width, filled))
    return "#" * filled + "-" * (width - filled)


def generate_recruiter_follow_up_probes(job_role, experience_level, missing_concepts, weak_skills):
    """
    Generates targeted follow-up questions for the Hiring Manager / On-Site Interviewer
    based specifically on the candidate's gaps and unverified concepts.
    """
    if client is not None and (missing_concepts or weak_skills):
        try:
            prompt = f"""
You are an executive technical recruiter and interview strategist.

Based on this candidate's technical interview results, generate 2 to 3 targeted follow-up questions
for the Hiring Manager to ask in the next round. These questions should specifically investigate
the candidate's identified knowledge gaps or test deeper applied knowledge.

Candidate Role: {experience_level} {job_role}
Skills with Identified Gaps: {", ".join(weak_skills) if weak_skills else "General Core Skills"}
Identified Missing Concepts / Gaps:
{json.dumps(missing_concepts[:8])}

Return ONLY valid JSON in this format:
{{
    "probes": [
        {{
            "topic": "Skill or Topic Area",
            "question": "Specific technical question for the hiring manager to ask",
            "rationale": "Why the recruiter recommends asking this (e.g. probes specific gap in...)"
        }}
    ]
}}
"""
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            raw = response.choices[0].message.content.strip()
            if "```" in raw:
                first = raw.find("```")
                last = raw.rfind("```")
                if first != -1 and last != -1 and last > first:
                    block = raw[first:last]
                    lines = block.splitlines()
                    raw = "\n".join(lines[1:]).strip() if lines and lines[0].startswith("```") else block.strip()
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                parsed = json.loads(raw[start:end + 1])
                if isinstance(parsed.get("probes"), list) and parsed["probes"]:
                    return parsed["probes"]
        except Exception:
            pass

    # Fallback probes if LLM is unavailable or candidate had few gaps
    fallback = []
    if missing_concepts:
        for c in missing_concepts[:3]:
            fallback.append({
                "topic": "Knowledge Gap Probe",
                "question": f"Can you walk us through a real-world scenario where you had to implement or debug {c}?",
                "rationale": f"Candidate did not demonstrate {c} during the screening round."
            })
    else:
        fallback.append({
            "topic": "System & Architectural Depth",
            "question": f"How would you design and optimize a production service for {job_role} handling 10k requests per second?",
            "rationale": "Candidate showed strong fundamentals; next round should test high-scale system design."
        })
    return fallback


def generate_structured_report(job_role, skills, question_history, candidate_name="Candidate", experience_level="Mid-Level"):
    """
    Constructs a comprehensive, professional Recruiter Technical Assessment Report:
    - Candidate Profile Header
    - Executive Hiring Recommendation & Recruiter Takeaway
    - Core Competencies Scorecard
    - In-Depth Skill Assessments
    - Key Technical Strengths (with candidate quotes)
    - Risk Assessment & Technical Red Flags
    - Suggested Next-Round Interview Probes for the Hiring Manager
    - Full Chronological Interview Transcript & Audit Log
    """
    total_rounds = len(question_history)

    # 1. Calculate structured scores for every skill using skill_scoring
    skill_assessments = {}
    total_weighted_score = 0.0
    total_weight = 0.0
    total_correctness = 0.0
    total_completeness = 0.0

    diff_weights = {"easy": 1.0, "medium": 1.15, "hard": 1.30}

    for skill in skills:
        skill_questions = [
            q for q in question_history if q["skill"].strip().casefold() == skill.strip().casefold()
        ]

        scoring_input = {
            "skill": skill,
            "questions": skill_questions
        }

        skill_result = json.loads(calculate_skill_score(json.dumps(scoring_input)))
        skill_assessments[skill] = skill_result

    for q in question_history:
        d = q.get("difficulty", "easy").lower()
        w = diff_weights.get(d, 1.0)
        total_weighted_score += q.get("score", 0) * w
        total_weight += w
        total_correctness += q.get("correctness_score", 0)
        total_completeness += q.get("completeness_score", 0)

    overall_score = round(total_weighted_score / total_weight, 2) if total_weight > 0 else 0.0
    avg_correctness = round(total_correctness / total_rounds, 2) if total_rounds > 0 else 0.0
    avg_completeness = round(total_completeness / total_rounds, 2) if total_rounds > 0 else 0.0

    # Skills covered check
    skills_tested = [
        s for s in skills if skill_assessments.get(s, {}).get("total_questions", 0) > 0
    ]
    coverage_pct = round((len(skills_tested) / len(skills)) * 100, 1) if skills else 0.0

    # Overall hiring recommendation verdict
    if overall_score >= 85:
        verdict = "STRONG HIRE"
        verdict_summary = (
            f"{candidate_name} demonstrated exceptional technical proficiency, precise accuracy, and "
            f"broad conceptual depth across all assessed competencies. High-confidence recommendation to advance."
        )
    elif overall_score >= 70:
        verdict = "HIRE"
        verdict_summary = (
            f"{candidate_name} demonstrated solid foundational competency and practical problem-solving capability. "
            f"Has minor knowledge gaps that can be readily addressed on the job or probed in the team round."
        )
    elif overall_score >= 50:
        verdict = "CONSIDER WITH RESERVATIONS"
        verdict_summary = (
            f"{candidate_name} showed partial technical understanding, but exhibited noticeable conceptual gaps "
            f"and inconsistency when answering medium-difficulty questions. Recommended for a targeted deep-dive."
        )
    else:
        verdict = "DO NOT HIRE"
        verdict_summary = (
            f"{candidate_name} struggled with foundational concepts across multiple core requirements. "
            f"Does not meet the current technical bar for the {experience_level} {job_role} position."
        )

    # Collect strengths, evidence, missing concepts, and weak skills
    all_matched = []
    all_missing = []
    key_evidence = []
    weak_skills = []

    for s, s_data in skill_assessments.items():
        if s_data.get("score", 0) < 65:
            weak_skills.append(s)
        for c in s_data.get("matched_concepts", []):
            if c not in all_matched:
                all_matched.append(c)
        for c in s_data.get("missing_concepts", []):
            if c not in all_missing:
                all_missing.append(c)
        for ev in s_data.get("evidence", []):
            if ev not in key_evidence:
                key_evidence.append(ev)

    # Generate targeted follow-up probes for the hiring manager
    next_round_probes = generate_recruiter_follow_up_probes(
        job_role=job_role,
        experience_level=experience_level,
        missing_concepts=all_missing,
        weak_skills=weak_skills
    )

    # Structure complete dossier object
    report_data = {
        "candidate_profile": {
            "name": candidate_name,
            "target_role": job_role,
            "experience_level": experience_level,
            "interview_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_rounds_completed": total_rounds,
            "skill_coverage_percentage": coverage_pct,
            "skills_targeted": skills,
            "skills_tested": skills_tested
        },
        "executive_summary": {
            "overall_score": overall_score,
            "hiring_verdict": verdict,
            "recruiter_takeaway": verdict_summary,
            "average_correctness": avg_correctness,
            "average_completeness": avg_completeness
        },
        "skill_assessments": skill_assessments,
        "key_strengths": all_matched[:6],
        "key_evidence": key_evidence[:4],
        "risk_assessment_red_flags": all_missing[:6],
        "next_round_interview_probes": next_round_probes,
        "question_audit_log": question_history
    }

    # =========================================================================
    # RENDER TERMINAL RECRUITER REPORT
    # =========================================================================
    print("\n" + "=" * 80)
    print("           RECRUITER TECHNICAL ASSESSMENT & CANDIDATE DOSSIER")
    print("=" * 80)

    print("\n[ 1. CANDIDATE PROFILE ]")
    print("-" * 80)
    print(f"Candidate Name    : {candidate_name}")
    print(f"Target Position   : {experience_level} {job_role}")
    print(f"Interview Date    : {report_data['candidate_profile']['interview_date']}")
    print(f"Rounds Completed  : {total_rounds} of 10 (100% Complete)")
    print(f"Skills Evaluated  : {', '.join(skills)} (Coverage: {coverage_pct}%)")

    print("\n[ 2. EXECUTIVE HIRING RECOMMENDATION ]")
    print("-" * 80)
    print(f"HIRING VERDICT    : [ {verdict} ]")
    print(f"Candidate Score   : {overall_score:.1f} / 100  [{format_progress_bar(overall_score)}] ({overall_score:.1f}%)")
    print(f"Technical Accuracy: {avg_correctness:.1f}%")
    print(f"Concept Depth     : {avg_completeness:.1f}%")
    print(f"\nRecruiter Takeaway:\n  \"{verdict_summary}\"")

    print("\n[ 3. CORE COMPETENCIES SCORECARD ]")
    print("-" * 80)
    print(f"{'Skill':<22} | {'Questions':<9} | {'Score':<7} | {'Level':<16} | {'Coverage':<8} | {'Recruiter Rating'}")
    print("-" * 80)

    for skill in skills:
        s_data = skill_assessments.get(skill, {})
        q_count = s_data.get("total_questions", 0)
        s_score = s_data.get("score", 0)
        s_level = s_data.get("level", "Unassessed")
        cov = s_data.get("concept_coverage_percentage", 0.0)

        if s_score >= 80:
            rating_tag = "Recommended (High)"
        elif s_score >= 65:
            rating_tag = "Competent (Acceptable)"
        elif s_score >= 50:
            rating_tag = "Marginal (Needs Probe)"
        elif q_count > 0:
            rating_tag = "Below Bar"
        else:
            rating_tag = "Unassessed"

        print(
            f"{skill:<22} | {q_count:<9} | {s_score:>3}/100 | {s_level:<16} | "
            f"{cov:>5.1f}%  | {rating_tag}"
        )

    print("-" * 80)

    print("\n[ 4. KEY TECHNICAL STRENGTHS ]")
    print("-" * 80)
    if all_matched:
        for idx, item in enumerate(all_matched[:5], 1):
            print(f"  {idx}. Mastered Concept: {item}")
    if key_evidence:
        print("\nDirect Candidate Quotes / Evidence:")
        for ev in key_evidence[:3]:
            print(f"  * \"{ev}\"")

    print("\n[ 5. RISK ASSESSMENT & TECHNICAL RED FLAGS ]")
    print("-" * 80)
    if all_missing:
        for idx, item in enumerate(all_missing[:5], 1):
            print(f"  [!] Identified Gap: {item} (Candidate failed to demonstrate or clarify)")
    else:
        print("  None detected: Candidate successfully addressed all expected technical concepts.")

    print("\n[ 6. SUGGESTED NEXT-ROUND INTERVIEW PROBES (FOR HIRING MANAGER) ]")
    print("-" * 80)
    for idx, probe in enumerate(next_round_probes, 1):
        print(f"\nProbe #{idx} - Focus Area: {probe.get('topic')}")
        print(f"  Question  : {probe.get('question')}")
        print(f"  Rationale : {probe.get('rationale')}")

    print("\n[ 7. FULL INTERVIEW TRANSCRIPT & QUESTION AUDIT LOG ]")
    print("-" * 80)
    for q in question_history:
        r_num = q.get("round", 0)
        sk = q.get("skill", "")
        df = q.get("difficulty", "")
        sc = q.get("score", 0)
        corr = q.get("correctness_score", 0)
        comp = q.get("completeness_score", 0)
        st = q.get("status", "")

        print(f"\nRound {r_num} [{sk} - {df.upper()}] | Result: {st} | Score: {sc}/100 (Accuracy: {corr}%, Depth: {comp}%)")
        print(f"  Interviewer : {q.get('question', '')}")
        ans_preview = q.get('candidate_answer', '').strip()
        print(f"  Candidate   : {ans_preview}")
        m_con = q.get("matched_concepts", [])
        if m_con:
            print(f"  Demonstrated: {', '.join(m_con)}")
        ms_con = q.get("missing_concepts", [])
        if ms_con:
            print(f"  Gaps / Miss : {', '.join(ms_con)}")

    print("\n" + "=" * 80)
    print("                    END OF RECRUITER DOSSIER")
    print("=" * 80 + "\n")

    # Save reports to workspace files
    try:
        report_json_path = os.path.join(os.path.dirname(__file__), "interview_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=4)
        print(f"[Recruiter Dossier Saved] JSON report saved to: {os.path.abspath(report_json_path)}")

        report_md_path = os.path.join(os.path.dirname(__file__), "interview_report.md")
        md_content = generate_markdown_report(report_data)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"[Recruiter Dossier Saved] Markdown report saved to: {os.path.abspath(report_md_path)}")

    except Exception as e:
        print(f"Notice: Could not save report files: {e}")

    return report_data


def generate_markdown_report(report_data):
    """Generates a professional recruiter assessment document in Markdown."""
    cand = report_data["candidate_profile"]
    exec_sum = report_data["executive_summary"]
    skills = cand["skills_targeted"]
    assessments = report_data["skill_assessments"]
    probes = report_data.get("next_round_interview_probes", [])

    md = f"""# Recruiter Technical Assessment Dossier

**Candidate Name:** {cand['name']}  
**Target Position:** {cand['experience_level']} {cand['target_role']}  
**Interview Date:** {cand['interview_date']}  
**Rounds Completed:** {cand['total_rounds_completed']} of 10 (100%)  
**Skill Coverage:** {cand['skill_coverage_percentage']}%  

---

## 1. Executive Hiring Recommendation

### Verdict: **{exec_sum['hiring_verdict']}**
- **Overall Match Score:** {exec_sum['overall_score']} / 100
- **Technical Accuracy:** {exec_sum['average_correctness']}%
- **Concept Depth:** {exec_sum['average_completeness']}%

> **Recruiter Takeaway:**  
> {exec_sum['recruiter_takeaway']}

---

## 2. Core Competencies Scorecard

| Skill | Questions Attempted | Score | Proficiency Level | Concept Coverage |
| :--- | :---: | :---: | :--- | :---: |
"""
    for s in skills:
        s_data = assessments.get(s, {})
        md += f"| {s} | {s_data.get('total_questions', 0)} | {s_data.get('score', 0)}/100 | {s_data.get('level', 'Unassessed')} | {s_data.get('concept_coverage_percentage', 0)}% |\n"

    md += "\n---\n\n## 3. Key Technical Strengths\n"
    strengths = report_data.get("key_strengths", [])
    if strengths:
        for s in strengths:
            md += f"- **Demonstrated Mastery:** {s}\n"

    evidence = report_data.get("key_evidence", [])
    if evidence:
        md += "\n### Verified Candidate Quotes\n"
        for ev in evidence:
            md += f"> \"{ev}\"\n\n"

    md += "---\n\n## 4. Risk Assessment & Technical Red Flags\n"
    red_flags = report_data.get("risk_assessment_red_flags", [])
    if red_flags:
        for rf in red_flags:
            md += f"- ⚠️ **Knowledge Gap:** {rf}\n"
    else:
        md += "- No major technical red flags detected.\n"

    md += "\n---\n\n## 5. Suggested Next-Round Interview Probes (For Hiring Manager)\n"
    for idx, p in enumerate(probes, 1):
        md += f"\n### Probe #{idx}: {p.get('topic')}\n"
        md += f"- **Question to Ask:** {p.get('question')}\n"
        md += f"- **Recruiter Rationale:** {p.get('rationale')}\n"

    md += "\n---\n\n## 6. Full Chronological Interview Transcript\n"
    for q in report_data.get("question_audit_log", []):
        md += f"\n### Round {q.get('round')}: {q.get('skill')} ({q.get('difficulty')})\n"
        md += f"- **Question:** {q.get('question')}\n"
        md += f"- **Candidate Response:** {q.get('candidate_answer')}\n"
        md += f"- **Evaluation Status:** {q.get('status')} | **Score:** {q.get('score')}/100 (Accuracy: {q.get('correctness_score')}%, Depth: {q.get('completeness_score')}%)\n"
        md += f"- **Evaluation Note:** {q.get('explanation')}\n"

    return md


def main():
    print("=" * 65)
    print("      AI TECHNICAL INTERVIEW PLATFORM - RECRUITER SUITE")
    print("=" * 65)

    candidate_name = input("\nEnter Candidate Name: ").strip() or "Candidate"
    experience_level = input("Enter Experience Level (e.g. Junior, Mid-Level, Senior): ").strip() or "Mid-Level"
    job_role = input("Enter Target Job Role: ").strip()
    skills_input = input("Enter Target Skills (separated by commas): ").strip()

    skills = [
        item.strip()
        for item in skills_input.split(",")
        if item.strip()
    ]

    if not job_role or not skills:
        print("Error: Target Job Role and at least one Skill are required.")
        return

    total_rounds = 10

    # Initialize per-skill tracking state
    skill_stats = {
        skill: {
            "questions_asked": 0,
            "consecutive": 0,
            "difficulty": "easy"
        }
        for skill in skills
    }

    current_skill = skills[0]
    next_question_data = None
    question_history = []

    print("\n" + "=" * 65)
    print("               TECHNICAL INTERVIEW SESSION: 10 ROUNDS")
    print("=" * 65)
    print(f"Candidate       : {candidate_name}")
    print(f"Target Role     : {experience_level} {job_role}")
    print(f"Target Skills   : {', '.join(skills)}")
    print("Total Rounds    : 10")
    print("Instructions    : Please answer each question thoroughly.")
    print("                  Your responses will be recorded and evaluated")
    print("                  in the final Recruiter Technical Dossier.")
    print("=" * 65)

    for round_number in range(1, total_rounds + 1):
        # 1. Obtain current question (either from adaptive transition or initial generation)
        if next_question_data is not None:
            question = next_question_data["next_question"]
            selected_skill = next_question_data["skill"]
            difficulty = next_question_data["difficulty"]
            expected_concepts = next_question_data["expected_concepts"]
            next_question_data = None
        else:
            selected_skill = current_skill
            difficulty = skill_stats[selected_skill]["difficulty"]

            try:
                question_result = generate_question_request(
                    json.dumps({
                        "job_role": job_role,
                        "skills": [selected_skill],
                        "difficulty": difficulty
                    })
                )
                question_data = json.loads(question_result)

                if "error" in question_data:
                    print("Question generation error:", question_data["error"])
                    break

                question = question_data["question"]
                expected_concepts = question_data["expected_concepts"]

            except Exception as e:
                print("Question generation failed:", e)
                break

        print(f"\n[ ROUND {round_number} OF {total_rounds} ] - {selected_skill.upper()} ({difficulty.upper()})")
        print("-" * 65)
        print(f"Interviewer:\n{question}\n")

        # 2. Get candidate's answer
        candidate_answer = input("Your answer: ").strip()

        # 3. Evaluate answer in the background
        try:
            evaluation_result = evaluate_answer(
                json.dumps({
                    "question": question,
                    "candidate_answer": candidate_answer,
                    "expected_concepts": expected_concepts
                })
            )

            evaluation = json.loads(evaluation_result)

            if "error" in evaluation:
                print("Evaluation error:", evaluation["error"])
                break

            status = evaluation["status"]
            score = evaluation["score"]
            correctness_score = evaluation["correctness_score"]
            completeness_score = evaluation["completeness_score"]

        except Exception as e:
            print("Answer evaluation failed:", e)
            break

        # In realistic interview mode: Clean confirmation without revealing grading rubrics
        print("\n✓ Response recorded.")

        # 4. Document this question into the structured history for the recruiter
        record = {
            "round": round_number,
            "skill": selected_skill,
            "difficulty": difficulty,
            "question": question,
            "candidate_answer": candidate_answer,
            "expected_concepts": expected_concepts,
            "matched_concepts": evaluation["matched_concepts"],
            "missing_concepts": evaluation["missing_concepts"],
            "evidence": evaluation.get("evidence", []),
            "correctness_score": correctness_score,
            "completeness_score": completeness_score,
            "score": score,
            "status": status,
            "explanation": evaluation["explanation"]
        }
        question_history.append(record)

        # 5. Update per-skill stats
        skill_stats[selected_skill]["questions_asked"] += 1
        skill_stats[selected_skill]["consecutive"] += 1

        for s in skills:
            if s != selected_skill:
                skill_stats[s]["consecutive"] = 0

        # Update difficulty for the skill adaptively
        difficulties = ["easy", "medium", "hard"]
        d_idx = difficulties.index(difficulty)
        if status == "correct":
            skill_stats[selected_skill]["difficulty"] = difficulties[min(d_idx + 1, 2)]
        elif status == "incorrect_or_unclear":
            skill_stats[selected_skill]["difficulty"] = difficulties[max(d_idx - 1, 0)]

        # 6. Adaptively plan next question if rounds remain
        if round_number < total_rounds:
            try:
                adaptive_payload = {
                    "job_role": job_role,
                    "skills": skills,
                    "current_skill": selected_skill,
                    "current_question": question,
                    "candidate_answer": candidate_answer,
                    "status": status,
                    "current_difficulty": difficulty,
                    "skill_stats": skill_stats,
                    "round_number": round_number,
                    "total_rounds": total_rounds
                }

                adaptive_result = select_next_question(json.dumps(adaptive_payload))
                adaptive_data = json.loads(adaptive_result)

                if "error" in adaptive_data:
                    print("Adaptive planning error:", adaptive_data["error"])
                    break

                next_question_data = adaptive_data
                current_skill = adaptive_data["skill"]

            except Exception as e:
                print("Adaptive planning failed:", e)
                break

    # 7. Generate and display the professional Recruiter Technical Dossier
    if question_history:
        print("\n" + "=" * 65)
        print("    INTERVIEW COMPLETED - COMPILING RECRUITER DOSSIER")
        print("=" * 65)
        generate_structured_report(
            job_role=job_role,
            skills=skills,
            question_history=question_history,
            candidate_name=candidate_name,
            experience_level=experience_level
        )
    else:
        print("\nNo questions completed. Cannot generate report.")


if __name__ == "__main__":
    main()