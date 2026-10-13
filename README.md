# Adaptive Interview Intelligence — Recruiter Evaluation Platform

## Project Overview

**Adaptive Interview Intelligence** is an AI-powered technical interview and candidate evaluation platform. It simulates realistic, professional technical interviews across 10 rounds, adaptively testing skill depth and coverage, while silently compiling an executive-grade **Recruiter Technical Assessment Dossier** for recruiters and hiring managers.

---

## Key Features & Architecture

### 1. Realistic Interview Experience
- **Candidate-Facing Flow:** The candidate is briefed on the 10-round interview session and asked technical questions naturally.
- **Clean Response Acknowledgment:** Candidate responses are confirmed cleanly (`✓ Response recorded.`) without disrupting the interview or exposing internal test rubrics, weights, or scores mid-session.
- **Dynamic Adaptive Questioning:** In the background, the interviewer balances deep probing (escalating from Easy → Medium → Hard when correct) with full resume skill coverage so all target competencies are evaluated.

### 2. Recruiter Technical Assessment Dossier
At the conclusion of the 10 rounds, the system compiles a complete recruiter dossier:
- **Candidate Profile Header:** Candidate name, target position, experience level (Junior, Mid-Level, Senior), timestamp, and skill coverage rate.
- **Executive Hiring Recommendation:** Clear hiring verdict (`STRONG HIRE`, `HIRE`, `CONSIDER WITH RESERVATIONS`, `DO NOT HIRE`), overall candidate fit score, and a concise 2-sentence Recruiter Decision Takeaway.
- **Core Competencies Scorecard:** Tabular scorecard comparing questions attempted, scores, proficiency tiers (`Advanced`, `Proficient`, `Competent`, `Needs Improvement`), concept coverage %, and recruiter rating.
- **Key Technical Strengths:** Itemized list of mastered concepts substantiated by direct candidate answer quotes as verified evidence.
- **Risk Assessment & Technical Red Flags:** Clear itemization of unverified concepts, misconceptions, or areas where the candidate struggled.
- **Suggested Next-Round Interview Probes (For Hiring Manager):** 2 to 3 tailored follow-up technical questions designed specifically for the on-site/hiring manager round to probe the candidate's identified gaps.
- **Full Interview Transcript & Question Audit Log:** Complete chronological transcript of all 10 rounds with question text, candidate responses, correctness %, completeness %, and score.
- **Dual Export:** Automatically saves the complete dossier to `interview_report.json` and a formatted GitHub-style `interview_report.md`.

### 3. Structured Scoring Engine
- **Technical Correctness (0–100):** Objective assessment of technical accuracy, logic, and reasoning.
- **Concept Completeness (0–100):** Deterministic ratio of matched expected concepts:
  $$\text{Completeness} = \frac{|\text{Matched Concepts}|}{|\text{Expected Concepts}|} \times 100$$
- **Composite Question Score:**
  $$\text{Score} = 0.60 \times \text{Correctness} + 0.40 \times \text{Completeness}$$
- **Difficulty-Weighted Skill Scores:** Easy ($1.0\times$), Medium ($1.15\times$), Hard ($1.30\times$).

---

## Project Files

- `main.py` — Orchestrates the interview loop, manages candidate interaction, and renders/saves the Recruiter Dossier.
- `question_generation.py` — Generates targeted technical questions and expected concepts using LLM.
- `answer_evaluation.py` — Evaluates candidate answers, extracting evidence, matched concepts, missing concepts, and computing structured correctness/completeness scores.
- `adaptive_questions.py` — Decides adaptive difficulty escalations, follow-ups, and skill transitions to ensure full resume skill coverage.
- `skill_scoring.py` — Aggregates question audit history into structured skill scores and proficiency metrics.
- `requirements.txt` — Required dependencies (`groq`, `python-dotenv`).
- `.env` — Secure storage for `GROQ_API_KEY`.

---

## Setup & Running Instructions

1. **Install Python** (Python 3.10+ recommended).

2. **Install dependencies:**
   ```powershell
   py -m pip install -r requirements.txt
   ```

3. **Configure your API Key:**
   Create a `.env` file in the project folder with your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the interview:**
   ```powershell
   py main.py
   ```