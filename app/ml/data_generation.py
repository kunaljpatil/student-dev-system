"""
Synthetic Data Generation Module (spec Section 6.4).

Generates a realistic 350-student dataset across 7 classes:
- Fixed random seed (2024) for reproducibility
- 3 latent traits per student: ability, motivation, engagement
- Realistic non-linear final outcome model (R2 ~0.75-0.85, not ~1.0)
- Classifier label generated independently from final outcome + attendance + trend
- 5 core subjects: Math, Science, English, Social Studies, Computer
- Two terms: Term 1 (complete with finals), Term 2 (current term without finals)
"""

import math
import random
from datetime import date, timedelta
from typing import List, Dict, Any

SEED = 2024
NUM_CLASSES = 7
STUDENTS_PER_CLASS = 50
TOTAL_STUDENTS = NUM_CLASSES * STUDENTS_PER_CLASS  # 350

SUBJECTS = [
    "Math",
    "Science",
    "English",
    "Social Studies",
    "Computer",
]

TERM1 = "Term 1"
TERM2 = "Term 2"

TERM1_MONTHS = ["2024-07", "2024-08", "2024-09", "2024-10"]
TERM2_MONTHS = ["2024-11", "2024-12", "2025-01", "2025-02"]

FIRST_NAMES = [
    "Aarav", "Sneha", "Rohan", "Ananya", "Ishaan", "Priya", "Vikram", "Diya",
    "Aditya", "Meera", "Kabir", "Tanvi", "Arjun", "Riya", "Karan", "Pooja",
    "Dhruv", "Neha", "Siddharth", "Kavya", "Rahul", "Nandini", "Varun", "Shruti",
    "Akash", "Anika", "Manish", "Divya", "Suresh", "Swati", "Nikhil", "Simran",
    "Gaurav", "Shreya", "Kunal", "Tara", "Ayush", "Bhavna", "Harsh", "Isha",
]

LAST_NAMES = [
    "Mehta", "Sharma", "Nair", "Patel", "Verma", "Iyer", "Rao", "Joshi",
    "Reddy", "Kulkarni", "Deshmukh", "Chopra", "Malhotra", "Bhat", "Menon",
    "Saxena", "Sen", "Pillai", "Das", "Bose", "Gupta", "Agarwal", "Bansal",
]

EXTRACURRICULAR_TEMPLATES = [
    ("Basketball", "sports", (4, 12)),
    ("Football", "sports", (4, 12)),
    ("Athletics", "sports", (3, 10)),
    ("Drawing & Painting", "arts", (2, 8)),
    ("Debate Club", "club", (2, 6)),
    ("Robotics Club", "club", (3, 10)),
    ("Classical Music", "music", (3, 8)),
    ("Band / Choir", "music", (2, 6)),
    ("Community Service", "volunteering", (2, 8)),
    ("Drama Club", "arts", (3, 8)),
    ("Coding Club", "club", (3, 10)),
]

COMPETITION_TEMPLATES = [
    ("Inter-School Math Olympiad", "district"),
    ("State Science Exhibition", "state"),
    ("National Coding Challenge", "national"),
    ("District Debate Championship", "district"),
    ("School Athletics Meet", "school"),
    ("State Level Art Competition", "state"),
    ("District Chess Tournament", "district"),
    ("National Spell Bee", "national"),
]


def clamp(val: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, val))


def round1(val: float) -> float:
    return round(val, 1)


def generate_dataset(seed: int = SEED) -> Dict[str, Any]:
    """Generates 350 students across 7 classes with 2 terms of data."""
    rng = random.Random(seed)

    students_data = []

    for s_idx in range(TOTAL_STUDENTS):
        class_idx = s_idx // STUDENTS_PER_CLASS
        roll_no = (s_idx % STUDENTS_PER_CLASS) + 1

        first_name = rng.choice(FIRST_NAMES)
        last_name = rng.choice(LAST_NAMES)
        full_name = f"{first_name} {last_name}"

        # Latent traits
        # Class-level slight offset to mimic real cohort differences
        class_ability_offset = (class_idx - 3) * 1.5
        ability = clamp(rng.gauss(55 + class_ability_offset, 14), 15, 95)
        motivation = clamp(rng.gauss(0.60, 0.18), 0.10, 1.0)
        engagement = clamp(rng.gauss(0.58, 0.20), 0.10, 1.0)

        # Subject affinity offsets (-10 to +10)
        subject_affinity = {
            subj: rng.gauss(0, 5) for subj in SUBJECTS
        }

        terms_data = []
        for term_num, term_name, months in [(1, TERM1, TERM1_MONTHS), (2, TERM2, TERM2_MONTHS)]:
            # Academic scores per subject
            exam_scores = []
            term_internals = []
            term_midterms = []
            subject_averages = {}

            for subj in SUBJECTS:
                aff = subject_affinity[subj]
                base_subj = clamp(ability + aff + (motivation - 0.5) * 10, 10, 98)

                # Internal 1
                i1 = clamp(rng.gauss(base_subj - 2, 6))
                # Internal 2 (with trend influenced by motivation)
                i2_trend = (motivation - 0.45) * 8 + rng.gauss(0, 4)
                i2 = clamp(base_subj + i2_trend)
                # Midterm
                mt = clamp(rng.gauss((i1 + i2) / 2 + 2, 5))

                exam_scores.append({"subject": subj, "exam_type": "internal1", "score": round1(i1)})
                exam_scores.append({"subject": subj, "exam_type": "internal2", "score": round1(i2)})
                exam_scores.append({"subject": subj, "exam_type": "midterm", "score": round1(mt)})

                term_internals.extend([i1, i2])
                term_midterms.append(mt)
                subject_averages[subj] = (i1 + i2 + mt) / 3

            avg_internal = sum(term_internals) / len(term_internals)
            avg_midterm = sum(term_midterms) / len(term_midterms)
            min_subj_score = min(subject_averages.values())

            # Attendance
            attendance_records = []
            att_pcts = []
            base_att = clamp(70 + engagement * 28 + (motivation - 0.5) * 10, 45, 99)

            for m_idx, month in enumerate(months):
                days_total = 22
                # slight drop in later months if motivation is low
                m_eff = base_att + (rng.gauss(0, 3) if m_idx < 2 else (motivation - 0.5) * 4 + rng.gauss(0, 3))
                pct = clamp(m_eff, 40, 100)
                days_present = int(round((pct / 100) * days_total))
                attendance_records.append({
                    "month": month,
                    "days_present": days_present,
                    "days_total": days_total,
                })
                att_pcts.append((days_present / days_total) * 100)

            avg_attendance = sum(att_pcts) / len(att_pcts)
            att_trend = (att_pcts[-1] + att_pcts[-2]) / 2 - (att_pcts[0] + att_pcts[1]) / 2

            # Assignments (2 per subject = 10)
            assignments = []
            asg_scores = []
            for subj in SUBJECTS:
                for asg_idx in (1, 2):
                    asg_score = clamp(ability + (engagement - 0.5) * 12 + rng.gauss(5, 7))
                    on_time = rng.random() < clamp(0.40 + engagement * 0.55 + motivation * 0.15, 0.2, 0.98)
                    assignments.append({
                        "subject": subj,
                        "title": f"{subj} Assignment {asg_idx}",
                        "score": round1(asg_score),
                        "submitted_on_time": on_time,
                    })
                    asg_scores.append(asg_score)

            avg_assignment = sum(asg_scores) / len(asg_scores)
            ontime_rate = sum(1 for a in assignments if a["submitted_on_time"]) / len(assignments)

            # Projects (1-2)
            projects = []
            num_projects = 1 if engagement < 0.4 else (2 if engagement > 0.75 else rng.choice([1, 2]))
            for p_idx in range(num_projects):
                is_lead = (ability > 65 and engagement > 0.6 and rng.random() < 0.45)
                p_score = clamp(ability + (engagement - 0.5) * 10 + rng.gauss(8, 6))
                projects.append({
                    "title": f"{'Advanced' if is_lead else 'Collaborative'} Term Project {p_idx + 1}",
                    "role": "lead" if is_lead else "member",
                    "score": round1(p_score),
                })

            # Extracurriculars (0 to 3)
            num_activities = 0 if engagement < 0.25 else (1 if engagement < 0.6 else rng.choice([1, 2, 3]))
            chosen_acts = rng.sample(EXTRACURRICULAR_TEMPLATES, min(num_activities, len(EXTRACURRICULAR_TEMPLATES)))
            activities = []
            total_act_hours = 0.0
            for act_name, act_cat, (min_h, max_h) in chosen_acts:
                hrs = round1(rng.uniform(min_h, max_h) * (0.8 + 0.4 * engagement))
                total_act_hours += hrs
                activities.append({
                    "activity_name": act_name,
                    "category": act_cat,
                    "hours_per_month": hrs,
                })

            # Competitions (0 to 2)
            competitions = []
            comp_chance = 0.20 + (ability / 100) * 0.35 + (engagement) * 0.25
            if rng.random() < comp_chance:
                comp_template, comp_level = rng.choice(COMPETITION_TEMPLATES)
                # Outcome depends on ability
                win_prob = (ability / 100) ** 1.8
                finalist_prob = (ability / 100) * 0.6
                r_val = rng.random()
                if r_val < win_prob:
                    result = "winner"
                elif r_val < finalist_prob:
                    result = "finalist"
                else:
                    result = "participated"

                comp_date = date(2024 if term_num == 1 else 2025, 8 if term_num == 1 else 1, rng.randint(5, 25))
                competitions.append({
                    "name": comp_template,
                    "level": comp_level,
                    "result": result,
                    "date": comp_date,
                })

            # Development indicators (1 to 5)
            def scale_ind(trait: float, noise_scale: float = 0.5) -> int:
                v = 1 + trait * 3.8 + rng.gauss(0, noise_scale)
                return max(1, min(5, int(round(v))))

            indicators = {
                "communication": scale_ind(motivation * 0.6 + engagement * 0.4),
                "presentation": scale_ind(ability * 0.007 + motivation * 0.4),
                "confidence": scale_ind(ability * 0.008 + motivation * 0.3),
                "participation": scale_ind(engagement),
                "teamwork": scale_ind((motivation + engagement) / 2),
            }

            # Composite engagement index (0-100)
            engagement_index = round1(
                100 * (
                    min(avg_attendance / 100, 1.0) * 0.35 +
                    ontime_rate * 0.25 +
                    (indicators["participation"] / 5.0) * 0.25 +
                    min(total_act_hours / 12.0, 1.0) * 0.15
                )
            )

            # Score trend between internals
            avg_i1 = sum(e["score"] for e in exam_scores if e["exam_type"] == "internal1") / len(SUBJECTS)
            avg_i2 = sum(e["score"] for e in exam_scores if e["exam_type"] == "internal2") / len(SUBJECTS)
            score_trend = round1(avg_i2 - avg_i1)

            # Term 1 Finals (the non-linear training target)
            final_overall = None
            risk_label = None

            if term_num == 1:
                # Spec non-linear formula
                hidden_exam_day_factor = rng.random()
                diminishing_engagement = 13.0 * math.sqrt(engagement) * (0.5 + 0.5 * avg_assignment / 100.0)
                attendance_penalty = 0.45 * (75.0 - avg_attendance) if avg_attendance < 75.0 else 0.0
                weak_drag = 5.0 if min_subj_score < 46.0 else 0.0
                noise = rng.gauss(0, 4.8)

                raw_final = (
                    5.0
                    + 0.46 * avg_internal
                    + 0.18 * avg_midterm
                    + 4.0 * motivation
                    + diminishing_engagement
                    - attendance_penalty
                    - weak_drag
                    + 4.0 * (indicators["confidence"] / 5.0 - 0.6)
                    + 3.0 * (hidden_exam_day_factor - 0.5)
                    + noise
                )
                final_overall = round1(clamp(raw_final, 10.0, 99.0))

                # Subject-wise finals for Term 1
                for subj in SUBJECTS:
                    s_avg = subject_averages[subj]
                    subj_final = clamp(final_overall + (s_avg - avg_internal) * 0.7 + rng.gauss(0, 3.5))
                    exam_scores.append({
                        "subject": subj,
                        "exam_type": "final",
                        "score": round1(subj_final),
                    })

                # Classifier ground-truth target (spec Section 6.3 - independent from regressor)
                if final_overall < 50.0 or avg_attendance < 75.0:
                    risk_label = "Needs Support"
                elif final_overall < 62.0 or score_trend <= -5.0 or engagement_index < 45.0:
                    risk_label = "Needs Monitoring"
                else:
                    risk_label = "On Track"

            terms_data.append({
                "term": term_name,
                "exam_scores": exam_scores,
                "attendance": attendance_records,
                "assignments": assignments,
                "projects": projects,
                "extracurriculars": activities,
                "competitions": competitions,
                "indicators": indicators,
                "final_overall": final_overall,
                "risk_label": risk_label,
                "avg_internal": round1(avg_internal),
                "avg_midterm": round1(avg_midterm),
                "avg_attendance": round1(avg_attendance),
                "engagement_index": engagement_index,
                "score_trend": score_trend,
            })

        students_data.append({
            "index": s_idx,
            "class_index": class_idx,
            "roll_no": roll_no,
            "full_name": full_name,
            "terms": terms_data,
        })

    return {
        "classes": [f"Class {i + 1}" for i in range(NUM_CLASSES)],
        "subjects": SUBJECTS,
        "students": students_data,
    }
