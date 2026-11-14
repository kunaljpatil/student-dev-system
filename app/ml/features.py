"""
Feature Engineering Module (spec Section 6.2).

Extracts 24 features per student per term:
- Academic: avg internal, avg midterm, score trend, min subject score, subject std
- Attendance: attendance %, attendance trend
- Assignments: avg assignment score, on-time rate
- Projects: count, avg score, lead roles
- Extracurriculars: count, total hours, category diversity
- Competitions: count, highest level, wins
- Development indicators: communication, presentation, confidence, participation, teamwork (1-5)
- Composite: engagement_index (0-100)

NO sensitive attributes (gender, religion, caste, income) are used anywhere.
"""

import math
import numpy as np
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.db.models import (
    ExamScore,
    Attendance,
    Assignment,
    Project,
    Extracurricular,
    Competition,
    DevelopmentIndicator,
)

FEATURE_NAMES = [
    "avg_internal_score",
    "avg_midterm_score",
    "score_trend",
    "min_subject_score",
    "subject_score_std",
    "attendance_pct",
    "attendance_trend",
    "avg_assignment_score",
    "ontime_rate",
    "project_count",
    "avg_project_score",
    "lead_roles",
    "activity_count",
    "activity_hours",
    "activity_diversity",
    "competition_count",
    "highest_competition_level",
    "competition_wins",
    "communication",
    "presentation",
    "confidence",
    "participation",
    "teamwork",
    "engagement_index",
]

CLUSTER_FEATURE_NAMES = [
    "avg_internal_score",
    "avg_midterm_score",
    "attendance_pct",
    "avg_assignment_score",
    "ontime_rate",
    "activity_hours",
    "activity_diversity",
    "communication",
    "presentation",
    "confidence",
    "participation",
    "teamwork",
    "engagement_index",
]

FEATURE_LABELS = {
    "avg_internal_score": "Internal Average",
    "avg_midterm_score": "Midterm Average",
    "score_trend": "Score Trend",
    "min_subject_score": "Weakest Subject",
    "subject_score_std": "Subject Spread",
    "attendance_pct": "Attendance %",
    "attendance_trend": "Attendance Trend",
    "avg_assignment_score": "Assignment Average",
    "ontime_rate": "On-Time Submissions",
    "project_count": "Projects Count",
    "avg_project_score": "Project Score",
    "lead_roles": "Lead Roles",
    "activity_count": "Activities Count",
    "activity_hours": "Activity Hours/mo",
    "activity_diversity": "Activity Diversity",
    "competition_count": "Competitions Count",
    "highest_competition_level": "Highest Level Reached",
    "competition_wins": "Competition Wins",
    "communication": "Communication",
    "presentation": "Presentation",
    "confidence": "Confidence",
    "participation": "Participation",
    "teamwork": "Teamwork",
    "engagement_index": "Engagement Index",
}

LEVEL_ORDER = {
    "school": 1,
    "district": 2,
    "state": 3,
    "national": 4,
}


def round2(val: float) -> float:
    return round(float(val), 2)


def extract_features_from_dict(term_data: Dict[str, Any]) -> List[float]:
    """Extracts the 24-feature vector from synthetic generator student term dictionary."""
    exam_scores = term_data.get("exam_scores", [])
    i1_scores = [s["score"] for s in exam_scores if s["exam_type"] == "internal1"]
    i2_scores = [s["score"] for s in exam_scores if s["exam_type"] == "internal2"]
    mt_scores = [s["score"] for s in exam_scores if s["exam_type"] == "midterm"]

    avg_internal = float(np.mean(i1_scores + i2_scores)) if (i1_scores or i2_scores) else 60.0
    avg_midterm = float(np.mean(mt_scores)) if mt_scores else 60.0

    avg_i1 = float(np.mean(i1_scores)) if i1_scores else 60.0
    avg_i2 = float(np.mean(i2_scores)) if i2_scores else 60.0
    score_trend = avg_i2 - avg_i1

    # Subject-wise averages across non-final exams
    subj_map: Dict[str, List[float]] = {}
    for sc in exam_scores:
        if sc["exam_type"] != "final":
            subj_name = sc.get("subject", "general")
            subj_map.setdefault(subj_name, []).append(sc["score"])

    subj_means = [float(np.mean(scores)) for scores in subj_map.values()] if subj_map else [60.0]
    min_subject_score = min(subj_means) if subj_means else 60.0
    subject_score_std = float(np.std(subj_means)) if len(subj_means) > 1 else 0.0

    # Attendance
    attendance = term_data.get("attendance", [])
    att_pcts = []
    for att in attendance:
        tot = att.get("days_total", 22)
        pres = att.get("days_present", 22)
        att_pcts.append((pres / tot) * 100.0 if tot > 0 else 100.0)

    attendance_pct = float(np.mean(att_pcts)) if att_pcts else 100.0
    attendance_trend = 0.0
    if len(att_pcts) >= 4:
        attendance_trend = float(np.mean(att_pcts[-2:]) - np.mean(att_pcts[:-2]))

    # Assignments
    assignments = term_data.get("assignments", [])
    asg_scores = [a["score"] for a in assignments]
    avg_assignment_score = float(np.mean(asg_scores)) if asg_scores else 70.0
    ontime_count = sum(1 for a in assignments if a.get("submitted_on_time", True))
    ontime_rate = (ontime_count / len(assignments)) if assignments else 0.85

    # Projects
    projects = term_data.get("projects", [])
    project_count = len(projects)
    proj_scores = [p["score"] for p in projects]
    avg_project_score = float(np.mean(proj_scores)) if proj_scores else 0.0
    lead_roles = sum(1 for p in projects if p.get("role") == "lead")

    # Extracurriculars
    activities = term_data.get("extracurriculars", [])
    activity_count = len(activities)
    activity_hours = sum(a.get("hours_per_month", 0.0) for a in activities)
    activity_diversity = len(set(a.get("category", "other") for a in activities))

    # Competitions
    competitions = term_data.get("competitions", [])
    competition_count = len(competitions)
    highest_level = max([LEVEL_ORDER.get(c.get("level", "school"), 1) for c in competitions], default=0)
    competition_wins = sum(1 for c in competitions if c.get("result") == "winner")

    # Development Indicators
    ind = term_data.get("indicators", {}) or {}
    communication = ind.get("communication", 3)
    presentation = ind.get("presentation", 3)
    confidence = ind.get("confidence", 3)
    participation = ind.get("participation", 3)
    teamwork = ind.get("teamwork", 3)

    # Composite engagement index (formula matching spec)
    engagement_index = 100.0 * (
        min(attendance_pct / 100.0, 1.0) * 0.35 +
        ontime_rate * 0.25 +
        (participation / 5.0) * 0.25 +
        min(activity_hours / 12.0, 1.0) * 0.15
    )

    return [
        round2(avg_internal),
        round2(avg_midterm),
        round2(score_trend),
        round2(min_subject_score),
        round2(subject_score_std),
        round2(attendance_pct),
        round2(attendance_trend),
        round2(avg_assignment_score),
        round2(ontime_rate),
        float(project_count),
        round2(avg_project_score),
        float(lead_roles),
        float(activity_count),
        round2(activity_hours),
        float(activity_diversity),
        float(competition_count),
        float(highest_level),
        float(competition_wins),
        float(communication),
        float(presentation),
        float(confidence),
        float(participation),
        float(teamwork),
        round2(engagement_index),
    ]


def build_features_from_db(student_id: str, term: str, db: Session) -> List[float]:
    """Queries student term records from database and builds the 24-feature vector."""
    # Exams
    scores = (
        db.query(ExamScore)
        .filter(ExamScore.student_id == student_id, ExamScore.term == term)
        .all()
    )
    i1_scores = [s.score for s in scores if s.exam_type == "internal1"]
    i2_scores = [s.score for s in scores if s.exam_type == "internal2"]
    mt_scores = [s.score for s in scores if s.exam_type == "midterm"]

    avg_internal = float(np.mean(i1_scores + i2_scores)) if (i1_scores or i2_scores) else 60.0
    avg_midterm = float(np.mean(mt_scores)) if mt_scores else 60.0
    avg_i1 = float(np.mean(i1_scores)) if i1_scores else 60.0
    avg_i2 = float(np.mean(i2_scores)) if i2_scores else 60.0
    score_trend = avg_i2 - avg_i1

    subj_map: Dict[str, List[float]] = {}
    for s in scores:
        if s.exam_type != "final":
            subj_map.setdefault(s.subject_id, []).append(s.score)

    subj_means = [float(np.mean(v)) for v in subj_map.values()] if subj_map else [60.0]
    min_subject_score = min(subj_means) if subj_means else 60.0
    subject_score_std = float(np.std(subj_means)) if len(subj_means) > 1 else 0.0

    # Attendance
    attendance_records = (
        db.query(Attendance)
        .filter(Attendance.student_id == student_id)
        .order_by(Attendance.month.asc())
        .all()
    )
    # Filter for term months (Term 1: 07-10, Term 2: 11-02)
    if term == "Term 1":
        term_att = [a for a in attendance_records if a.month in ["2024-07", "2024-08", "2024-09", "2024-10"]]
    else:
        term_att = [a for a in attendance_records if a.month in ["2024-11", "2024-12", "2025-01", "2025-02"]]

    if not term_att:
        term_att = attendance_records[-4:] if len(attendance_records) >= 4 else attendance_records

    att_pcts = [
        (a.days_present / a.days_total * 100.0) if a.days_total > 0 else 100.0
        for a in term_att
    ]
    attendance_pct = float(np.mean(att_pcts)) if att_pcts else 100.0
    attendance_trend = 0.0
    if len(att_pcts) >= 4:
        attendance_trend = float(np.mean(att_pcts[-2:]) - np.mean(att_pcts[:-2]))

    # Assignments
    assignments = (
        db.query(Assignment)
        .filter(Assignment.student_id == student_id, Assignment.term == term)
        .all()
    )
    asg_scores = [a.score for a in assignments]
    avg_assignment_score = float(np.mean(asg_scores)) if asg_scores else 70.0
    ontime_count = sum(1 for a in assignments if a.submitted_on_time)
    ontime_rate = (ontime_count / len(assignments)) if assignments else 0.85

    # Projects
    projects = (
        db.query(Project)
        .filter(Project.student_id == student_id, Project.term == term)
        .all()
    )
    project_count = len(projects)
    proj_scores = [p.score for p in projects]
    avg_project_score = float(np.mean(proj_scores)) if proj_scores else 0.0
    lead_roles = sum(1 for p in projects if p.role == "lead")

    # Extracurriculars
    activities = (
        db.query(Extracurricular)
        .filter(Extracurricular.student_id == student_id, Extracurricular.term == term)
        .all()
    )
    activity_count = len(activities)
    activity_hours = sum(a.hours_per_month for a in activities)
    activity_diversity = len(set(a.category for a in activities))

    # Competitions
    competitions = (
        db.query(Competition)
        .filter(Competition.student_id == student_id)
        .all()
    )
    competition_count = len(competitions)
    highest_level = max([LEVEL_ORDER.get(c.level, 1) for c in competitions], default=0)
    competition_wins = sum(1 for c in competitions if c.result == "winner")

    # Development Indicators
    ind = (
        db.query(DevelopmentIndicator)
        .filter(DevelopmentIndicator.student_id == student_id, DevelopmentIndicator.term == term)
        .first()
    )
    communication = ind.communication if ind else 3
    presentation = ind.presentation if ind else 3
    confidence = ind.confidence if ind else 3
    participation = ind.participation if ind else 3
    teamwork = ind.teamwork if ind else 3

    # Composite engagement index
    engagement_index = 100.0 * (
        min(attendance_pct / 100.0, 1.0) * 0.35 +
        ontime_rate * 0.25 +
        (participation / 5.0) * 0.25 +
        min(activity_hours / 12.0, 1.0) * 0.15
    )

    return [
        round2(avg_internal),
        round2(avg_midterm),
        round2(score_trend),
        round2(min_subject_score),
        round2(subject_score_std),
        round2(attendance_pct),
        round2(attendance_trend),
        round2(avg_assignment_score),
        round2(ontime_rate),
        float(project_count),
        round2(avg_project_score),
        float(lead_roles),
        float(activity_count),
        round2(activity_hours),
        float(activity_diversity),
        float(competition_count),
        float(highest_level),
        float(competition_wins),
        float(communication),
        float(presentation),
        float(confidence),
        float(participation),
        float(teamwork),
        round2(engagement_index),
    ]
