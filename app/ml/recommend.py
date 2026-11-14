"""
Recommendation Engine Module (spec Section 6.7).

Rule-based recommendation engine driven by model predictions, risk category,
cluster profile, top negative SHAP factors, and developmental indicators.
Returns max 5 supportive, prioritized recommendations per student.
Wording is positive, growth-oriented, and actionable.
"""

from typing import Dict, Any, List


def generate_recommendations(
    predicted_score: float,
    risk_category: str,
    cluster_name: str,
    shap_explanation: Dict[str, Any],
    features: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Evaluates student multidimensional factors and generates up to 5 prioritized
    supportive recommendations.
    
    Priority levels:
    1: Immediate support / high impact
    2: Skill & habit building / medium impact
    3: Growth, leadership & enrichment
    """
    recommendations: List[Dict[str, Any]] = []

    att_pct = features.get("attendance_pct", 100.0)
    ontime = features.get("ontime_rate", 1.0)
    asg_score = features.get("avg_assignment_score", 70.0)
    min_subj = features.get("min_subject_score", 60.0)
    internal_avg = features.get("avg_internal_score", 60.0)
    comm = features.get("communication", 3.0)
    pres = features.get("presentation", 3.0)
    part = features.get("participation", 3.0)
    team = features.get("teamwork", 3.0)
    act_hrs = features.get("activity_hours", 0.0)
    comp_count = features.get("competition_count", 0.0)
    lead_roles = features.get("lead_roles", 0.0)

    # Negative SHAP factor names
    neg_features = [f["feature"] for f in shap_explanation.get("top_negative", [])]

    # Rule 1: Attendance intervention (Priority 1)
    if att_pct < 85.0 or ("attendance_pct" in neg_features and att_pct < 90.0):
        recommendations.append({
            "category": "Attendance & Regularity",
            "text": f"Improve attendance (currently {att_pct:.1f}%, target 90%+). Schedule a supportive check-in with your class teacher to review any attendance obstacles.",
            "priority": 1,
        })

    # Rule 2: Academic subject recovery (Priority 1)
    if min_subj < 50.0 or internal_avg < 50.0 or "min_subject_score" in neg_features:
        recommendations.append({
            "category": "Academic Strengthening",
            "text": f"Focus on foundational concepts in your lowest-scoring subject (current baseline: {min_subj:.1f}). Attend weekly guided problem-solving or doubt-clearing sessions.",
            "priority": 1,
        })

    # Rule 3: Mentorship support for students needing support (Priority 1)
    if risk_category == "Needs Support":
        recommendations.append({
            "category": "Mentorship & Goal Setting",
            "text": "Establish a bi-weekly academic progress check-in with your class mentor to set realistic milestones and track continuous improvements.",
            "priority": 1,
        })

    # Rule 4: Assignment consistency & time management (Priority 2)
    if ontime < 0.80 or asg_score < 60.0 or "ontime_rate" in neg_features:
        recommendations.append({
            "category": "Study Habits & Submissions",
            "text": f"Set up a personal weekly assignment submission calendar to ensure consistent, timely turn-in (current on-time rate: {int(ontime * 100)}%).",
            "priority": 2,
        })

    # Rule 5: Communication and presentation development (Priority 2)
    if comm <= 2.5 or pres <= 2.5 or "communication" in neg_features:
        recommendations.append({
            "category": "Communication & Presentation",
            "text": "Join the class debate club or public speaking circle; volunteer to present a short 3-minute summary in class once every fortnight.",
            "priority": 2,
        })

    # Rule 6: Active participation and teamwork (Priority 2)
    if part <= 2.5 or team <= 2.5:
        recommendations.append({
            "category": "Collaboration & Classroom Engagement",
            "text": "Participate actively in peer study groups and take on collaborative responsibilities in upcoming team projects.",
            "priority": 2,
        })

    # Rule 7: Co-curricular balance (Priority 3)
    if act_hrs < 4.0:
        recommendations.append({
            "category": "Holistic Engagement",
            "text": "Explore joining at least one co-curricular activity (sports, arts, music, or STEM club) to cultivate balanced wellbeing and leadership.",
            "priority": 3,
        })

    # Rule 8: Competition & enrichment opportunities (Priority 3)
    if comp_count > 0 or lead_roles > 0 or "High Achiever" in cluster_name:
        recommendations.append({
            "category": "Advanced Growth & Leadership",
            "text": "Leverage your strong skill set to prepare for regional competitions and mentor peers in collaborative study sessions.",
            "priority": 3,
        })

    # Rule 9: High performer stretch goal (Priority 3)
    if risk_category == "On Track" and predicted_score >= 75.0 and len(recommendations) < 3:
        recommendations.append({
            "category": "Enrichment & Special Projects",
            "text": "Undertake an independent self-directed research or capstone project in an area of passionate interest to deepen subject mastery.",
            "priority": 3,
        })

    # Fallback if no rules matched
    if not recommendations:
        recommendations.append({
            "category": "Steady Continuous Progress",
            "text": "Maintain your balanced study routine and steady participation across all academic and co-curricular areas.",
            "priority": 2,
        })

    # Sort strictly by priority (1 = highest) and take max 5
    recommendations.sort(key=lambda r: r["priority"])
    return recommendations[:5]
