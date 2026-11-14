"""
AI Study Companion & Interactive Assistant Router.

Features:
- Role-aware context engine (Student, Teacher, Admin)
- Dual-brain inference:
  1. Primary: LLM generation if GEMINI_API_KEY / GROQ_API_KEY configured
  2. Fallback: Intelligent data-grounded contextual response engine
- Real-time live metrics and personalized recommendations integration
"""

import os
import re
import json
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.db.session import get_db
from app.db.models import (
    User,
    Student,
    Class,
    ExamScore,
    Attendance,
    Recommendation,
    MlPrediction,
    MlModelInfo,
    DevelopmentIndicator,
)
from app.core.dependencies import get_current_user
from app.ml.predict import predict_student

logger = logging.getLogger(__name__)

router = APIRouter(tags=["AI Assistant"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text content")


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="User query message")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent conversation turns")


class AssistantChatResponse(BaseModel):
    reply: str
    mode: str = Field(..., description="'ai' or 'companion'")
    intent: Optional[str] = None


def _detect_intent(message: str) -> str:
    m = message.lower()
    if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening)|namaste|greetings)\b", m):
        return "greeting"
    if re.search(r"\b(institution|campus|school|overview|health|status|summary)\b", m):
        return "overview"
    if re.search(r"\b(attendance|absent|present|leave|bunk)\b", m):
        return "attendance"
    if re.search(r"\b(assignment|assignments|homework|submit|submission|on-time|ontime|due)\b", m):
        return "assignments"
    if re.search(r"\b(study|prepare|revision|exam|exams|test|routine|schedule)\b", m):
        return "study"
    if re.search(r"\b(improve|improvement|better|score|scores|grade|grades|prediction|predict|boost|marks)\b", m):
        return "improve"
    if re.search(r"\b(activity|activities|sport|sports|club|clubs|extracurricular|competition|hobby)\b", m):
        return "activities"
    if re.search(r"\b(goal|goals|target|aim|plan)\b", m):
        return "goal"
    if re.search(r"\b(thank|thanks|awesome|great|cool)\b", m):
        return "thanks"
    if re.search(r"\b(flag|flagged|risk|support|monitor|monitoring|attention)\b", m):
        return "flagged"
    if re.search(r"\b(work|system|model|ai|how|what|why|fair|fairness|shap)\b", m):
        return "explain"
    return "general"


def _build_student_context(user: User, db: Session) -> Dict[str, Any]:
    student = db.query(Student).filter(Student.user_id == user.id).first()
    if not student:
        return {}

    cls = db.query(Class).filter(Class.id == student.class_id).first()
    class_name = cls.name if cls else "Enrolled Class"

    # Prediction
    pred = (
        db.query(MlPrediction)
        .filter(MlPrediction.student_id == student.id)
        .order_by(MlPrediction.generated_at.desc())
        .first()
    )
    if not pred:
        try:
            pred_res = predict_student(student_id=student.id, term="Term 1", db=db, persist=True)
            pred = (
                db.query(MlPrediction)
                .filter(MlPrediction.student_id == student.id)
                .first()
            )
        except Exception:
            pred = None

    # Attendance
    attendances = db.query(Attendance).filter(Attendance.student_id == student.id).all()
    total_present = sum(a.days_present for a in attendances)
    total_days = sum(a.days_total for a in attendances)
    if total_days > 0:
        att_pct = round(total_present / total_days * 100, 1)
    else:
        att_pct = 85.0

    # Scores
    scores = db.query(ExamScore).filter(ExamScore.student_id == student.id).all()
    avg_score = round(sum(s.score for s in scores) / len(scores), 1) if scores else 75.0

    # Indicators
    ind = (
        db.query(DevelopmentIndicator)
        .filter(DevelopmentIndicator.student_id == student.id)
        .order_by(DevelopmentIndicator.created_at.desc())
        .first()
    )
    engagement_index = round((ind.participation + ind.teamwork) * 10, 1) if ind else 70.0
    assignment_avg = round((ind.confidence + ind.presentation) * 10, 1) if ind else 80.0

    # Recommendations & SHAP
    recs = []
    improvements = []
    if pred:
        rec_objs = (
            db.query(Recommendation)
            .filter(Recommendation.prediction_id == pred.id)
            .order_by(Recommendation.priority.asc())
            .limit(3)
            .all()
        )
        recs = [r.text for r in rec_objs]
        if pred.shap_json:
            try:
                shap_obj = json.loads(pred.shap_json)
                neg_factors = shap_obj.get("top_negative", shap_obj.get("topNegative", []))
                for f in neg_factors[:3]:
                    feat_name = f.get("feature", "").replace("_", " ").title()
                    improvements.append(feat_name)
            except Exception:
                pass

    return {
        "role": "student",
        "name": user.full_name or student.full_name,
        "roll_no": student.roll_no,
        "class_name": class_name,
        "predicted_score": round(pred.predicted_score, 1) if pred else avg_score,
        "risk_category": pred.risk_category if pred else "On Track",
        "cluster_name": pred.cluster_name if pred and pred.cluster_name else "Steady Progressors",
        "attendance_pct": att_pct,
        "avg_score": avg_score,
        "assignment_avg": assignment_avg,
        "engagement_index": engagement_index,
        "improvements": improvements,
        "recommendations": recs,
    }


def _build_teacher_context(user: User, db: Session) -> Dict[str, Any]:
    cls = db.query(Class).filter(Class.teacher_id == user.id).first()
    if not cls:
        cls = db.query(Class).first()

    class_name = cls.name if cls else "General Class"
    class_id = cls.id if cls else ""

    students = db.query(Student).filter(Student.class_id == class_id).all() if class_id else []
    n_students = len(students)
    student_ids = [s.id for s in students]

    preds = (
        db.query(MlPrediction)
        .filter(MlPrediction.student_id.in_(student_ids))
        .all()
    ) if student_ids else []

    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    scores = []
    flagged = []

    for p in preds:
        cat = p.risk_category or "On Track"
        risk_counts[cat] = risk_counts.get(cat, 0) + 1
        if p.predicted_score is not None:
            scores.append(p.predicted_score)
        if cat in ["Needs Support", "Needs Monitoring"]:
            st = next((s for s in students if s.id == p.student_id), None)
            if st:
                flagged.append(f"{st.full_name} ({cat}, predicted {round(p.predicted_score, 1)})")

    avg_pred = round(sum(scores) / len(scores), 1) if scores else 74.0

    # Attendance average
    attendances = (
        db.query(Attendance)
        .filter(Attendance.student_id.in_(student_ids))
        .all()
    ) if student_ids else []
    total_present = sum(a.days_present for a in attendances)
    total_days = sum(a.days_total for a in attendances)
    if total_days > 0:
        avg_att = round(total_present / total_days * 100, 1)
    else:
        avg_att = 82.5

    return {
        "role": "teacher",
        "name": user.full_name or "Teacher",
        "class_name": class_name,
        "n_students": n_students,
        "avg_predicted": avg_pred,
        "avg_attendance": avg_att,
        "on_track": risk_counts.get("On Track", 0),
        "monitoring": risk_counts.get("Needs Monitoring", 0),
        "support": risk_counts.get("Needs Support", 0),
        "flagged": flagged[:5],
    }


def _build_admin_context(user: User, db: Session) -> Dict[str, Any]:
    n_students = db.query(Student).count()
    n_classes = db.query(Class).count()
    n_teachers = db.query(User).filter(User.role == "teacher").count()

    preds = db.query(MlPrediction).all()
    risk_counts = {"On Track": 0, "Needs Monitoring": 0, "Needs Support": 0}
    scores = []
    for p in preds:
        cat = p.risk_category or "On Track"
        risk_counts[cat] = risk_counts.get(cat, 0) + 1
        if p.predicted_score is not None:
            scores.append(p.predicted_score)

    avg_pred = round(sum(scores) / len(scores), 1) if scores else 73.5
    model_info = db.query(MlModelInfo).order_by(MlModelInfo.trained_at.desc()).first()

    return {
        "role": "admin",
        "name": user.full_name or "Administrator",
        "n_students": n_students,
        "n_classes": n_classes,
        "n_teachers": n_teachers,
        "avg_predicted": avg_pred,
        "on_track": risk_counts.get("On Track", 0),
        "monitoring": risk_counts.get("Needs Monitoring", 0),
        "support": risk_counts.get("Needs Support", 0),
        "model_version": model_info.version if model_info else "v1.0.0",
        "trained_at": model_info.trained_at.strftime("%b %d, %Y") if model_info and model_info.trained_at else "Recent",
    }


def _generate_rule_reply(ctx: Dict[str, Any], message: str) -> str:
    role = ctx.get("role", "student")
    intent = _detect_intent(message)
    lines: List[str] = []

    if role == "student":
        name = ctx.get("name", "Student").split()[0]
        pred_score = ctx.get("predicted_score", 75.0)
        risk = ctx.get("risk_category", "On Track")
        att = ctx.get("attendance_pct", 85.0)
        recs = ctx.get("recommendations", [])
        improvements = ctx.get("improvements", [])

        if intent in ["greeting", "thanks"]:
            lines.append(f"Hello {name}! 👋")
            lines.append(
                f"Here is your term snapshot: predicted score **{pred_score}/100** ({risk}), "
                f"attendance **{att}%**, and profile **{ctx.get('cluster_name', 'Steady Progressor')}**."
            )
            lines.append("Feel free to ask me about your attendance, exam preparation, personalized recommendations, or study routines!")

        elif intent == "attendance":
            lines.append(f"Your current attendance rate is **{att}%**.")
            if att < 75.0:
                lines.append(
                    "⚠️ Notice: This is currently below the 75% school threshold. "
                    "In our ML model, regular attendance has the highest single positive impact on your final score. "
                    "Attending just 3 more consecutive classes can notably boost your predicted performance!"
                )
            else:
                lines.append("✅ Great job! Your attendance is solid and above the required 75% benchmark. Maintaining this protects your academic momentum.")

        elif intent == "assignments":
            assign_avg = ctx.get("assignment_avg", 80.0)
            lines.append(f"Your assignment completion rate is **{assign_avg}%**.")
            if assign_avg < 80.0:
                lines.append("💡 Tip: Submitting homework on time directly enhances your continuous assessment score. Try reserving 30 minutes right after school for same-day submissions.")
            else:
                lines.append("⭐ Your submission habits are strong. Keep following your daily routine!")

        elif intent in ["study", "improve"]:
            lines.append(f"Your current predicted score is **{pred_score}/100** ({risk}).")
            if improvements:
                lines.append(f"Top areas with the highest improvement potential: **{', '.join(improvements)}**.")
            if recs:
                lines.append("\n**Action Recommendations:**")
                for i, r in enumerate(recs[:3], 1):
                    lines.append(f"{i}. {r}")
            lines.append("\n💡 Pro-tip: You can use the What-If Simulator on your dashboard to see how increasing study hours or attendance moves your score in real time.")

        elif intent == "activities":
            eng = ctx.get("engagement_index", 70.0)
            lines.append(f"Your extracurricular engagement score is **{eng}/100**.")
            lines.append("Participating in student clubs, science exhibitions, or sports helps cultivate soft skills and strengthens your overall development cluster rating.")

        elif intent == "explain":
            lines.append(
                "How this AI works: It evaluates 18 holistic development indicators—exam scores, attendance consistency, "
                "assignment submissions, projects, and co-curriculars—using an ensemble XGBoost model."
            )
            lines.append(
                "Crucially, our system does NOT look at demographic or personal background factors, and predictions "
                "are supportive guideposts for growth, NOT permanent marks!"
            )

        elif intent == "goal":
            lines.append(f"Setting clear targets is a proven way to succeed! Your current prediction is **{pred_score}/100**.")
            lines.append(f"A great target would be **{round(pred_score + 5, 1)}** for next term. Break it down into weekly focus tasks.")

        else:
            lines.append(f"Here's where you stand right now, {name}:")
            lines.append(f"• Predicted Score: **{pred_score}/100** ({risk})")
            lines.append(f"• Attendance: **{att}%**")
            lines.append(f"• Development Profile: **{ctx.get('cluster_name', 'Active Learner')}**")
            if recs:
                lines.append(f"• Key Recommendation: {recs[0]}")
            lines.append("\nYou can ask me: 'How can I improve my marks?', 'Check my attendance', or 'Study tips'.")

    elif role == "teacher":
        name = ctx.get("name", "Teacher")
        cls_name = ctx.get("class_name", "Class")
        n_st = ctx.get("n_students", 0)
        avg_pred = ctx.get("avg_predicted", 75.0)
        avg_att = ctx.get("avg_attendance", 80.0)
        support = ctx.get("support", 0)
        monitoring = ctx.get("monitoring", 0)
        on_track = ctx.get("on_track", 0)
        flagged = ctx.get("flagged", [])

        if intent in ["greeting", "thanks", "overview"]:
            lines.append(f"Hello {name}! 👩‍🏫")
            lines.append(
                f"Here is your **{cls_name}** summary: **{n_st}** students, "
                f"class average predicted score **{avg_pred}/100**, and average attendance **{avg_att}%**."
            )
            lines.append(f"Status distribution: **{on_track}** On Track, **{monitoring}** Needs Monitoring, **{support}** Needs Support.")

        elif intent in ["flagged", "improve"]:
            lines.append(f"In **{cls_name}**, {support} student(s) need support and {monitoring} need monitoring.")
            if flagged:
                lines.append("\n**Students requiring attention:**")
                for f in flagged[:4]:
                    lines.append(f"• {f}")
            lines.append("\nRecommended action: Filter your student roster by 'Needs Support' to review their specific SHAP factor breakdowns and log targeted interventions.")

        elif intent == "attendance":
            lines.append(f"Class attendance for **{cls_name}** averages **{avg_att}%**.")
            lines.append("Students falling below 75% trigger an automatic monitoring flag. Check the attendance tab to see monthly trends.")

        elif intent in ["explain", "model"]:
            lines.append("Analytics Guide: The teacher dashboard combines three ML models:")
            lines.append("1. **XGBoost Regressor**: forecasts end-of-term composite scores.")
            lines.append("2. **XGBoost Classifier**: categorizes support levels (On Track, Monitoring, Support).")
            lines.append("3. **K-Means Clustering**: groups students into intuitive development archetypes.")

        else:
            lines.append(f"**{cls_name} Overview**: {n_st} students enrolled.")
            lines.append(f"• Average Predicted Score: **{avg_pred}/100**")
            lines.append(f"• Class Attendance: **{avg_att}%**")
            lines.append(f"• Needs Support: **{support}** | Needs Monitoring: **{monitoring}**")
            lines.append("\nAsk me about students needing attention, class attendance trends, or grading analytics.")

    else:  # Admin
        name = ctx.get("name", "Administrator")
        n_st = ctx.get("n_students", 0)
        n_cls = ctx.get("n_classes", 0)
        n_t = ctx.get("n_teachers", 0)
        avg_pred = ctx.get("avg_predicted", 75.0)
        support = ctx.get("support", 0)
        monitoring = ctx.get("monitoring", 0)
        on_track = ctx.get("on_track", 0)

        if intent in ["greeting", "thanks", "overview"]:
            lines.append(f"Welcome, {name}! 🏛️")
            lines.append(
                f"Institution health: **{n_st}** students across **{n_cls}** classes and **{n_t}** faculty members. "
                f"Campus average predicted score is **{avg_pred}/100**."
            )
            lines.append(f"Active ML model: **{ctx.get('model_version', 'v1.0')}** (trained {ctx.get('trained_at', 'recently')}).")

        elif intent in ["flagged", "improve"]:
            lines.append(f"Institution-wide risk breakdown: **{support}** Needs Support, **{monitoring}** Needs Monitoring, **{on_track}** On Track.")
            lines.append("You can inspect the Classes Comparison chart on the Admin portal to identify classes that require additional teacher aides or tutoring resources.")

        elif intent in ["explain", "model"]:
            lines.append(f"ML Pipeline Status: Running version **{ctx.get('model_version', 'v1.0')}**.")
            lines.append("The system features an automated retraining trigger on the Admin dashboard that refreshes regression, classification, clustering, and SHAP weights on live database records.")

        else:
            lines.append("Institution Statistics:")
            lines.append(f"• Students: **{n_st}** | Classes: **{n_cls}** | Teachers: **{n_t}**")
            lines.append(f"• Institution Mean Score: **{avg_pred}/100**")
            lines.append(f"• Needs Support: **{support}** | Needs Monitoring: **{monitoring}**")
            lines.append("\nAsk me about overall risk distributions, class benchmarks, or ML model retraining.")

    return "\n".join(lines)


def _get_groq_keys() -> List[str]:
    keys = []
    for k in ["GROQ_API_KEY_1", "GROQ_API_KEY_2", "GROQ_API_KEY"]:
        val = os.environ.get(k, "").strip()
        if val and val not in keys:
            keys.append(val)
    return keys


_groq_cursor = 0


async def _call_groq_llm(messages: List[Dict[str, str]]) -> Optional[str]:
    global _groq_cursor
    keys = _get_groq_keys()
    if not keys:
        return None

    ordered_keys = keys[_groq_cursor:] + keys[:_groq_cursor]
    _groq_cursor = (_groq_cursor + 1) % len(keys)
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    import httpx
    for key in ordered_keys:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.6,
                        "max_tokens": 600,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                    if reply:
                        return reply
                elif res.status_code in [401, 403, 429, 500, 502, 503]:
                    logger.warning(f"Groq API returned status {res.status_code}, rotating key...")
                    continue
        except Exception as e:
            logger.warning(f"Groq API connection error: {e}")
            continue

    return None


async def _call_gemini_llm(system_instruction: str, message: str, history: List[ChatMessage]) -> Optional[str]:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        contents = [{"role": "user", "parts": [{"text": system_instruction}]}]
        for h in history[-4:]:
            contents.append({
                "role": "model" if h.role == "assistant" else "user",
                "parts": [{"text": h.content}],
            })
        contents.append({"role": "user", "parts": [{"text": message}]})

        async with httpx.AsyncClient(timeout=12.0) as client:
            resp = await client.post(url, json={"contents": contents})
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
    except Exception as exc:
        logger.warning(f"Gemini API assistant error: {exc}")

    return None


async def _generate_llm_reply(ctx: Dict[str, Any], message: str, history: List[ChatMessage]) -> Optional[str]:
    """Cascading real LLM generation (Groq LLaMA 3.3 -> Gemini 1.5 Flash -> Local Fallback)."""
    system_instruction = (
        "You are the friendly AI Study Companion inside the AI-Based Student Performance & Development System. "
        "You provide empathetic, encouraging, accurate guidance using the provided live institutional/student context. "
        "Never share negative or fatalistic judgments; use supportive coaching language. Keep replies concise, structured, "
        "and under 180 words. Role context: " + json.dumps(ctx)
    )

    # 1. Try Groq (high-speed inference with rotating keys)
    groq_messages = [{"role": "system", "content": system_instruction}]
    for h in history[-4:]:
        groq_messages.append({
            "role": "assistant" if h.role == "assistant" else "user",
            "content": h.content,
        })
    groq_messages.append({"role": "user", "content": message})

    groq_reply = await _call_groq_llm(groq_messages)
    if groq_reply:
        return groq_reply

    # 2. Try Gemini
    gemini_reply = await _call_gemini_llm(system_instruction, message, history)
    if gemini_reply:
        return gemini_reply

    return None


@router.post("/assistant/chat", response_model=AssistantChatResponse)
@router.post("/ai/assistant", response_model=AssistantChatResponse)
async def chat_with_assistant(
    req: AssistantChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Role-aware interactive AI assistant chat endpoint.
    - Dynamically gathers live metrics for student, teacher, or admin.
    - Cascades from LLM generation to resilient local context engine.
    """
    # 1. Build live role-specific context
    if current_user.role == "student":
        ctx = _build_student_context(current_user, db)
    elif current_user.role == "teacher":
        ctx = _build_teacher_context(current_user, db)
    else:
        ctx = _build_admin_context(current_user, db)

    # 2. Try external LLM if configured
    llm_reply = await _generate_llm_reply(ctx, req.message, req.history or [])
    if llm_reply:
        return AssistantChatResponse(reply=llm_reply, mode="ai", intent=_detect_intent(req.message))

    # 3. Fast, reliable local context engine
    rule_reply = _generate_rule_reply(ctx, req.message)
    return AssistantChatResponse(reply=rule_reply, mode="companion", intent=_detect_intent(req.message))
