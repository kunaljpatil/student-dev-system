# AI-Based Student Performance & Development System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![SQLAlchemy 2.x](https://img.shields.io/badge/SQLAlchemy-2.0+-D71F00?logo=sqlalchemy&logoColor=white)](https://www.sqlalchemy.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-TreeExplainer-FF6F00?logo=scikitlearn&logoColor=white)](https://shap.readthedocs.io/)
[![Tests Passing](https://img.shields.io/badge/pytest-59%20passed-2ea44f?logo=pytest&logoColor=white)](#testing)

> **One-Line Summary:** An AI-powered full-stack student development platform that analyzes multidimensional student data (academics, attendance, projects, activities, competitions, and developmental soft skills), predicts final performance, identifies student growth profiles, explains influential drivers with SHAP, and provides personalized supportive recommendations through responsive Student, Teacher, and Administrator dashboards.

---

## 1. System Architecture

```
Browser (Student / Teacher / Admin)
        |
        |  HTML5 + Vanilla CSS + JS (Bootstrap 5, Chart.js)
        |  Glassmorphism Aesthetic (60-30-10 Warm Beige & Espresso)
        v
FastAPI Application  (routers: auth, students, teachers, admin, ml, views)
   |                                 |
   v                                 v
Database Layer                ML Service Layer
(SQLite / PostgreSQL 15+)     preprocessing -> features (24) -> models -> SHAP
SQLAlchemy 2.x + Alembic             |
                                     v
                             Recommendation Engine
                             (rule-based, max 5 actions)
                                     |
                                     v
                             Stored Predictions + REST APIs
```

---

## 2. Technology Stack

| Layer | Technology | Description |
|---|---|---|
| **Language** | Python 3.11.9+ | Modern typed Python |
| **Backend Framework** | FastAPI, Uvicorn, Pydantic v2 | High-performance asynchronous API & routing |
| **Database & ORM** | SQLite 3 / PostgreSQL 15+, SQLAlchemy 2.x | Relational persistence with foreign keys & indexes |
| **Database Migrations** | Alembic | Version-controlled schema migrations |
| **Security & Auth** | Native Bcrypt, PyJWT | Irreversible salt hashing & 60-min JWT access tokens |
| **Frontend Templates** | Server-rendered Jinja2 | Fast server-side rendering without heavy SPA toolchains |
| **Styling & Theme** | Bootstrap 5, Vanilla CSS | Custom 60-30-10 Warm Beige Glassmorphic design system |
| **Visualizations** | Chart.js 4.4+ | Interactive line, bar, doughnut, and radar charts |
| **Machine Learning** | scikit-learn, XGBoost, Pandas, NumPy | Predictive regression, multi-class classification, clustering |
| **Explainability** | SHAP (`shap.TreeExplainer`) | Local and global game-theoretic feature attribution |
| **Artifact Serialization** | Joblib | Binary model weights, scalers, and metadata persistence |
| **Testing** | Pytest, HTTPX TestClient | Comprehensive unit, API, ML, and security test suites |

---

## 3. UI Design System: 60-30-10 Warm Beige Glassmorphism

The frontend adheres strictly to the **60-30-10 rule of visual balance**:

* **60% Dominant Canvas & Neutral:** Warm Oatmeal Beige & Soft Cream (`#f6f1ea` to `#ede5da`) with delicate radial gradients and translucent frosted glass card surfaces (`rgba(255, 253, 250, 0.75)`, `backdrop-filter: blur(14px)`, `box-shadow: 0 10px 30px -5px rgba(62, 44, 34, 0.07)`).
* **30% Secondary Structure & Typography:** Rich Roasted Espresso Brown (`#2a1d15`, `#3e2b20`) for headings, brand badges, navbar structure, and table headers.
* **10% High-Impact Accent Color (CTAs):** Radiant Caramel Amber & Sunlit Copper (`#d97706`, `#e07a24`) with glow effects (`rgba(217, 119, 6, 0.25)`) for action buttons, prediction highlights, and interactive focal points.
* **Supportive, Non-Alarming Status Badges:**
  * **On Track:** Sage Green (`#227c49` on `#e8f5ed`)
  * **Needs Monitoring:** Golden Honey Amber (`#b45309` on `#fdf6e7`)
  * **Needs Support:** Soft Terracotta Rose (`#c2410c` on `#fbeee8`)

---

## 4. Machine Learning Specification

### 4.1 Feature Engineering (24 Features)

The pipeline extracts **24 engineered features** per student per term:
* **Academic Performance:** `avg_internal_score`, `avg_midterm_score`, `score_trend` (internal 2 vs internal 1), `min_subject_score`, `subject_score_std`.
* **Attendance & Regularity:** `attendance_pct`, `attendance_trend` (last 2 months vs earlier).
* **Coursework & Assignments:** `avg_assignment_score`, `ontime_rate` (on-time submission ratio).
* **Project Work:** `project_count`, `avg_project_score`, `lead_roles`.
* **Extracurricular Activities:** `activity_count`, `activity_hours` (hours/month), `activity_diversity` (category count across sports, arts, music, STEM, volunteering).
* **Competitions:** `competition_count`, `highest_competition_level` (school, district, state, national), `competition_wins`.
* **Development Indicators (Soft Skills 1-5):** `communication`, `presentation`, `confidence`, `participation`, `teamwork`.
* **Composite Indicator:** `engagement_index` (normalized weighted index of attendance, submissions, classroom participation, and co-curricular hours).

> [!IMPORTANT]
> **Ethical Safeguard:** No demographic or sensitive attributes (*gender, religion, caste, family income, race*) are used anywhere in the ML feature pipeline.

### 4.2 Model Evaluation & Baselines Comparison

All models are trained with a fixed random seed (`2024`) on an 80/20 stratified train/test split.

| Model # | Project Name | Algorithm | Evaluation Metrics | Baseline Comparison |
|---|---|---|---|---|
| **Model 1** | **Student Performance Prediction Model** | **XGBoost Regressor** (`max_depth=4`, `lr=0.08`, `n_est=150`) | **$R^2$: 0.841**<br>MAE: 4.08 marks<br>RMSE: 5.24 marks<br>5-Fold CV $R^2$: $0.800 \pm 0.036$ | **Ridge Regression:**<br>$R^2$: 0.840<br>MAE: 4.45 marks |
| **Model 2** | **Student Performance Risk Classification Model** | **XGBoost Classifier** (`max_depth=3`, `lr=0.08`, `n_est=120`) | **Accuracy: 74.3%**<br>**Macro F1: 0.714**<br>On Track F1: 0.750<br>Needs Monitoring F1: 0.550<br>Needs Support F1: 0.842 | **Logistic Regression:**<br>Accuracy: 71.4%<br>Macro F1: 0.673 |
| **Model 3** | **Student Development Profiling Model** | **K-Means Clustering** ($k=3$, StandardScaler) | **Silhouette Score: 0.188**<br>Inertia: 2823.7<br>Elbow search validated over $k \in [2..6]$ | Cluster 0: *Steady Progressors* (33.7%)<br>Cluster 1: *High Achiever - Well Rounded* (35.7%)<br>Cluster 2: *Academically Strong - Low Engagement* (30.6%) |

### 4.3 Target & Label Definitions

* **Regressor Target (`final_exam_overall`):** Represents the overall outcome from the final term assessment (0–100), predicted from prior indicators (internals, attendance, assignments, activities). It is not a trivial linear sum of inputs.
* **Classifier Target (`risk_category`):** Defined independently of the regressor output:
  * `Needs Support`: Final score < 50 OR attendance < 75%.
  * `Needs Monitoring`: Borderline score (50–65) OR downward score trend ($\le -5$).
  * `On Track`: Consistently performing and engaged ($\ge 65$).

### 4.4 Explainability (SHAP)

* Implemented with `shap.TreeExplainer` on the trained XGBoost Regressor.
* **Local Attribution:** Per student prediction, extracts the **top 5 positive drivers** (pushing forecast higher) and **top 5 negative focus areas** (pulling score down), stored as JSON in `ml_predictions.shap_json`.
* **Global Attribution:** Computes mean absolute SHAP values across the entire cohort to identify the top institutional drivers of student success (*Weakest Subject*, *Internal Average*, *Assignment Average*, *Midterm Average*, *Score Trend*).

### 4.5 Recommendation Engine

Rule-based decision support mapping student performance, risk category, cluster profile, and SHAP factors to at most **5 prioritized, constructive actions**:
* **Priority 1 (High Impact Intervention):** Attendance obstacles check-in with teacher, doubt-clearing sessions for lowest-scoring subject, bi-weekly mentor milestones.
* **Priority 2 (Habit & Skill Building):** Weekly submission calendar, debate and public speaking circles, collaborative project responsibilities.
* **Priority 3 (Enrichment & Leadership):** Co-curricular club balance, regional competition preparation, independent research capstones.

---

## 5. Security & Ownership Architecture

* **Authentication:** Password hashing via native `bcrypt` with individual random salts. JWT access tokens signed with `HS256` and 60-minute expiration.
* **Role-Based Access Control (RBAC):**
  * `student`: Restricted strictly to personal records and personal insights.
  * `teacher`: Class teachers can only view and mutate students enrolled in their assigned classroom.
  * `admin`: Institutional oversight across all classrooms and faculty, plus one-click ML retraining.
* **Ownership Enforcement:** Dependency-injected `verify_student_access()` blocks unauthorized cross-classroom queries with HTTP 403 Forbidden.
* **SQL Injection Safety:** 100% parameterized SQLAlchemy ORM queries; no raw concatenated SQL strings.
* **Boundary Validation:** Pydantic v2 enforces strict score limits [0..100] and indicators [1..5].

---

## 6. Seed Accounts & Default Credentials

Seeded across 7 classes $\times$ 50 students = **350 students**, 7 class teachers, and 1 administrator.

| Role | Email | Password | Notes |
|---|---|---|---|
| **Admin** | `admin@school.test` | `Admin@123` | Institutional dashboard, model governance |
| **Teacher 1** | `teacher1@school.test` | `Teach@123` | Class Teacher for **Class 1** (Roll No 1 to 50) |
| **Teacher 2** | `teacher2@school.test` | `Teach@123` | Class Teacher for **Class 2** (Roll No 51 to 100) |
| ... | ... | ... | ... |
| **Teacher 7** | `teacher7@school.test` | `Teach@123` | Class Teacher for **Class 7** (Roll No 301 to 350) |
| **Student 1** | `student001@school.test` | `Study@123` | Student Roll No 1 in Class 1 |
| **Student 50** | `student050@school.test` | `Study@123` | Student Roll No 50 in Class 1 |

*(All passwords can be modified in `.env` via `DEMO_ADMIN_PASSWORD`, `DEMO_TEACHER_PASSWORD`, `DEMO_STUDENT_PASSWORD`).*

---

## 7. Quick Start & Local Setup

### 7.1 Prerequisites
* Python 3.11.0 or higher
* Git

### 7.2 Installation

```bash
# 1. Clone repository and navigate to student-dev-system
cd student-dev-system

# 2. Create and activate virtual environment
python -m venv .venv

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment configuration
cp .env.example .env
```

### 7.3 Database Setup & Seeding

The application defaults to SQLite (`sqlite:///./student_dev.db`). To switch to PostgreSQL, set `DATABASE_URL` in `.env`:
`DATABASE_URL=postgresql://user:pass@localhost:5432/student_dev`

```bash
# 1. Apply Alembic database migrations
alembic upgrade head

# 2. Seed 350 students across 7 classes with multidimensional history
python scripts/seed_db.py

# 3. Train all 3 ML models, serialize artifacts, and generate metrics report
python scripts/train_all.py
```

### 7.4 Running the Application

```bash
uvicorn app.main:app --reload --port 8000
```

Open your browser:
* **Web Portal:** [http://localhost:8000/](http://localhost:8000/) *(auto-redirects to `/login`)*
* **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **API Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)

---

## 8. REST API Reference

All endpoints return JSON and are prefixed with `/api`.

### Authentication
* `POST /api/auth/login` — Authenticate and receive JWT access token.
* `GET /api/auth/me` — Retrieve authenticated user profile and contextual role details.

### Student Portal
* `GET /api/student/me/profile` — Student profile (roll number, class name, academic year).
* `GET /api/student/me/scores` — All exam scores and subject averages.
* `GET /api/student/me/attendance` — Monthly attendance records and overall percentage.
* `GET /api/student/me/activities` — Projects, co-curriculars, and competitions.
* `GET /api/student/me/insights` — Forecasted score, support tier, development profile, SHAP factors, and recommendations.

### Teacher Portal (Class Teacher Only)
* `GET /api/teacher/class/students` — Class roster with latest forecasts, status badges, and attention flags.
* `GET /api/teacher/students/{id}` — Full multidimensional student dossier.
* `GET /api/teacher/students/{id}/insights` — AI forecast, SHAP waterfall, and recommendations for a student.
* `POST /api/teacher/scores` — Record or update exam marks.
* `POST /api/teacher/attendance` — Update monthly attendance records.
* `POST /api/teacher/indicators` — Update 1–5 soft-skill development indicators.
* `GET /api/teacher/class/analytics` — Cohort performance, subject averages, and soft-skill radar stats.
* `GET /api/teacher/class/attention` — Filtered list of students requiring support or monitoring.
* `POST /api/teacher/class/predict` — Re-run AI predictions for all students in teacher's class.

### Administrator Portal
* `GET /api/admin/overview` — Schoolwide totals, average forecast, and risk distribution.
* `GET /api/admin/classes` — Benchmarking table across all 7 classes.
* `GET /api/admin/teachers` — Faculty directory with assigned classes.
* `GET /api/admin/clusters` — Institutional development profile segmentation.
* `GET /api/admin/feature-importance` — Ranked global SHAP systemic drivers.
* `POST /api/admin/ml/retrain` — One-click trigger to retrain all models and reload live inference.

### Machine Learning
* `POST /api/ml/predict/{student_id}` — On-demand inference for an individual student.
* `POST /api/ml/predict-cohort/{class_id}` — Batch cohort inference for a classroom.
* `GET /api/ml/model-info` — Active model metadata, metrics summary, and SHAP importance.

---

## 9. Testing

The repository includes a comprehensive Pytest test suite covering all phases:

```bash
pytest -v
```

```
tests/test_auth.py (4 tests) ............................................ PASSED
tests/test_health.py (2 tests) .......................................... PASSED
tests/test_ml_pipeline.py (7 tests) ..................................... PASSED
tests/test_phase5_recommendations_and_apis.py (14 tests) ................ PASSED
tests/test_phases_6_7_8_dashboards_and_apis.py (18 tests) ............... PASSED
tests/test_phase9_security_and_edge_cases.py (9 tests) .................. PASSED
tests/test_seed_and_models.py (4 tests) ................................. PASSED

======================= 58 passed, 1 warning in 22.15s ========================
```

---

## 10. Responsible AI & Ethical Statement

1. **Decision Support, Not Determination:** Predictions and development profiles generated by this system are **advisory decision-support aids** intended to assist teachers in identifying students who could benefit from encouragement or additional help. They do not replace teacher judgment or determine permanent academic grades.
2. **Transparent Explainability:** Every prediction is paired with local SHAP feature attributions, explaining exactly *why* a particular forecast was produced.
3. **Growth-Mindset Recommendations:** Model outputs trigger positive, actionable recommendations (e.g., study habits, club participation, peer mentoring) formulated to build confidence and avoid stigmatizing language.
4. **Fairness by Design:** Sensitive demographic attributes (*gender, religion, caste, family income*) are explicitly excluded from the feature space.
