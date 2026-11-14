"""
Database Seeding Script (spec Section 6.4, 11).

Seeds the database with:
- 1 Admin: admin@school.test
- 7 Teachers: teacher1@school.test .. teacher7@school.test
- 7 Classes: Class 1 .. Class 7
- 5 Subjects: Math, Science, English, Social Studies, Computer
- 350 Students: student001@school.test .. student350@school.test
- 2 terms of academic, attendance, assignment, project, activity, competition, indicator records.
"""

import sys
import os
import time


# Configure UTF-8 for console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.security import hash_password
from app.db.session import SessionLocal
from app.db.models import (
    User,
    Class,
    Student,
    Subject,
    ExamScore,
    Attendance,
    Assignment,
    Project,
    Extracurricular,
    Competition,
    DevelopmentIndicator,
    MlPrediction,
    Recommendation,
    MlModelInfo,
)
from app.ml.data_generation import generate_dataset, SEED, NUM_CLASSES, STUDENTS_PER_CLASS


TEACHER_NAMES = [
    "Mrs. Ananya Krishnan",
    "Mr. Rajesh Nambiar",
    "Ms. Farida Sheikh",
    "Mr. Vikram Deshpande",
    "Mrs. Latha Srinivasan",
    "Mr. Arun Kumar",
    "Ms. Neelofar Ali",
]


def seed_database():
    start_time = time.time()
    db = SessionLocal()

    try:
        print("[*] Clearing existing database records...")
        db.query(Recommendation).delete()
        db.query(MlPrediction).delete()
        db.query(DevelopmentIndicator).delete()
        db.query(Competition).delete()
        db.query(Extracurricular).delete()
        db.query(Project).delete()
        db.query(Assignment).delete()
        db.query(Attendance).delete()
        db.query(ExamScore).delete()
        db.query(Student).delete()
        db.query(Subject).delete()
        db.query(Class).delete()
        db.query(MlModelInfo).delete()
        db.query(User).delete()
        db.commit()

        print("[+] Creating Admin and Teachers...")
        admin_hash = hash_password(settings.DEMO_ADMIN_PASSWORD)
        teacher_hash = hash_password(settings.DEMO_TEACHER_PASSWORD)
        student_hash = hash_password(settings.DEMO_STUDENT_PASSWORD)

        admin = User(
            email="admin@school.test",
            password_hash=admin_hash,
            role="admin",
            full_name="Dr. Priya Menon (Principal)",
            is_active=True,
        )
        db.add(admin)

        teachers = []
        for i in range(NUM_CLASSES):
            t = User(
                email=f"teacher{i + 1}@school.test",
                password_hash=teacher_hash,
                role="teacher",
                full_name=TEACHER_NAMES[i],
                is_active=True,
            )
            db.add(t)
            teachers.append(t)

        db.flush()  # populate IDs

        print("[+] Creating 7 Classes...")
        classes = []
        for i in range(NUM_CLASSES):
            c = Class(
                name=f"Class {i + 1}",
                academic_year="2024-25",
                teacher_id=teachers[i].id,
            )
            db.add(c)
            classes.append(c)

        print("[+] Creating 5 Core Subjects...")
        raw_dataset = generate_dataset(SEED)
        subjects_map = {}
        for s_name in raw_dataset["subjects"]:
            s = Subject(name=s_name)
            db.add(s)
            subjects_map[s_name] = s

        db.flush()

        print("[+] Creating 350 Students...")
        students = []
        for st_info in raw_dataset["students"]:
            email = f"student{st_info['index'] + 1:03d}@school.test"
            user = User(
                email=email,
                password_hash=student_hash,
                role="student",
                full_name=st_info["full_name"],
                is_active=True,
            )
            db.add(user)
            db.flush()

            student = Student(
                user_id=user.id,
                class_id=classes[st_info["class_index"]].id,
                roll_no=st_info["roll_no"],
                full_name=st_info["full_name"],
                date_of_birth=None,
            )
            db.add(student)
            students.append(student)

        db.flush()

        print("[+] Inserting Multi-Dimensional Student Records (2 terms)...")
        exam_rows = []
        att_rows = []
        asg_rows = []
        proj_rows = []
        act_rows = []
        comp_rows = []
        ind_rows = []

        for st_info, student_entity in zip(raw_dataset["students"], students):
            sid = student_entity.id
            teacher_id = teachers[st_info["class_index"]].id

            for term_data in st_info["terms"]:
                term_name = term_data["term"]

                # Exam scores
                for sc in term_data["exam_scores"]:
                    exam_rows.append(
                        ExamScore(
                            student_id=sid,
                            subject_id=subjects_map[sc["subject"]].id,
                            exam_type=sc["exam_type"],
                            score=sc["score"],
                            max_score=100.0,
                            term=term_name,
                            recorded_by=teacher_id,
                        )
                    )

                # Attendance
                for att in term_data["attendance"]:
                    att_rows.append(
                        Attendance(
                            student_id=sid,
                            month=att["month"],
                            days_present=att["days_present"],
                            days_total=att["days_total"],
                        )
                    )

                # Assignments
                for asg in term_data["assignments"]:
                    asg_rows.append(
                        Assignment(
                            student_id=sid,
                            subject_id=subjects_map[asg["subject"]].id,
                            title=asg["title"],
                            score=asg["score"],
                            submitted_on_time=asg["submitted_on_time"],
                            term=term_name,
                        )
                    )

                # Projects
                for pr in term_data["projects"]:
                    proj_rows.append(
                        Project(
                            student_id=sid,
                            title=pr["title"],
                            role=pr["role"],
                            score=pr["score"],
                            term=term_name,
                        )
                    )

                # Extracurriculars
                for act in term_data["extracurriculars"]:
                    act_rows.append(
                        Extracurricular(
                            student_id=sid,
                            activity_name=act["activity_name"],
                            category=act["category"],
                            hours_per_month=act["hours_per_month"],
                            term=term_name,
                        )
                    )

                # Competitions
                for comp in term_data["competitions"]:
                    comp_rows.append(
                        Competition(
                            student_id=sid,
                            name=comp["name"],
                            level=comp["level"],
                            result=comp["result"],
                            date=comp["date"],
                        )
                    )

                # Development Indicators
                inds = term_data["indicators"]
                ind_rows.append(
                    DevelopmentIndicator(
                        student_id=sid,
                        term=term_name,
                        communication=inds["communication"],
                        presentation=inds["presentation"],
                        confidence=inds["confidence"],
                        participation=inds["participation"],
                        teamwork=inds["teamwork"],
                        recorded_by=teacher_id,
                    )
                )

        # Batch add records
        print(f"   Inserting {len(exam_rows)} Exam Scores...")
        db.bulk_save_objects(exam_rows)
        print(f"   Inserting {len(att_rows)} Attendance Records...")
        db.bulk_save_objects(att_rows)
        print(f"   Inserting {len(asg_rows)} Assignments...")
        db.bulk_save_objects(asg_rows)
        print(f"   Inserting {len(proj_rows)} Projects...")
        db.bulk_save_objects(proj_rows)
        print(f"   Inserting {len(act_rows)} Extracurricular Activities...")
        db.bulk_save_objects(act_rows)
        print(f"   Inserting {len(comp_rows)} Competitions...")
        db.bulk_save_objects(comp_rows)
        print(f"   Inserting {len(ind_rows)} Development Indicators...")
        db.bulk_save_objects(ind_rows)

        db.commit()

        duration = time.time() - start_time
        print("\n[OK] Seed complete in {:.2f}s!".format(duration))
        print("=" * 55)
        print("Summary of Seeded Data:")
        print(f"  * Users:        {db.query(User).count()} (1 Admin, 7 Teachers, 350 Students)")
        print(f"  * Classes:      {db.query(Class).count()}")
        print(f"  * Subjects:     {db.query(Subject).count()}")
        print(f"  * Students:     {db.query(Student).count()}")
        print(f"  * Exam Scores:  {db.query(ExamScore).count()}")
        print(f"  * Attendance:   {db.query(Attendance).count()}")
        print(f"  * Assignments:  {db.query(Assignment).count()}")
        print(f"  * Projects:     {db.query(Project).count()}")
        print(f"  * Activities:   {db.query(Extracurricular).count()}")
        print(f"  * Competitions: {db.query(Competition).count()}")
        print(f"  * Indicators:   {db.query(DevelopmentIndicator).count()}")
        print("=" * 55)
        print("Demo credentials:")
        print(f"  Admin:   admin@school.test       / {settings.DEMO_ADMIN_PASSWORD}")
        print(f"  Teacher: teacher1@school.test    / {settings.DEMO_TEACHER_PASSWORD} (..teacher7@)")
        print(f"  Student: student001@school.test  / {settings.DEMO_STUDENT_PASSWORD} (..student350@)")
        print("=" * 55)

    except Exception as e:
        db.rollback()
        print("[ERROR] Error during seed:", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
