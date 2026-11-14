"""
Announcements API Router.

Features:
- Role-based bulletin delivery (All, Students, Teachers, Class-specific)
- Teachers can publish announcements to their assigned class or student cohorts
- Administrators have top priority to broadcast to All, Teachers, Students, or any specific class
- Students receive class-targeted and school-wide notices with priority levels (Urgent, Important, Normal)
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.db.session import get_db
from app.db.models import User, Student, Class, Announcement
from app.core.dependencies import get_current_user
from app.schemas.announcement import AnnouncementCreate, AnnouncementResponse

router = APIRouter(prefix="/announcements", tags=["Announcements"])


@router.get("", response_model=List[AnnouncementResponse])
def get_announcements(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Retrieve announcements visible to the current authenticated user.
    - Students see 'all', 'students', and notices for their specific class.
    - Teachers see 'all', 'teachers', plus notices they authored or targeted to their class.
    - Admins see all announcements system-wide.
    """
    query = db.query(Announcement)

    if current_user.role == "student":
        student = db.query(Student).filter(Student.user_id == current_user.id).first()
        student_class_id = student.class_id if student else None
        
        conditions = [Announcement.target_audience.in_(["all", "students"])]
        if student_class_id:
            conditions.append(
                and_(
                    Announcement.target_audience == "class",
                    Announcement.class_id == student_class_id,
                )
            )
        query = query.filter(or_(*conditions))

    elif current_user.role == "teacher":
        teacher_cls = db.query(Class).filter(Class.teacher_id == current_user.id).first()
        conditions = [
            Announcement.target_audience.in_(["all", "teachers"]),
            Announcement.created_by_id == current_user.id,
        ]
        if teacher_cls:
            conditions.append(
                and_(
                    Announcement.target_audience == "class",
                    Announcement.class_id == teacher_cls.id,
                )
            )
        query = query.filter(or_(*conditions))

    # Admins see all announcements without filter
    announcements = query.order_by(Announcement.created_at.desc()).all()

    # Pre-fetch classes for class_name lookup
    classes_map = {c.id: c.name for c in db.query(Class).all()}

    results = []
    for a in announcements:
        author = a.author
        can_delete = (current_user.role == "admin") or (a.created_by_id == current_user.id)
        results.append(
            AnnouncementResponse(
                id=a.id,
                title=a.title,
                content=a.content,
                priority=a.priority,
                target_audience=a.target_audience,
                class_id=a.class_id,
                class_name=classes_map.get(a.class_id) if a.class_id else None,
                author_name=author.full_name if author else "System",
                author_role=author.role if author else "staff",
                created_at=a.created_at.strftime("%b %d, %Y %I:%M %p") if a.created_at else "",
                can_delete=can_delete,
            )
        )

    return results


@router.post("", response_model=AnnouncementResponse, status_code=status.HTTP_201_CREATED)
def create_announcement(
    data: AnnouncementCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Create a new announcement.
    - Teachers can post to their assigned class or to students.
    - Admins have full priority to broadcast to any audience (All, Students, Teachers, or Specific Class).
    """
    if current_user.role == "student":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Students cannot create announcements",
        )

    target_class_id = data.class_id
    audience = data.target_audience

    if current_user.role == "teacher":
        teacher_cls = db.query(Class).filter(Class.teacher_id == current_user.id).first()
        if not teacher_cls:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Teacher has no assigned class to announce to",
            )
        # Teachers can only target their own class or general students
        if audience not in ["class", "students"]:
            audience = "class"
        target_class_id = teacher_cls.id

    elif current_user.role == "admin":
        if audience == "class" and not target_class_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid class_id must be provided when target audience is 'class'",
            )

    announcement = Announcement(
        title=data.title,
        content=data.content,
        priority=data.priority if data.priority in ["normal", "important", "urgent"] else "normal",
        target_audience=audience,
        class_id=target_class_id,
        created_by_id=current_user.id,
    )
    db.add(announcement)
    db.commit()
    db.refresh(announcement)

    target_class = db.query(Class).filter(Class.id == target_class_id).first() if target_class_id else None

    return AnnouncementResponse(
        id=announcement.id,
        title=announcement.title,
        content=announcement.content,
        priority=announcement.priority,
        target_audience=announcement.target_audience,
        class_id=announcement.class_id,
        class_name=target_class.name if target_class else None,
        author_name=current_user.full_name or "Staff",
        author_role=current_user.role,
        created_at=announcement.created_at.strftime("%b %d, %Y %I:%M %p") if announcement.created_at else "",
        can_delete=True,
    )


@router.delete("/{announcement_id}")
def delete_announcement(
    announcement_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Delete an announcement.
    - Admins can delete any announcement.
    - Teachers can delete announcements they created.
    """
    announcement = db.query(Announcement).filter(Announcement.id == announcement_id).first()
    if not announcement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Announcement not found",
        )

    if current_user.role != "admin" and announcement.created_by_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this announcement",
        )

    db.delete(announcement)
    db.commit()
    return {"status": "success", "message": "Announcement deleted successfully", "id": announcement_id}
