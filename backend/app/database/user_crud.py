from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.models import User


def get_user_by_email(
    db: Session,
    email: str,
) -> User | None:

    normalized_email = email.strip().lower()

    statement = select(User).where(
        User.email == normalized_email
    )

    return db.scalars(statement).first()


def get_user_by_id(
    db: Session,
    user_id: int,
) -> User | None:

    statement = select(User).where(
        User.id == user_id
    )

    return db.scalars(statement).first()


def create_user(
    db: Session,
    email: str,
    password_hash: str,
) -> User:

    user = User(
        email=email.strip().lower(),
        password_hash=password_hash,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return user
