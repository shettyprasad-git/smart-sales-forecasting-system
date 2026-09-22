from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.core.rate_limiter import rate_limit_auth
from backend.app.core.security import (
    create_access_token,
    hash_password,
    verify_password,
)
from backend.app.database.database import get_db
from backend.app.database.user_crud import (
    create_user,
    get_user_by_email,
)
from backend.app.dependencies import get_current_user
from backend.app.schemas.auth import (
    TokenResponse,
    UserCreate,
    UserResponse,
)


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(rate_limit_auth)],
)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
):
    # Reserve configured administrator email from public self-registration
    admin_email = (settings.admin_email or "admin@smart-sales.local").strip().lower()
    if user_data.email.strip().lower() == admin_email:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    existing_user = get_user_by_email(
        db,
        user_data.email,
    )

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    password_hash = hash_password(
        user_data.password
    )

    try:
        user = create_user(
            db=db,
            email=user_data.email,
            password_hash=password_hash,
        )

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    return user


@router.post(
    "/login",
    response_model=TokenResponse,
    dependencies=[Depends(rate_limit_auth)],
)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    login_identifier = (form_data.username or "").strip()

    # Resolve configured administrator username (e.g. "admin") to the internal administrator email
    admin_user = (settings.admin_username or "admin").strip().lower()
    if login_identifier.lower() == admin_user:
        lookup_email = (settings.admin_email or "admin@smart-sales.local").strip().lower()
    else:
        lookup_email = login_identifier.lower()

    user = get_user_by_email(
        db,
        lookup_email,
    )

    if user is None or not verify_password(
        form_data.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user.email,
        }
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user,
    }


@router.get(
    "/me",
    response_model=UserResponse,
)
def get_me(
    current_user=Depends(get_current_user),
):

    return current_user
