from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.orm import Session

from backend.app.core.security import decode_access_token
from backend.app.database.database import get_db
from backend.app.database.user_crud import get_user_by_email


oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login"
)

oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl="/api/auth/login",
    auto_error=False,
)


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate authentication credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_access_token(token)

        email = payload.get("sub")

        if not email:
            raise credentials_exception

    except InvalidTokenError:
        raise credentials_exception

    user = get_user_by_email(
        db,
        email,
    )

    if user is None:
        raise credentials_exception

    return user


def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
):
    if not token:
        return None

    try:
        payload = decode_access_token(token)
        email = payload.get("sub")
        if not email:
            return None
    except InvalidTokenError:
        return None

    return get_user_by_email(db, email)
