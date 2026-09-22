from datetime import datetime
import email_validator

# Permit internal `.local` domain names (e.g. admin@smart-sales.local) for enterprise service accounts
if "local" in getattr(email_validator, "SPECIAL_USE_DOMAIN_NAMES", []):
    email_validator.SPECIAL_USE_DOMAIN_NAMES = [
        d for d in email_validator.SPECIAL_USE_DOMAIN_NAMES if d != "local"
    ]

from pydantic import BaseModel, EmailStr, Field, field_validator


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
    )

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value):
        return str(value).strip().lower()


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    created_at: datetime

    model_config = {
        "from_attributes": True,
    }


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse
