"""Authentication schemas."""

from uuid import UUID

from pydantic import BaseModel, EmailStr, Field

from signalforge.models.user import UserType


class UserRegister(BaseModel):
    """Schema for user registration."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    user_type: UserType = UserType.CASUAL_OBSERVER


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Schema for token response."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request."""

    refresh_token: str


class UserResponse(BaseModel):
    """Schema for user response."""

    id: UUID
    email: str
    username: str
    user_type: UserType
    is_active: bool
    is_verified: bool

    model_config = {"from_attributes": True}
