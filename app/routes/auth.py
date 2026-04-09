"""
Authentication routes.

POST /auth/login   → returns {user: {id, username, role, token}}
POST /auth/logout  → best-effort invalidation (stateless JWT)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db_dep
from app.models import OrixUser

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Crypto ─────────────────────────────────────────────────────────────────────
_pwd_ctx = CryptContext(schemes=["sha256_crypt"], deprecated="auto")

ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    return jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])


# ── Schemas ────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: str
    username: str
    role: str
    token: str


class LoginResponse(BaseModel):
    user: UserOut


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/login", response_model=LoginResponse)
async def login(
    body: LoginRequest,
    db: AsyncSession = Depends(get_db_dep),
) -> LoginResponse:
    result = await db.execute(
        select(OrixUser).where(
            OrixUser.username == body.username,
            OrixUser.active == True,
        )
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token(
        {"sub": str(user.id), "username": user.username, "role": user.role}
    )

    return LoginResponse(
        user=UserOut(
            id=str(user.id),
            username=user.username,
            role=user.role,
            token=token,
        )
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout() -> None:
    """
    Stateless JWT — client discards the token.
    Extend with a token blacklist (Redis SET with TTL) for strict revocation.
    """
    pass
