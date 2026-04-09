"""
User management routes (admin only in production — no auth guard added here
to keep it simple; add a Depends(require_admin) decorator when ready).

GET    /users          → list all users
POST   /users          → create user
DELETE /users/{id}     → soft-delete user
"""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_dep
from app.models import OrixUser
from app.routes.auth import hash_password

router = APIRouter(prefix="/users", tags=["users"])


# ── Schemas ────────────────────────────────────────────────────────────────────

class UserOut(BaseModel):
    id: str
    username: str
    role: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("", response_model=List[UserOut])
async def list_users(
    db: AsyncSession = Depends(get_db_dep),
) -> List[UserOut]:
    result = await db.execute(
        select(OrixUser).where(OrixUser.active == True).order_by(OrixUser.username)
    )
    return [
        UserOut(id=str(u.id), username=u.username, role=u.role)
        for u in result.scalars().all()
    ]


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db_dep),
) -> UserOut:
    # Check duplicate
    existing = await db.execute(
        select(OrixUser).where(OrixUser.username == body.username)
    )
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Username already exists")

    if body.role not in ("admin", "operator", "user"):
        raise HTTPException(status_code=422, detail="Invalid role")

    user = OrixUser(
        username=body.username,
        hashed_password=hash_password(body.password),
        role=body.role,
    )
    db.add(user)
    await db.flush()

    return UserOut(id=str(user.id), username=user.username, role=user.role)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db_dep),
) -> None:
    result = await db.execute(
        select(OrixUser).where(OrixUser.id == uuid.UUID(user_id))
    )
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.active = False
