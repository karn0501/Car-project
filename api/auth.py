"""
Phase 9: JWT Authentication & User Accounts Module.
Handles user registration, login JWT bearer token issuing, and user saved valuation history.
"""

import os
import time
import base64
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, Field


SECRET_KEY = os.getenv("JWT_SECRET_KEY", "car-prediction-jwt-secret-key-2026-super-secure")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

USER_DB_FILE = os.path.abspath("models/users_db.json")

router = APIRouter(prefix="/auth", tags=["User Authentication & Accounts"])


# ─── Auth Schemas ──────────────────────────────────────────────────────────────

class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")
    full_name: str = Field(..., description="Full name")


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_email: str


# ─── Helper Functions ──────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password using SHA256 + salt."""
    salt = "car_app_salt_2026"
    return hashlib.sha256((password + salt).encode('utf-8')).hexdigest()


def create_token(email: str) -> str:
    """Generate simple signed token string."""
    payload = {
        "sub": email,
        "exp": int(time.time()) + (TOKEN_EXPIRE_HOURS * 3600),
        "iat": int(time.time())
    }
    raw = json.dumps(payload).encode('utf-8')
    encoded = base64.b64encode(raw).decode('utf-8')
    sig = hashlib.sha256((encoded + SECRET_KEY).encode('utf-8')).hexdigest()[:16]
    return f"{encoded}.{sig}"


def verify_token(token: str) -> Optional[str]:
    """Verify signed token and return sub email if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 2:
            return None
        encoded, sig = parts[0], parts[1]
        expected_sig = hashlib.sha256((encoded + SECRET_KEY).encode('utf-8')).hexdigest()[:16]
        if sig != expected_sig:
            return None
        raw = base64.b64decode(encoded.encode('utf-8')).decode('utf-8')
        payload = json.loads(raw)
        if payload.get("exp", 0) < time.time():
            return None
        return payload.get("sub")
    except Exception:
        return None


def get_current_user_email(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency for authenticating JWT token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.replace("Bearer ", "").strip()
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return email


def _load_users() -> Dict[str, Dict[str, Any]]:
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_users(users: Dict[str, Dict[str, Any]]):
    os.makedirs(os.path.dirname(USER_DB_FILE), exist_ok=True)
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ─── Auth Endpoints ───────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
def register_user(req: UserRegisterRequest):
    users = _load_users()
    email = req.email.strip().lower()
    if email in users:
        raise HTTPException(status_code=400, detail="User email already registered")

    users[email] = {
        "email": email,
        "full_name": req.full_name,
        "password_hash": hash_password(req.password),
        "created_at": datetime.now().isoformat(),
        "valuations": []
    }
    _save_users(users)

    token = create_token(email)
    return TokenResponse(
        access_token=token,
        expires_in=TOKEN_EXPIRE_HOURS * 3600,
        user_email=email
    )


@router.post("/login", response_model=TokenResponse)
def login_user(req: UserLoginRequest):
    users = _load_users()
    email = req.email.strip().lower()
    user = users.get(email)
    if not user or user["password_hash"] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_token(email)
    return TokenResponse(
        access_token=token,
        expires_in=TOKEN_EXPIRE_HOURS * 3600,
        user_email=email
    )


@router.get("/me")
def get_user_profile(user_email: str = Depends(get_current_user_email)):
    users = _load_users()
    user = users.get(user_email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "email": user["email"],
        "full_name": user["full_name"],
        "created_at": user.get("created_at"),
        "valuation_count": len(user.get("valuations", []))
    }
