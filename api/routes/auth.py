from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

from core.limiter import limiter
from core.logger import get_logger
from core.security import create_access_token, hash_password, verify_password
from db.session import get_supabase
from schemas.all import TokenOut, UserCreate, UserOut

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


_DUMMY_HASH = hash_password("__dummy_password_for_timing_safety__")

_WWW_AUTH = {"WWW-Authenticate": "Bearer"}


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
@limiter.limit("3/minute")
async def register(request: Request, payload: UserCreate):
    sb = get_supabase()

    try:
        result = sb.table("users").insert({
            "email": payload.email,
            "hashed_password": hash_password(payload.password),
            "full_name": payload.full_name,
            "is_active": True,

            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        if not result.data:
            logger.error("User insert returned empty data", extra={"email": payload.email})
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed. Please try again.",
            )

        user = result.data[0]
        logger.info("User registered", extra={"user_id": user["id"]})

        return UserOut(
            id=user["id"],
            email=user["email"],
            full_name=user.get("full_name"),
            created_at=user["created_at"],
        )

    except HTTPException:
        raise 

    except Exception as e:
        err_str = str(e)


        if "duplicate key" in err_str or "unique constraint" in err_str or "23505" in err_str:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists.",
            )

        logger.error("Registration error", exc_info=True, extra={"email": payload.email})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed. Please try again.",
        )


@router.post("/token", response_model=TokenOut)
@limiter.limit("5/minute")
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    sb = get_supabase()

    REQUIRED_COLS = "id, email, hashed_password, full_name, is_active"

    try:
        result = sb.table("users").select(REQUIRED_COLS).eq(
            "email", form.username
        ).execute()

        user = result.data[0] if result.data else None

        stored_hash = user["hashed_password"] if user else _DUMMY_HASH
        password_valid = verify_password(form.password, stored_hash)

        if not user or not password_valid:
            logger.warning(
                "Failed login attempt",
                extra={"email": form.username, "reason": "not_found" if not user else "bad_password"},
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials.",
                headers=_WWW_AUTH,
            )

        if not user.get("is_active", True):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive. Please contact support.",
            )

        token = create_access_token({"sub": str(user["id"])})
        logger.info("User logged in", extra={"user_id": user["id"]})

        return TokenOut(access_token=token, token_type="bearer")

    except HTTPException:
        raise

    except Exception as e:
        logger.error("Login error", exc_info=True, extra={"email": form.username})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed. Please try again.",
        )