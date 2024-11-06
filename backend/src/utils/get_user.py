from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv
from ..config.settings import settings
from fastapi import APIRouter, HTTPException
import hashlib

load_dotenv()

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key")


def get_user(
    api_key_header: str = Security(api_key_header),
):
    if check_api_key(api_key_header):
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key"
    )


def check_api_key(api_key: str):
    env_api_key_hashed = settings.API_KEY
    provided_api_key_hashed = hash_api_key(api_key)

    if provided_api_key_hashed == env_api_key_hashed:
        return True
    return False


def hash_api_key(api_key: str) -> str:
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    return hashed_key
