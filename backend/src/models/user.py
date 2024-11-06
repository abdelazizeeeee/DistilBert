from datetime import datetime
from pydantic import BaseModel, EmailStr
from typing import Optional
from beanie import Document


class UserSchema(BaseModel):
    username: str
    email: EmailStr


class RegisterSchema(BaseModel):
    username: str
    email: str
    password: str


class LoginSchema(BaseModel):
    email: str
    password: str


class User(Document):
    username: str
    email: EmailStr
    password: str
    verification_code: str
    is_verified: bool
