import re
import secrets
import string
from fastapi_mail import FastMail, MessageSchema
from src.models.email import email_config


async def send_verification_email(email: str, verification_code: str):
    message = MessageSchema(
        subject="Your Verification Code",
        recipients=[email],
        body=f"Your verification code is {verification_code}.",
        subtype="plain",
    )

    fm = FastMail(email_config)
    await fm.send_message(message)
