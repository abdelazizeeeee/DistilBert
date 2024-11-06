from pydantic import BaseModel, EmailStr
from fastapi_mail import ConnectionConfig


class EmailConfig(BaseModel):
    MAIL_USERNAME: str = "mail_username"
    MAIL_PASSWORD: str = "mail_password"
    MAIL_FROM: EmailStr = "email@example.com"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_STARTTLS: bool = True  # Correct field name
    MAIL_SSL_TLS: bool = False  # Correct field name
    USE_CREDENTIALS: bool = True
    VALIDATE_CERTS: bool = True


# Create an instance of EmailConfig
email_config_instance = EmailConfig()

# Use the instance's attributes to set up ConnectionConfig
email_config = ConnectionConfig(
    MAIL_USERNAME=email_config_instance.MAIL_USERNAME,
    MAIL_PASSWORD=email_config_instance.MAIL_PASSWORD,
    MAIL_FROM=email_config_instance.MAIL_FROM,
    MAIL_PORT=email_config_instance.MAIL_PORT,
    MAIL_SERVER=email_config_instance.MAIL_SERVER,
    MAIL_STARTTLS=email_config_instance.MAIL_STARTTLS,  # Corrected field name
    MAIL_SSL_TLS=email_config_instance.MAIL_SSL_TLS,  # Corrected field name
    USE_CREDENTIALS=email_config_instance.USE_CREDENTIALS,
    VALIDATE_CERTS=email_config_instance.VALIDATE_CERTS,
)
