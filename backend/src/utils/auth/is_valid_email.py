import re


def is_valid_email(email: str) -> bool:
    pat = "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

    if re.match(pat, email):
        return True
    return False
