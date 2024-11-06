import re


def is_valid_phone_number(phone_number: str) -> bool:
    pat = "^[0-9]{8}$"

    if re.match(pat, phone_number):
        return True
    return False
