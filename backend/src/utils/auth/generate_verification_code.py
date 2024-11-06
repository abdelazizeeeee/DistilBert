import string
import secrets


def generate_verification_code(length=6):

    characters = string.ascii_uppercase + string.digits

    verification_code = "".join(secrets.choice(characters) for _ in range(length))
    return verification_code
