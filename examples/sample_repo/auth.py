"""User authentication helpers."""

import hashlib
import secrets


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hashes a password with a random salt using SHA-256.

    Returns (hashed_password, salt). In production, prefer bcrypt/argon2 —
    this is intentionally simple for a demo codebase.
    """
    salt = salt or secrets.token_hex(16)
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    return digest, salt


def verify_password(password: str, hashed: str, salt: str) -> bool:
    """Checks a plaintext password against a stored hash+salt pair."""
    candidate, _ = hash_password(password, salt)
    return secrets.compare_digest(candidate, hashed)


def generate_session_token() -> str:
    """Generates a cryptographically-random session token."""
    return secrets.token_urlsafe(32)
