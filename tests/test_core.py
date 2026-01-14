"""Tests for core module."""

from signalforge.core.config import Settings, get_settings
from signalforge.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    get_password_hash,
    verify_password,
    verify_token_type,
)


def test_settings_defaults() -> None:
    """Test default settings values."""
    settings = Settings()
    assert settings.app_name == "SignalForge"
    assert settings.app_env == "development"
    assert settings.jwt_algorithm == "HS256"


def test_get_settings_cached() -> None:
    """Test that settings are cached."""
    settings1 = get_settings()
    settings2 = get_settings()
    assert settings1 is settings2


def test_password_hashing() -> None:
    """Test password hashing and verification."""
    password = "testpassword123"
    hashed = get_password_hash(password)
    assert hashed != password
    assert verify_password(password, hashed)
    assert not verify_password("wrongpassword", hashed)


def test_access_token_creation() -> None:
    """Test access token creation and decoding."""
    data = {"sub": "user123"}
    token = create_access_token(data)
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user123"
    assert payload["type"] == "access"


def test_refresh_token_creation() -> None:
    """Test refresh token creation and decoding."""
    data = {"sub": "user123"}
    token = create_refresh_token(data)
    payload = decode_token(token)
    assert payload is not None
    assert payload["sub"] == "user123"
    assert payload["type"] == "refresh"


def test_verify_token_type() -> None:
    """Test token type verification."""
    access_token = create_access_token({"sub": "user123"})
    refresh_token = create_refresh_token({"sub": "user123"})

    access_payload = decode_token(access_token)
    refresh_payload = decode_token(refresh_token)

    assert access_payload is not None
    assert refresh_payload is not None

    assert verify_token_type(access_payload, "access")
    assert not verify_token_type(access_payload, "refresh")
    assert verify_token_type(refresh_payload, "refresh")
    assert not verify_token_type(refresh_payload, "access")


def test_decode_invalid_token() -> None:
    """Test decoding invalid token."""
    payload = decode_token("invalid.token.here")
    assert payload is None
