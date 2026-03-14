"""
Tests for Rithmic account configuration and encryption helpers in
``lib.integrations.rithmic_client``.

Covers:
  1. ``RithmicAccountConfig`` creation with defaults
  2. ``set_credentials`` + ``get_username`` / ``get_password`` round-trip (encryption)
  3. ``to_storage_dict`` / ``from_storage_dict`` round-trip
  4. ``to_ui_dict`` masks credentials
  5. ``_derive_fernet_key`` is deterministic
  6. ``_encrypt`` / ``_decrypt`` round-trip
  7. ``RithmicAccountManager`` initialisation (mock Redis)
  8. ``get_all_ui`` returns list of account dicts

All tests are self-contained — no Redis, no network, no real credentials.
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

# Ensure Redis is disabled before any transitive imports
os.environ.setdefault("DISABLE_REDIS", "1")

# Set a stable encryption key so Fernet derivation is deterministic across runs
os.environ["SECRET_KEY"] = "test-secret-key-for-rithmic-tests"

from lib.integrations.rithmic_client import (  # noqa: E402
    RithmicAccountConfig,
    RithmicAccountManager,
    _decrypt,
    _derive_fernet_key,
    _encrypt,
)

# ---------------------------------------------------------------------------
# 1. RithmicAccountConfig creation with defaults
# ---------------------------------------------------------------------------


class TestAccountConfigDefaults:
    def test_default_prop_firm(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.prop_firm == "tpt"

    def test_default_gateway(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.gateway == "Chicago"

    def test_default_enabled(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.enabled is True

    def test_default_app_name(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.app_name == "ruby_futures"

    def test_default_app_version(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.app_version == "1.0"

    def test_default_account_size(self) -> None:
        cfg = RithmicAccountConfig(key="test1", label="Test Account")
        assert cfg.account_size == 150_000

    def test_key_and_label_stored(self) -> None:
        cfg = RithmicAccountConfig(key="mykey", label="My Label")
        assert cfg.key == "mykey"
        assert cfg.label == "My Label"

    def test_system_name_from_preset(self) -> None:
        cfg = RithmicAccountConfig(key="t1", label="TPT", prop_firm="tpt")
        assert cfg.system_name == "Rithmic Paper Trading"

    def test_system_name_explicit_override(self) -> None:
        cfg = RithmicAccountConfig(key="t1", label="Custom", prop_firm="tpt", system_name="Override")
        assert cfg.system_name == "Override"


# ---------------------------------------------------------------------------
# 2. set_credentials + get_username / get_password round-trip
# ---------------------------------------------------------------------------


class TestCredentialRoundTrip:
    def test_set_and_get_username(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg.set_credentials("my_user", "my_pass")
        assert cfg.get_username() == "my_user"

    def test_set_and_get_password(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg.set_credentials("my_user", "my_pass")
        assert cfg.get_password() == "my_pass"

    def test_credentials_stored_encrypted(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg.set_credentials("plainuser", "plainpass")
        # The internal encrypted values should NOT be the plaintext
        assert cfg._username_enc != "plainuser"
        assert cfg._password_enc != "plainpass"
        # But they should be non-empty
        assert len(cfg._username_enc) > 0
        assert len(cfg._password_enc) > 0

    def test_empty_credentials(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg.set_credentials("", "")
        assert cfg.get_username() == ""
        assert cfg.get_password() == ""

    def test_special_characters_in_credentials(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        username = "user@domain.com"
        password = "p@$$w0rd!#%^&*()"
        cfg.set_credentials(username, password)
        assert cfg.get_username() == username
        assert cfg.get_password() == password

    def test_unicode_credentials(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg.set_credentials("ユーザー", "パスワード")
        assert cfg.get_username() == "ユーザー"
        assert cfg.get_password() == "パスワード"


# ---------------------------------------------------------------------------
# 3. to_storage_dict / from_storage_dict round-trip
# ---------------------------------------------------------------------------


class TestStorageDictRoundTrip:
    def test_round_trip_preserves_key(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("user1", "pass1")
        d = cfg.to_storage_dict()
        restored = RithmicAccountConfig.from_storage_dict(d)
        assert restored.key == cfg.key

    def test_round_trip_preserves_label(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="My Account")
        d = cfg.to_storage_dict()
        restored = RithmicAccountConfig.from_storage_dict(d)
        assert restored.label == "My Account"

    def test_round_trip_preserves_credentials(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("round_user", "round_pass")
        d = cfg.to_storage_dict()
        restored = RithmicAccountConfig.from_storage_dict(d)
        assert restored.get_username() == "round_user"
        assert restored.get_password() == "round_pass"

    def test_round_trip_preserves_all_fields(self) -> None:
        cfg = RithmicAccountConfig(
            key="acc2",
            label="Full Config",
            prop_firm="apex",
            system_name="Apex Trader Funding",
            gateway="Chicago",
            enabled=False,
            app_name="custom_app",
            app_version="2.0",
            account_size=50_000,
        )
        cfg.set_credentials("apex_user", "apex_pass")

        d = cfg.to_storage_dict()
        restored = RithmicAccountConfig.from_storage_dict(d)

        assert restored.key == "acc2"
        assert restored.label == "Full Config"
        assert restored.prop_firm == "apex"
        assert restored.gateway == "Chicago"
        assert restored.enabled is False
        assert restored.app_name == "custom_app"
        assert restored.app_version == "2.0"
        assert restored.account_size == 50_000
        assert restored.get_username() == "apex_user"
        assert restored.get_password() == "apex_pass"

    def test_storage_dict_contains_encrypted_creds(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("user1", "pass1")
        d = cfg.to_storage_dict()
        assert "username_enc" in d
        assert "password_enc" in d
        # The stored values are encrypted, not plaintext
        assert d["username_enc"] != "user1"
        assert d["password_enc"] != "pass1"

    def test_storage_dict_is_json_serializable(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("user1", "pass1")
        d = cfg.to_storage_dict()
        # Should not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # And should round-trip through JSON
        parsed = json.loads(serialized)
        restored = RithmicAccountConfig.from_storage_dict(parsed)
        assert restored.get_username() == "user1"


# ---------------------------------------------------------------------------
# 4. to_ui_dict masks credentials
# ---------------------------------------------------------------------------


class TestUiDictMasking:
    def test_ui_dict_has_no_plaintext_credentials(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("my_secret_user", "my_secret_pass")
        ui = cfg.to_ui_dict()
        # Should not contain the actual credential values
        assert "my_secret_user" not in json.dumps(ui)
        assert "my_secret_pass" not in json.dumps(ui)

    def test_ui_dict_has_username_set_flag(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("user1", "pass1")
        ui = cfg.to_ui_dict()
        assert ui["username_set"] is True
        assert ui["password_set"] is True

    def test_ui_dict_flags_false_when_no_creds(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        ui = cfg.to_ui_dict()
        assert ui["username_set"] is False
        assert ui["password_set"] is False

    def test_ui_dict_has_username_hint(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("johndoe", "secret")
        ui = cfg.to_ui_dict()
        assert "username_hint" in ui
        # Hint should show first 3 chars + ***
        assert ui["username_hint"] == "joh***"

    def test_ui_dict_short_username_hint(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test")
        cfg.set_credentials("ab", "pass")
        ui = cfg.to_ui_dict()
        # For short usernames (<=4 chars), show first char + ***
        assert ui["username_hint"] == "a***"

    def test_ui_dict_has_expected_keys(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test", prop_firm="tpt")
        ui = cfg.to_ui_dict()
        expected_keys = {
            "key",
            "label",
            "prop_firm",
            "prop_firm_label",
            "system_name",
            "gateway",
            "enabled",
            "app_name",
            "app_version",
            "account_size",
            "username_set",
            "password_set",
            "username_hint",
        }
        assert expected_keys.issubset(set(ui.keys()))

    def test_ui_dict_prop_firm_label_resolved(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Test", prop_firm="tpt")
        ui = cfg.to_ui_dict()
        assert ui["prop_firm_label"] == "Take Profit Trader (TPT)"


# ---------------------------------------------------------------------------
# 5. _derive_fernet_key is deterministic
# ---------------------------------------------------------------------------


class TestDeriveFernetKey:
    def test_deterministic(self) -> None:
        key1 = _derive_fernet_key()
        key2 = _derive_fernet_key()
        assert key1 == key2

    def test_returns_bytes(self) -> None:
        key = _derive_fernet_key()
        assert isinstance(key, bytes)

    def test_key_length_valid_for_fernet(self) -> None:
        # Fernet requires a 32-byte key, URL-safe base64 encoded = 44 bytes
        key = _derive_fernet_key()
        assert len(key) == 44

    def test_different_secret_produces_different_key(self) -> None:
        key1 = _derive_fernet_key()

        # Temporarily change the secret key used by the module
        import lib.integrations.rithmic_client as rc

        original = rc._SECRET_KEY
        try:
            rc._SECRET_KEY = "a-completely-different-secret"
            key2 = _derive_fernet_key()
        finally:
            rc._SECRET_KEY = original

        assert key1 != key2


# ---------------------------------------------------------------------------
# 6. _encrypt / _decrypt round-trip
# ---------------------------------------------------------------------------


class TestEncryptDecrypt:
    def test_round_trip_simple(self) -> None:
        plaintext = "hello world"
        encrypted = _encrypt(plaintext)
        decrypted = _decrypt(encrypted)
        assert decrypted == plaintext

    def test_round_trip_empty_string(self) -> None:
        assert _encrypt("") == ""
        assert _decrypt("") == ""

    def test_encrypted_differs_from_plaintext(self) -> None:
        plaintext = "sensitive data"
        encrypted = _encrypt(plaintext)
        assert encrypted != plaintext

    def test_encrypted_is_string(self) -> None:
        encrypted = _encrypt("test")
        assert isinstance(encrypted, str)

    def test_round_trip_long_string(self) -> None:
        plaintext = "A" * 10_000
        encrypted = _encrypt(plaintext)
        decrypted = _decrypt(encrypted)
        assert decrypted == plaintext

    def test_round_trip_special_characters(self) -> None:
        plaintext = 'p@$$w0rd!#%^&*()_+-={}[]|\\:";<>?,./'
        encrypted = _encrypt(plaintext)
        decrypted = _decrypt(encrypted)
        assert decrypted == plaintext

    def test_round_trip_unicode(self) -> None:
        plaintext = "日本語テスト 🔑"
        encrypted = _encrypt(plaintext)
        decrypted = _decrypt(encrypted)
        assert decrypted == plaintext

    def test_each_encryption_produces_different_ciphertext(self) -> None:
        # Fernet uses a random IV, so encrypting the same value twice
        # should produce different ciphertexts.  When cryptography is not
        # installed the code falls back to plain base64 which is
        # deterministic — skip the uniqueness check in that case.
        plaintext = "same input"
        enc1 = _encrypt(plaintext)
        enc2 = _encrypt(plaintext)

        try:
            from cryptography.fernet import Fernet  # noqa: F401

            has_fernet = True
        except ImportError:
            has_fernet = False

        if has_fernet:
            assert enc1 != enc2
        else:
            # base64 fallback is deterministic — just verify round-trip
            assert enc1 == enc2

        # Both must decrypt to the original value regardless
        assert _decrypt(enc1) == plaintext
        assert _decrypt(enc2) == plaintext


# ---------------------------------------------------------------------------
# 7. RithmicAccountManager initialisation (mock Redis)
# ---------------------------------------------------------------------------


class TestAccountManagerInit:
    def test_manager_creates_empty_configs(self) -> None:
        mgr = RithmicAccountManager()
        assert mgr._configs == {}
        assert mgr._status == {}
        assert mgr._loaded is False

    def test_reload_configs_sets_loaded_flag(self) -> None:
        mgr = RithmicAccountManager()

        with patch(
            "lib.integrations.rithmic_client._load_configs",
            return_value=[],
        ):
            mgr.reload_configs()

        assert mgr._loaded is True

    def test_reload_configs_loads_from_redis(self) -> None:
        cfg1 = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg1.set_credentials("user1", "pass1")
        cfg2 = RithmicAccountConfig(key="acc2", label="Account 2")

        with (
            patch(
                "lib.integrations.rithmic_client._load_configs",
                return_value=[cfg1, cfg2],
            ),
            patch(
                "lib.integrations.rithmic_client._load_status",
                return_value={},
            ),
        ):
            mgr = RithmicAccountManager()
            mgr.reload_configs()

        assert "acc1" in mgr._configs
        assert "acc2" in mgr._configs
        assert len(mgr._configs) == 2

    def test_reload_configs_restores_cached_status(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1")
        cached_status = {"key": "acc1", "connected": True, "pnl": 500.0}

        with (
            patch(
                "lib.integrations.rithmic_client._load_configs",
                return_value=[cfg],
            ),
            patch(
                "lib.integrations.rithmic_client._load_status",
                return_value=cached_status,
            ),
        ):
            mgr = RithmicAccountManager()
            mgr.reload_configs()

        assert mgr._status["acc1"] == cached_status

    def test_get_status_unknown_key(self) -> None:
        mgr = RithmicAccountManager()
        status = mgr.get_status("nonexistent")
        assert status["key"] == "nonexistent"
        assert status["connected"] is False


# ---------------------------------------------------------------------------
# 8. get_all_ui returns list of account dicts
# ---------------------------------------------------------------------------


class TestGetAllUi:
    def test_returns_list(self) -> None:
        cfg1 = RithmicAccountConfig(key="acc1", label="Account 1")
        cfg1.set_credentials("user1", "pass1")
        cfg2 = RithmicAccountConfig(key="acc2", label="Account 2")

        with (
            patch(
                "lib.integrations.rithmic_client._load_configs",
                return_value=[cfg1, cfg2],
            ),
            patch(
                "lib.integrations.rithmic_client._load_status",
                return_value={},
            ),
        ):
            mgr = RithmicAccountManager()
            result = mgr.get_all_ui()

        assert isinstance(result, list)
        assert len(result) == 2

    def test_each_entry_is_ui_dict(self) -> None:
        cfg = RithmicAccountConfig(key="acc1", label="Account 1", prop_firm="tpt")
        cfg.set_credentials("myuser", "mypass")

        with (
            patch(
                "lib.integrations.rithmic_client._load_configs",
                return_value=[cfg],
            ),
            patch(
                "lib.integrations.rithmic_client._load_status",
                return_value={},
            ),
        ):
            mgr = RithmicAccountManager()
            result = mgr.get_all_ui()

        entry = result[0]
        assert entry["key"] == "acc1"
        assert entry["label"] == "Account 1"
        assert entry["username_set"] is True
        assert entry["password_set"] is True
        assert entry["username_hint"] == "myu***"
        # Plaintext must NEVER appear
        assert "myuser" not in json.dumps(entry)
        assert "mypass" not in json.dumps(entry)

    def test_returns_empty_list_when_no_accounts(self) -> None:
        with patch(
            "lib.integrations.rithmic_client._load_configs",
            return_value=[],
        ):
            mgr = RithmicAccountManager()
            result = mgr.get_all_ui()

        assert result == []

    def test_lazy_loads_configs_on_first_call(self) -> None:
        """get_all_ui should call reload_configs if not yet loaded."""
        cfg = RithmicAccountConfig(key="acc1", label="Lazy Load")

        with (
            patch(
                "lib.integrations.rithmic_client._load_configs",
                return_value=[cfg],
            ),
            patch(
                "lib.integrations.rithmic_client._load_status",
                return_value={},
            ),
        ):
            mgr = RithmicAccountManager()
            assert mgr._loaded is False
            result = mgr.get_all_ui()
            assert mgr._loaded is True
            assert len(result) == 1
