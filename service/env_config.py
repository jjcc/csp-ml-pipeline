"""
env_config.py — Unified configuration loader.

Priority order:
  1. config.yaml (YAML file, section-flattened to UPPER_SNAKE keys)
  2. .env / OS environment variables

Template placeholders in values (e.g. {active_train_profile}) are resolved
against the top-level active_* keys in config.yaml.

Usage
-----
    from service.env_config import getenv, config

    # Simple key lookup
    val = getenv("COMMON_OUTPUT_DIR", "./output")

    # Dataset-specific config for current active_process_dataset
    ds_cfg = config.get_active_dataset_config()
    data_dir = ds_cfg["data_dir"]
"""

import os
import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load config from YAML + .env with template resolution."""

    def __init__(self, config_path: str = "config.yaml", fallback_to_env: bool = True):
        self.config_path = config_path
        self.fallback_to_env = fallback_to_env
        self._config: dict | None = None   # flat, UPPER_SNAKE keys
        self._yaml_raw: dict | None = None  # raw nested YAML for section helpers

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_yaml_raw(self) -> dict:
        """Return raw nested YAML dict (cached)."""
        if self._yaml_raw is None:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r") as fh:
                    self._yaml_raw = yaml.safe_load(fh) or {}
            else:
                self._yaml_raw = {}
        return self._yaml_raw

    def _flatten(self, d: dict, prefix: str = "", sep: str = "_") -> dict:
        """Flatten nested dict to UPPER_SNAKE keys."""
        out = {}
        for k, v in d.items():
            key = f"{prefix}{sep}{k}" if prefix else k
            if isinstance(v, dict):
                out.update(self._flatten(v, key.upper(), sep))
            else:
                out[key.upper()] = str(v) if v is not None else ""
        return out

    def _ensure_loaded(self) -> None:
        if self._config is None:
            raw = self._load_yaml_raw()
            self._config = self._flatten(raw)
            if self.fallback_to_env:
                load_dotenv(".env", override=False)

    def _resolve_template(self, value: str) -> str:
        """Replace {variable} placeholders with resolved values from config."""
        if not isinstance(value, str):
            return value
        placeholders = {
            "{active_train_profile}":   self._config.get("ACTIVE_TRAIN_PROFILE", ""),
            "{active_score_dataset}":   self._config.get("ACTIVE_SCORE_DATASET", ""),
            "{active_process_dataset}": self._config.get("ACTIVE_PROCESS_DATASET", ""),
        }
        for ph, val in placeholders.items():
            if ph in value:
                value = value.replace(ph, val)
        return value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, key: str, default=None):
        """Return value for *key* (UPPER_SNAKE), with env fallback and template resolution."""
        self._ensure_loaded()
        value = self._config.get(key)
        if value is not None:
            value = self._resolve_template(value)
        if value is None and self.fallback_to_env:
            value = os.getenv(key)
        return value if value is not None else default

    def get_section(self, section_name: str) -> dict:
        """Return all flat keys that start with SECTION_NAME_."""
        self._ensure_loaded()
        prefix = section_name.upper() + "_"
        return {k: v for k, v in self._config.items() if k.startswith(prefix)}

    def get_common_configs_raw(self) -> dict:
        """Return the raw common_configs dict from YAML."""
        return self._load_yaml_raw().get("common_configs", {})

    def get_active_dataset_config(self) -> dict:
        """Return the dataset config dict for the current active_process_dataset.

        The tag is derived dynamically from each config's data_basic_csv, so
        no hard-coded tag-to-key mapping is needed when new datasets are added.

        Returns {} if no matching config is found.
        """
        raw = self._load_yaml_raw()
        active_tag = str(raw.get("active_process_dataset", "")).strip()
        if not active_tag:
            return {}

        common_configs = raw.get("common_configs", {})
        for _key, cfg in common_configs.items():
            if isinstance(cfg, dict):
                tag = _extract_dataset_tag(cfg.get("data_basic_csv", ""))
                if tag == active_tag:
                    return cfg

        return {}

    def get_derived_file(self, basic_csv: str):
        """Derive macro_csv and output_csv from the basic_csv stem."""
        if "trades_raw_" in basic_csv:
            section = basic_csv.split("trades_raw_")[-1].split(".csv")[0]
            macro_csv = f"trades_with_gex_macro_{section}.csv"
            output_csv = f"labeled_trades_{section}.csv"
            return macro_csv, output_csv
        return None, None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _extract_dataset_tag(basic_csv: str) -> str:
    """Derive the one-word dataset tag from a basic_csv filename.

    Examples:
        trades_raw_orig.csv      -> 'orig'
        trades_raw_a_0811.csv    -> 'a'
        trades_raw_f_1027.csv    -> 'f'
    """
    stem = basic_csv.replace(".csv", "")
    parts = stem.split("_")
    try:
        idx = parts.index("raw")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


# ---------------------------------------------------------------------------
# Global singleton + convenience functions
# ---------------------------------------------------------------------------

config = ConfigLoader()


def getenv(key: str, default=None):
    """Drop-in replacement for os.getenv that checks YAML config first."""
    return config.get(key, default)


def get_derived_file(basic_csv: str):
    return config.get_derived_file(basic_csv)
