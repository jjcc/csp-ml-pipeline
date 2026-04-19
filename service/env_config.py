"""
env_config.py — Unified configuration loader.

Priority order:
  1. config.yaml (YAML file, section-flattened to UPPER_SNAKE keys)
  2. .env / OS environment variables

Template placeholders in path values are resolved automatically:

  {active_score_date}        — events_start_date of dataset as YYYYMMDD
                                (e.g. "20251027" for events_start_date: 2025-10-27)
                                Used in all output filenames so they are self-documenting.
  {active_score_labeled_csv} — output_csv filename from dataset config
                                (e.g. "labeled_trades_current.csv")
  {rolling_window_weeks}     — integer rolling window size from config
  {model_type}               — winner.model_type (e.g. "lgbm")

Usage
-----
    from service.env_config import getenv, config

    # Simple key lookup
    val = getenv("ROLLING_WINDOW_WEEKS", "14")

    # Dataset config (single dataset, no batch tags)
    ds_cfg = config.get_active_dataset_config()

    # Rolling training window is applied in a05_merge_datasets.py by
    # date-filtering the single labeled CSV (no batch list needed).
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

    def _compute_score_date(self, raw: dict) -> str:
        """Return events_start_date of the dataset as YYYYMMDD string.

        Reads from the single `dataset:` block in config.yaml.
        Falls back to empty string if not found.
        """
        date_str = raw.get("dataset", {}).get("events_start_date", "")
        return date_str.replace("-", "") if date_str else ""

    def _compute_score_labeled_csv(self, raw: dict) -> str:
        """Return the output_csv filename from the dataset config.

        E.g. "labeled_trades_current.csv"
        """
        return raw.get("dataset", {}).get("output_csv", "")

    def _ensure_loaded(self) -> None:
        if self._config is None:
            raw = self._load_yaml_raw()
            self._config = self._flatten(raw)
            # Inject computed derived keys so templates can reference them
            self._config["ACTIVE_SCORE_DATE"]       = self._compute_score_date(raw)
            self._config["ACTIVE_SCORE_LABELED_CSV"] = self._compute_score_labeled_csv(raw)
            # Kept for any legacy template references; always "current" in single-dataset mode
            self._config.setdefault("ACTIVE_SCORE_DATASET",   "current")
            self._config.setdefault("ACTIVE_PROCESS_DATASET", "current")
            if self.fallback_to_env:
                load_dotenv(".env", override=False)

    def _resolve_template(self, value: str) -> str:
        """Replace {variable} placeholders with resolved config values."""
        if not isinstance(value, str):
            return value
        placeholders = {
            "{active_score_dataset}":    self._config.get("ACTIVE_SCORE_DATASET", ""),
            "{active_process_dataset}":  self._config.get("ACTIVE_PROCESS_DATASET", ""),
            "{active_score_date}":       self._config.get("ACTIVE_SCORE_DATE", ""),
            "{active_score_labeled_csv}": self._config.get("ACTIVE_SCORE_LABELED_CSV", ""),
            "{data_dir}":                self._config.get("DATASET_DATA_DIR", ""),
            "{rolling_window_weeks}":    self._config.get("ROLLING_WINDOW_WEEKS", "14"),
            "{model_type}":              self._config.get("WINNER_MODEL_TYPE", "lgbm"),
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
        """Return a dict with the single dataset config.

        Returns {"current": dataset_cfg} for backward compatibility with any
        code that iterates get_common_configs_raw().items().  With the
        single-dataset design there is always exactly one entry.
        """
        ds = self._load_yaml_raw().get("dataset", {})
        return {"current": ds} if ds else {}

    def get_active_dataset_config(self) -> dict:
        """Return the single dataset config from config.yaml.

        In single-dataset mode there is no active_process_dataset tag —
        every pipeline step operates on the one `dataset:` block.

        Returns {} if the dataset block is missing from config.
        """
        return self._load_yaml_raw().get("dataset", {})

    def get_score_dataset_config(self) -> dict:
        """Return the single dataset config (same as get_active_dataset_config)."""
        return self._load_yaml_raw().get("dataset", {})

    def get_score_date(self) -> str:
        """Return events_start_date of the dataset as YYYYMMDD (e.g. '20251027')."""
        self._ensure_loaded()
        return self._config.get("ACTIVE_SCORE_DATE", "")

    def get_rolling_train_batches(self) -> list:
        """Not used in single-dataset mode.

        Rolling window date filtering is done directly in a05_merge_datasets.py
        by slicing the single labeled CSV on tradeTime.  This method returns []
        so any legacy callers degrade gracefully rather than crashing.
        """
        return []

    def get_derived_file(self, basic_csv: str):
        """Derive macro_csv and output_csv from the basic_csv stem."""
        if "trades_raw_" in basic_csv:
            section = basic_csv.split("trades_raw_")[-1].split(".csv")[0]
            macro_csv = f"trades_with_gex_macro_{section}.csv"
            output_csv = f"labeled_trades_{section}.csv"
            return macro_csv, output_csv
        return None, None


# ---------------------------------------------------------------------------
# Global singleton + convenience functions
# ---------------------------------------------------------------------------

config = ConfigLoader()


def getenv(key: str, default=None):
    """Drop-in replacement for os.getenv that checks YAML config first."""
    return config.get(key, default)


def get_derived_file(basic_csv: str):
    return config.get_derived_file(basic_csv)
