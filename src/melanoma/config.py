"""Configuration loading with attribute access."""

from pathlib import Path

import yaml


class Config(dict):
    """Dictionary with recursive attribute access for nested config values."""

    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"Config key not found: {name}") from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def load_config(path):
    """Load a YAML config file into a Config object.

    Raises FileNotFoundError if the file does not exist and ValueError if the
    file does not contain a YAML mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if raw is None:
        return Config({})
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")
    return Config(raw)
