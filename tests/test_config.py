"""Tests for melanoma.config."""

import pytest

from melanoma.config import Config, load_config


def write_config(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_nested_attribute_and_item_access(tmp_path):
    config = load_config(write_config(tmp_path, "a:\n  b: 1\nnum: 2\nname: run\n"))
    assert config.num == 2
    assert config.name == "run"
    assert config.a.b == 1
    assert config["a"]["b"] == 1


def test_config_is_dict(tmp_path):
    config = load_config(write_config(tmp_path, "a: 1\n"))
    assert isinstance(config, dict)
    assert isinstance(config, Config)


def test_get_returns_default(tmp_path):
    config = load_config(write_config(tmp_path, "a: 1\n"))
    assert config.get("missing", "fallback") == "fallback"
    assert config.get("a") == 1


def test_missing_key_raises_attribute_error(tmp_path):
    config = load_config(write_config(tmp_path, "a: 1\n"))
    with pytest.raises(AttributeError):
        _ = config.does_not_exist


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/config.yaml")


def test_empty_file_is_empty_config(tmp_path):
    config = load_config(write_config(tmp_path, ""))
    assert dict(config) == {}
    with pytest.raises(AttributeError):
        _ = config.anything


def test_non_mapping_yaml_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, "- 1\n- 2\n"))


def test_scalar_yaml_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_config(tmp_path, "just a string\n"))
