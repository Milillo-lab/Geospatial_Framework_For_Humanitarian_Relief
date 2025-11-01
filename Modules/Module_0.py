# !pip install -r requirements.txt

# Configuration File Load

import yaml

with open("config.yaml", "r", encoding="utf-8") as _f:
    CFG = yaml.safe_load(_f) or {}

class ConfigError(Exception):
    pass

def require(key, cast=None):
    if key not in CFG:
        raise ConfigError(f"Missing required config key: {key}")
    val = CFG[key]
    try:
        return cast(val) if cast else val
    except Exception as e:
        raise ConfigError(f"Bad type for key '{key}': {val!r} ({e})")

def optional(key, cast=None):
    if key not in CFG:
        return None
    val = CFG[key]
    return cast(val) if cast else val