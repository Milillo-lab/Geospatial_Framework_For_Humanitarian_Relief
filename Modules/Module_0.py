## 0. Necessities
Run the below lying code (Without "#") to download all of the necessities at once.
# !pip install -r requirements.txt
Run the code snippet below once, before starting your work every time. This will allow the code to access configurations.
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