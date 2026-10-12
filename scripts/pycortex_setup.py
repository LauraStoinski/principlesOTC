"""Use the project pycortex database and workspace-local configuration/cache."""
import os
from pathlib import Path
from unittest.mock import patch

rootdir = Path(__file__).resolve().parents[1]
configdir = rootdir / "results" / "pycortex_config"
configdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CONFIG_HOME", str(configdir))

# This pycortex version hardcodes the macOS user configuration location.
expanduser = os.path.expanduser

def local_config_path(path):
    if path == "~/Library/Application Support/":
        return str(configdir)
    return expanduser(path)

with patch("os.path.expanduser", local_config_path):
    import cortex

filestore = str(rootdir / "data" / "pycortex" / "db")
cortex.options.config.set("basic", "filestore", filestore)
cortex.options.config.set("basic", "cache", str(rootdir / "results" / "pycortex_cache"))
cortex.options.config.set("webgl", "colormaps", str(rootdir / "data" / "pycortex" / "colormaps"))
cortex.db.filestore = filestore
