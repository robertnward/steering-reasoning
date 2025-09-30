import os
from pathlib import Path

import yaml

path = os.environ["CONFIG_PATH"]

path = Path(path)
with path.open("r", encoding="utf-8") as fh:
    data: dict = yaml.safe_load(fh) or {}
print(data["training_setup"])
