import yaml, os
from tls_anom.utils.logging import setup_logging

class AppContext:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.logger = setup_logging(cfg.get("paths", {}).get("logs_dir", "logs"))
        for k in ["raw_dir","processed_dir","features_dir","models_dir","outputs_dir","logs_dir"]:
            os.makedirs(cfg["paths"][k], exist_ok=True)

    @staticmethod
    def _merge(a: dict, b: dict):
        for k, v in b.items():
            if isinstance(v, dict) and k in a:
                AppContext._merge(a[k], v)
            else:
                a[k] = v
        return a

    @classmethod
    def from_yaml(cls, base_path: str, overlay: str|None=None):
        with open(base_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if overlay:
            with open(overlay, "r", encoding="utf-8") as f:
                over = yaml.safe_load(f)
            cfg = cls._merge(cfg, over)
        return cls(cfg)
