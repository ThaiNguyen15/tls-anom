import logging, os

def setup_logging(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("tls-anom")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(log_dir, "app.log"))
        ch = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger
