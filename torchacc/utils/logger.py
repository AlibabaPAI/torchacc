import logging
import os

logging.basicConfig()
logger = logging.getLogger("TorchAcc")
log_level = os.getenv('ACC_LOG_LEVEL', 'INFO').upper()
logger.setLevel(level=getattr(logging, log_level))

logger.propagate = False
console_handler = logging.StreamHandler()
console_handler.setLevel(level=getattr(logging, log_level))
formatter = logging.Formatter('[%(asctime)s - %(levelname)s] %(message)s')
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
