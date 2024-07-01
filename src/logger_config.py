from loguru import logger

logger.remove()
logger.add("run.log", format="{time} {level} {message}", level="INFO", mode="w")

__all__ = ["logger"]
