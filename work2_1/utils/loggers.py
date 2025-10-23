import logging
import sys
import yaml
from utils import mkdirs

def setup_logger(name, filename, console_level=logging.WARNING):
    """设置logger的通用函数"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # 设置最低级别，让所有消息都能被处理

    # 如果已经有处理器，先清除避免重复
    if logger.handlers:
        return logger

    # 1. 文件处理器 - 记录所有级别（INFO及以上）
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setLevel(logging.INFO)  # 文件记录INFO及以上
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # 2. 控制台处理器 - 只记录WARNING及以上（错误和警告）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)  # 控制台只显示WARNING及以上
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger

with open('config.yaml', 'r') as config:
    opt = yaml.safe_load(config)
parent = f"{opt['TRAINING']['SAVE_DIR']}/{opt['MODEL']['MODE']}"
mkdirs(parent)
# 创建两个logger
result_logger = setup_logger('result_logger', f"{parent}/result.log.txt")
minio_logger = setup_logger('minio_logger', f"{parent}/minio.log.txt")
