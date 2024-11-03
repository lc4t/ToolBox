import sys
from pathlib import Path

from loguru import logger

# 获取项目根目录
project_dir = Path(__file__).parent.parent
# 创建日志目录
log_dir = project_dir / "logs"
log_dir.mkdir(exist_ok=True)

# 日志文件路径
log_file = log_dir / "etf_trading.log"

# 配置日志格式
fmt = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# 移除默认的控制台输出
logger.remove()

# 添加控制台输出
logger.add(
    sys.stderr,
    format=fmt,
    level="DEBUG",
    colorize=True,
)

# 添加文件输出
logger.add(
    log_file,
    format=fmt,
    level="INFO",
    rotation="100 MB",  # 当文件大于100MB时轮转
    compression="zip",  # 压缩旧日志文件
    retention="30 days",  # 保留30天的日志
    enqueue=True,  # 异步写入
)

# 导出logger实例
logger = logger.bind(name="etf_trading")
