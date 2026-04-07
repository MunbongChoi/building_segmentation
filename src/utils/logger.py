"""로깅 설정"""
import sys
from pathlib import Path
from loguru import logger as _logger


def setup_logger(
    log_dir: str = "./outputs/logs",
    log_level: str = "INFO",
    name: str = "pipeline"
) -> None:
    """로거 설정
    
    Args:
        log_dir: 로그 파일 저장 디렉토리
        log_level: 로그 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        name: 로거 이름
    """
    # 디렉토리 생성
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 기존 핸들러 제거
    _logger.remove()
    
    # 콘솔 출력 핸들러
    _logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )
    
    # 파일 출력 핸들러
    _logger.add(
        f"{log_dir}/{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="100 MB",
        retention="7 days",
    )
    
    _logger.info(f"Logger initialized: {name} (level={log_level})")


def get_logger(name: str = __name__):
    """로거 인스턴스 반환"""
    return _logger.bind(name=name)


# 기본 로거
logger = get_logger("building_seg")
