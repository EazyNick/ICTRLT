import sys
import os
import logging
from pathlib import Path
from datetime import datetime
import glob
import colorlog

class LogManager:
    def __init__(self, directory='D:\\ICT_Project\\ICTRLT\\utils\\Log', max_files=10):
        self.directory = directory
        self.max_files = max_files
        self._timestamp = self._init_timestamp()
        self.logger = self._init_logger()
        self.clean_up_logs()

    def _init_timestamp(self):
        """ 타임스탬프 초기화 """
        return datetime.now().strftime("%Y%m%d-%H%M%S")


    def _init_logger(self):
        """ 로거 초기화 """
        logger = logging.getLogger('TEST')
        log_colors_config = {
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red', 
            'CRITICAL': 'red,bg_white'
        }

        formatter = colorlog.ColoredFormatter(
            '[%(log_color)s%(asctime)s.%(msecs)03d][%(levelname).1s][%(filename)s(%(funcName)s):%(lineno)d] %(message)s',
            log_colors=log_colors_config,
            datefmt='%Y-%m-%d %H:%M:%S',
            reset=True,
            secondary_log_colors={}
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        print(f'실행한 파일명: {os.path.basename(sys.argv[0])}')

        # 로그 저장 주소
        logfile = f"{self._timestamp}_{os.path.basename(sys.argv[0])}.log"
        logpath = Path(self.directory)

        if not logpath.exists():
            os.makedirs(logpath)

        file_handler = logging.FileHandler(logpath / logfile, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)

        logger.addHandler(file_handler)

        logger.propagate = False
        logger.setLevel(logging.DEBUG)

        logger.debug('Logger initialized')

        return logger

    def get_timestamp(self) -> str:
        """ 시작 타임스탬프 반환 """
        return self._timestamp

    def clean_up_logs(self):
        # 디렉토리 내의 특정 패턴의 파일 목록을 가져옵니다.
        files = glob.glob(os.path.join(self.directory, 'main.py.log'))

        # 파일을 생성 시간에 따라 정렬합니다.
        files.sort(key=os.path.getmtime)

        self.logger.debug('clean_up_logs')

        # 지정된 개수를 초과하는 파일이 있다면, 가장 오래된 파일부터 삭제합니다.
        while len(files) > self.max_files:
            os.remove(files.pop(0))  # 가장 오래된 파일을 삭제하고 목록에서 제거합니다.

if __name__ == "__main__":
    log_manager = LogManager()
    log_manager.logger.info("This is an info message for testing purposes.")
    log_manager.logger.debug("This is a debug message for testing purposes.")
    log_manager.logger.error("This is an error message for testing purposes.")
    print(f"Timestamp: {log_manager.get_timestamp()}")
