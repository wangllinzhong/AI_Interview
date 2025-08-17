import os

# 基础配置
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", 8000))
WORKERS = int(os.getenv("WORKERS", 1))

# 文件存储配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static/uploads")
REPORT_DIR = os.path.join(BASE_DIR, "static/reports")

# 允许的文件类型
ALLOWED_FILE_TYPES = ["application/pdf"]

# 最大文件大小 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024