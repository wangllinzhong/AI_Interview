FROM datajoint/miniconda3:22.11.1-py3.11-debian

USER root

WORKDIR /AI-Interview-copy02

# 复制环境配置文件
COPY requirements.txt requirements.txt

# 更新base环境而不是创建新环境
RUN pip install -r requirements.txt

# 设置默认命令
CMD ["python", "main.py"]