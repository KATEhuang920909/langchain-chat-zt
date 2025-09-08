# 使用官方Python镜像作为基础
FROM python:3.11.9

# 设置工作目录
WORKDIR /app
COPY . /app
# 复制依赖文件
COPY requirements.txt .
# 设置pip源
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt


# 暴露Streamlit默认端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "app_v2.py","--server.fileWatcherType", "none","--server.headless", "true","--browser.gatherUsageStats","false","--server.address","0.0.0.0"]