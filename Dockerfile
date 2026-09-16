#使用官方Python 3.10 精简版镜像
FROM python:3.10-slim

#设置容器内工作目录
WORKDIR /app

#安装系统依赖(libmagic 是 python-magic 的底层库,用于文件类型检测,unstructured可能用到)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

#先复制依赖文件,利用Docker曾缓存加速构建
COPY requirements.txt .
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

#复制项目全部文件到容器
COPY . .

#容器启动是运行FastAPI应用,监听8000端口
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]