# 使用官方的 python:3.12.3 鏡像作為基礎鏡像
FROM python:3.12.3

# 將 aopengaze 目錄中的所有文件複製到工作目錄
COPY ./opengaze /home/opengaze

# 設置工作目錄
WORKDIR /home/opengaze

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx

# 安裝必要套件
RUN pip install -r requirements.txt