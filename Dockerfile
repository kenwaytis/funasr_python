FROM registry.cn-hangzhou.aliyuncs.com/modelscope-repo/modelscope:ubuntu20.04-cuda11.3.0-py37-torch1.11.0-tf1.15.5-1.6.0
WORKDIR /modelscope/pytorch

RUN pip install --no-cache-dir \
    loguru \
    pydub \
    fastapi \
    uvicorn \
    pydantic==1.10.8 && \
    rm -rf /root/.cache/pip/*

COPY . .

RUN python download_model.py 

