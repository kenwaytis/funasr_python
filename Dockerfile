FROM paidax/dev-containers:modelscope-v0.6

ARG HTTP_PROXY
ENV HTTP_PROXY=${HTTP_PROXY}
ENV HTTPS_PROXY=${HTTP_PROXY}

WORKDIR /home/funasr

RUN pip install --no-cache-dir \
    loguru \
    pydub \
    fastapi \
    uvicorn \
    pydantic==1.10.8 && \
    rm -rf /root/.cache/pip/*

COPY . .

RUN python download_model.py && \
    git clone https://github.com/alibaba/FunASR.git && \
    cd FunASR && \
    pip install -e ./

WORKDIR /home/funasr