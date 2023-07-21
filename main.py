from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from modelscope.utils.logger import get_logger
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger as log
from pydub import AudioSegment
import io
import base64
import requests

app = FastAPI()
executor = ThreadPoolExecutor()
loaded_model = None
log.add("serving_{time}.log", level="INFO", rotation="5 MB", retention=2)

loaded_model = {"model_type": None, "model": None}
hotword_parm = {"hotword": None}


class Audio(BaseModel):
    file: str
    isurl: bool = False
    hotword: str = None
    model_type: str = "normal"


def initialize_model(model_type, hotword):
    model = None
    if model_type != "hotword" and hotword != None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    elif model_type == "normal":
        log.debug("lodding model: normal")
        loaded_model["model_type"] = "normal"
        model = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            # lm_model='damo/speech_transformer_lm_zh-cn-common-vocab8404-pytorch',
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
            timestamp_model="damo/speech_timestamp_prediction-v1-16k-offline"
        )
    elif model_type == "long":
        log.debug("lodding model: long")
        loaded_model["model_type"] = "long"
        model = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        )
    elif model_type == "whisper":
        log.debug("lodding model: UniASR")
        loaded_model["model_type"] = "UniASR"
        model = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline"
        )
    elif (model_type == "hotword" and hotword_parm["hotword"] != hotword) or hotword == None:
        log.debug("lodding model: hotword")
        loaded_model["model_type"] = "hotword"
        hotword_parm["hotword"] = hotword
        model = pipeline(
            task=Tasks.auto_speech_recognition,
            model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
            param_dict=hotword_parm
        )
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
    return model


def load_model(model_type, hotword):
    if loaded_model["model_type"] is None or loaded_model["model_type"] != model_type or (loaded_model["model_type"] == "hotword" and hotword_parm["hotword"] != hotword):
        loaded_model["model"] = initialize_model(model_type, hotword)

@app.on_event("startup")
async def startup_event():
    load_model(model_type="long", hotword=None)
    rec_result = loaded_model["model"](audio_in="./16000_001.wav")

@app.post("/asr", tags=["ASR"], summary="聚合ASR模型接口服务")
async def predict(items: Audio):
    """
    #### 说明
    1. 支持mp3和wav。服务端会自动转格式为16000采样率、单声道的wav。
    2. 模型选择。normal适用20s以下短音频。可手动切割音频文件。long适用任意长度音频文件。hotword支持热词，20s以内音频。whisper为微调后的模型，20s以内音频。
    #### 参数
    - **file**: (必需)(string) base64编码后的二进制音频文件，或音频文件的URL地址。
    - **isurl**: (bool)默认False，若为True，则会从URL下载文件后推理。
    - **hotword**: (string) 默认为None，仅当 model_type 为 hotword 时生效。接受一个字符串，每个hotword用空格隔开。 e.g."大娱号 主播 寿山石 包邮"
    - **model_type**: (string) 默认为normal。可选项：normal/long/hotword/whisper.

    """
    if not items.isurl:
        decoded_data = base64.b64decode(items.file)
        log.info("Received a file in byte.")
    else :
        log.info(f"Received a url in string, url: {items.file}")
        decoded_data = requests.get(items.file).content

    load_model(model_type="long", hotword=items.hotword)
    result = []
    try:
        rec_result = loaded_model["model"](audio_in=decoded_data)
        log.info(rec_result)
    except:
        result.append(
            {
                "text": "",
                "start": 0.0,
                "end": 0.0
            }
        )
        log.info("pass small file")
    try:
        for sentence in rec_result["sentences"]:
            result.append(
                {
                    "text": sentence["text"],
                    "start": sentence["start"] / 1000.0,
                    "end": sentence["end"] / 1000.0
                }
            )
    except:
        result.append(
            {
                "text": "",
                "start": 0.0,
                "end": 0.0
            }
        )
        log.info("mute file")
    log.info(result)
    return result

@app.get("/health")
async def health_check():
    try:
        log.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

@app.get("/health/inference")
async def health_check():
    try:
        load_model(model_type="long", hotword=None)
        rec_result = loaded_model["model"](audio_in="./16000_001.wav")
        log.info("health 200")
        return status.HTTP_200_OK

    except:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)
