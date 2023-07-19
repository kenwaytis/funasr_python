from modelscope.hub.snapshot_download import snapshot_download

snapshot_download("damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch") # 长音频模型
snapshot_download("damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch") # 标点符号模型
snapshot_download("damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch") # 语音端点检测模型
snapshot_download("damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
