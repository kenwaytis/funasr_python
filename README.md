### What is it?
Write a Restful API interface for FunASR using the Python language, running everything inside Docker.

What it does: Based on the incoming parameters, it selects different models for speech transcription.

**How to use**:
```shell
docker compose up
```
The FunASR API service is now running on port 9527. You can open **localhost:9527/docs** to view the API documentation.

When you send the first request to the server, it will automatically download the model files, which may take some time. However, this download process only occurs once.
