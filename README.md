# Triton example

## 실행

host 서버에서

```
docker build -f <YOUR_PATH>/triton_llm/dockerfile -t triton .

docker run -d --gpus '"device=0"' \
  -p 10400:8000 \
  -p 10401:8001 \
  -p 10402:8002 \
  --name triton_llm triton_llm
```


## Test

http://<YOUR_IP>:<YOUR_PORT>/v2/models/opt125m/versions/1/generate

POST 요청

```
{
    "system_prompt": "test",
    "query": "test",
    "context": "test"
}
```