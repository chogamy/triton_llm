# Triton

## Environment

- RTX A6000, Ada6000, 3090

- Ubuntu 22.04.3 LTS

## Config

- docker-compose.yml 에서 <YOUR_MODEL_PATH> 설정!!!!! (모델 미리 다운로드 받아야 함)

## Install

host 서버에서

```
git clone https://github.com/chogamy/triton_llm.git

cd triton_llm

docker compose up -d
```

## Test

http://<YOUR_IP>:<YOUR_PORT>/v2/models/gemma-3-12b-it/versions/1/generate

POST 요청

```
{
    "system_prompt": "test",
    "query": "test",
    "context": "test"
}
```