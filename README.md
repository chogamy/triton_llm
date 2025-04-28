# Triton example

## Env

- RTX A6000

- Ubuntu 22.04.3 LTS

## 실행

host 서버에서

```
git clone <이 레포지토리>

cd triton_llm

docker compose up -d
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