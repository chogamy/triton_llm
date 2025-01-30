FROM nvcr.io/nvidia/tritonserver:23.11-py3

WORKDIR /triton_llm

COPY ./triton_llm/ .

RUN pip install --no-cache-dir -r /triton_llm/requirements.txt

CMD ["tritonserver", "--model-repository=/triton_llm/models", "--disable-auto-complete-config"]