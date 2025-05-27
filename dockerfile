FROM nvcr.io/nvidia/tritonserver:23.11-py3

WORKDIR /triton_llm

COPY ./ .

RUN pip install --no-cache-dir -r /triton_llm/requirements.txt