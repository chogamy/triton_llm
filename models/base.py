import gc
import torch
import triton_python_backend_utils as pb_utils
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)
import numpy as np
import json
import logging

# gemma3에서는 이렇게 해야 한다고 함: https://jimmy-ai.tistory.com/521
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class Base:
    def initialize(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, use_fast=True, trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{self.model_instance_device_id}"},
            trust_remote_code=True,
        )

    def execute(self, requests):
        responses = []

        for request in requests:
            original_generation_config = self.model.generation_config
            
            do_sample = pb_utils.get_input_tensor_by_name(request, "do_sample")
            max_new_tokens = pb_utils.get_input_tensor_by_name(
                request, "max_new_tokens"
            )
            top_k = pb_utils.get_input_tensor_by_name(request, "top_k")
            top_p = pb_utils.get_input_tensor_by_name(request, "top_p")
            temperature = pb_utils.get_input_tensor_by_name(request, "temperature")

            do_sample = None if do_sample is None else do_sample.as_numpy()
            max_new_tokens = None if max_new_tokens is None else max_new_tokens.as_numpy()
            top_k = None if top_k is None else top_k.as_numpy()
            top_p = None if top_p is None else top_p.as_numpy()
            temperature = None if temperature is None else temperature.as_numpy()
            
            do_sample = True if do_sample is None else do_sample[0]
            max_new_tokens = None if max_new_tokens is None else max_new_tokens[0]
            top_k = None if top_k is None else top_k[0]
            top_p = None if top_p is None else top_p[0]
            temperature = None if temperature is None else temperature[0]
                
            # do_sample = bool(do_sample) # 없어도 되지 않나
            max_new_tokens = 1500 if max_new_tokens is None else int(max_new_tokens)
            top_k = None if top_k is None else int(top_k)
            top_p = None if top_p is None else float(top_p)
            temperature = None if temperature is None else float(temperature)

            decoder = np.vectorize(lambda x: x.decode("UTF-8"))

            prompt_tensor = pb_utils.get_input_tensor_by_name(
                request, "system_prompt"
            ).as_numpy()
            query_tensor = pb_utils.get_input_tensor_by_name(
                request, "query"
            ).as_numpy()
            context_tensor = pb_utils.get_input_tensor_by_name(
                request, "context"
            ).as_numpy()
            prompt = decoder(prompt_tensor)[0][0]
            query = decoder(query_tensor)[0][0]
            context = decoder(context_tensor)[0][0]


            # # 요청 로그
            logging.warning(f"REQUEST: [do_sample: {do_sample} / top_p: {top_p} / top_k: {top_k} / temperature: {temperature}]")


            messages = [
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"{context}\n{query}"},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            input_tokens = inputs.shape[1]

            self.model.generation_config.do_sample = do_sample
            self.model.generation_config.max_new_tokens = max_new_tokens
            if top_k is not None:
                self.model.generation_config.top_k = top_k
            if top_p is not None:
                self.model.generation_config.top_p = top_p
            if temperature is not None:
                self.model.generation_config.temperature = temperature

            # 넣은 후
            logging.warning(self.model.generation_config)

            output = self.model.generate(inputs)
            gen_tokens = len(output[:, inputs.shape[-1] :].tolist()[0])
            response = self.tokenizer.batch_decode(
                output[:, inputs.shape[-1] :], skip_special_tokens=True
            )[0]


            # logging.warning(f"response: {response} \n input_tokens: {input_tokens} output_tokens: {gen_tokens}")
            logging.warning(f"input_tokens: {input_tokens} output_tokens: {gen_tokens}")

            output = pb_utils.Tensor(
                "response", np.array(response.strip(), dtype=np.object_)
            )
            output_tokens = pb_utils.Tensor(
                "output_tokens", np.array(gen_tokens, dtype=np.int64)
            )
            input_tokens = pb_utils.Tensor(
                "input_tokens", np.array(input_tokens, dtype=np.int64)
            )
            response = pb_utils.InferenceResponse(
                output_tensors=[output, output_tokens, input_tokens]
            )

            # generation_config 복원
            self.model.generation_config = original_generation_config
            self.model.generation_config.temperature = 1.0

            responses.append(response)

            gc.collect()
            torch.cuda.empty_cache()

        return responses

    def finalize(self):
        pass
