import gc
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
import numpy as np
import json


# class TritonPythonModel:
class Base:
    def initialize(self):

        # self.model_instance_device_id = json.loads(args['model_instance_device_id'])
        # torch.cuda.set_device(self.model_instance_device_id)

        # self.logger = pb_utils.Logger
        # # self.model_path = "/weights/Meta-Llama-3.1-8B-Instruct"
        # # self.model_path = "/home/miruware/triton/weights/opt125m"
        # self.model_path = "/triton/weights/opt125m"
        
        
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config = GenerationConfig(
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            max_new_tokens=1500,
            top_k=85,
            top_p=0.85,
            temperature=0.2,
            use_cache=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{self.model_instance_device_id}"},
            trust_remote_code=True
        )

    def execute(self, requests):
        responses = []

        for request in requests:

            decoder = np.vectorize(lambda x: x.decode('UTF-8'))

            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "system_prompt").as_numpy()
            query_tensor = pb_utils.get_input_tensor_by_name(request, "query").as_numpy()
            context_tensor = pb_utils.get_input_tensor_by_name(request, "context").as_numpy()
            prompt = decoder(prompt_tensor)[0][0]
            query = decoder(query_tensor)[0][0]
            context = decoder(context_tensor)[0][0]

            messages = [
                {"role": "system", "content": f"{prompt}"},
                {"role": "user", "content": f"텍스트:{context}\n사용자:{query}"},
            ]

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt"
            ).to(self.model.device)

            output = self.model.generate(inputs, generation_config=self.config)
            gen_tokens = len(output[:, inputs.shape[-1]:].tolist()[0])
            response = self.tokenizer.batch_decode(output[:, inputs.shape[-1]:], skip_special_tokens=True)[0]

            log_data = {
                "system_prompt": prompt,
                "context": len(context),
                "query": query,
                "response": response
            }

            self.logger.log_info(f"{log_data}")

            out_tensor = pb_utils.Tensor("response", np.array(response.strip(), dtype=np.object_))
            out2_tensor = pb_utils.Tensor("tokens", np.array(gen_tokens, dtype=np.int64))
            response = pb_utils.InferenceResponse(output_tensors=[out_tensor, out2_tensor])
            responses.append(response)

            gc.collect()
            torch.cuda.empty_cache()

        return responses

    def finalize(self):
        pass
