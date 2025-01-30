import os
import torch
import triton_python_backend_utils as pb_utils
import json

from models.base import Base

class TritonPythonModel(Base):

    def initialize(self, args):

        self.model_instance_device_id = json.loads(args['model_instance_device_id'])
        torch.cuda.set_device(self.model_instance_device_id)

        self.logger = pb_utils.Logger
        self.model_path = "/triton_llm/weights/opt125m"
        
        if os.path.exists(self.model_path):
            pass
        else:
            from transformers import AutoModel, AutoTokenizer
            model_name = "facebook/opt-125m"

            # 모델과 토크나이저 로드
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # 저장
            model.save_pretrained(self.model_path)
            tokenizer.save_pretrained(self.model_path)
            

            
        super().initialize()
        
    def execute(self, requests):
        return super().execute(requests)


    def finalize(self):
        pass
