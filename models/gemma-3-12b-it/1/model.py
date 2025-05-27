import os
import torch
import triton_python_backend_utils as pb_utils
import json

from models.base import Base


class TritonPythonModel(Base):

    def initialize(self, args):

        self.model_instance_device_id = json.loads(args["model_instance_device_id"])
        torch.cuda.set_device(self.model_instance_device_id)

        self.logger = pb_utils.Logger
        self.model_path = "/triton_llm/weights/gemma-3-12b-it"

        super().initialize()
