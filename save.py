import torch
from doubutsu.configuration_doubutsu import DoubutsuConfig
from doubutsu.modeling_doubutsu import Doubutsu, ProjectionModule
from transformers import (
    Qwen2ForCausalLM,
    SiglipImageProcessor,
    SiglipVisionModel,
    AutoTokenizer,
)

DoubutsuConfig.register_for_auto_class()
Doubutsu.register_for_auto_class("AutoModelForCausalLM")
config = DoubutsuConfig()
model = Doubutsu(config)
model_name = "google/siglip-so400m-patch14-384"
model.vision_model = SiglipVisionModel.from_pretrained(model_name)
model.processor = SiglipImageProcessor.from_pretrained(model_name)
model.text_model = Qwen2ForCausalLM.from_pretrained(
    "Qwen/Qwen2-1.5B-Instruct",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct", use_fast=True)
model.text_model.config.max_length = 32768
config.text_config = model.text_model.config
config.vision_config = model.vision_model.config


def load_projection_module():
    projection_module = ProjectionModule()
    checkpoint = torch.load("./checkpoints/mm_projector.bin", map_location="cpu")
    checkpoint = {
        k.replace("model.mm_projector.", ""): v for k, v in checkpoint.items()
    }

    projection_module.load_state_dict(checkpoint)
    return projection_module


model.mm_projector = load_projection_module()
model = model.to(dtype=torch.float16)
tokenizer.save_pretrained("./checkpoints/qwenvision-pretrain-uhd")


def save_full_model(model, path):
    full_state_dict = model.state_dict()

    model.config.save_pretrained(path)

    torch.save(full_state_dict, f"{path}/pytorch_model.bin")

    print(f"Full model saved to {path}")
