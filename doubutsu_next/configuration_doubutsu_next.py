from transformers import PretrainedConfig, Qwen2Config, SiglipVisionConfig


class DoubutsuNextConfig(PretrainedConfig):
    model_type = "doubutsu_next"

    def __init__(self, **kwargs):
        self.text_config = Qwen2Config(
            **kwargs.pop(
                "text_config",
                {},
            ),
        )
        self.vision_config = SiglipVisionConfig(**kwargs.pop("vision_config", {}))
        super().__init__(**kwargs)
