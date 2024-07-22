import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoModel,
    SiglipImageProcessor,
)
from .configuration_doubutsu import DoubutsuConfig


class ProjectionModule(nn.Module):
    def __init__(self, mm_hidden_size=1152, hidden_size=1536):
        super(ProjectionModule, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(mm_hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        return self.model(x)


class Doubutsu(PreTrainedModel):
    config_class = DoubutsuConfig

    def __init__(self, config):
        super().__init__(config)

        self.vision_model = AutoModel.from_config(self.config.vision_config)
        self.text_model = AutoModelForCausalLM.from_config(self.config.text_config)
        self.processor = SiglipImageProcessor()
        self.mm_projector = ProjectionModule(
            mm_hidden_size=config.vision_config.hidden_size,
            hidden_size=config.text_config.hidden_size,
        )

    @property
    def device(self):
        return self.text_model.device

    def encode_image(self, image):
        image = image.convert("RGB")
        image = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": 378, "width": 378},
        )["pixel_values"].to(
            device=self.vision_model.device, dtype=self.vision_model.dtype
        )
        with torch.no_grad():
            return self.vision_model(image, output_hidden_states=True).hidden_states[-2]

    def input_embeds(self, prompt, image_embeds, tokenizer):
        def _tokenize(txt):
            return tokenizer(
                txt, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(self.device)

        text_emb = self.text_model.get_input_embeddings()

        embeds = []

        tokenized_prompt = _tokenize(prompt)
        if (
            tokenizer.bos_token_id is not None
            and tokenized_prompt[0][0] != tokenizer.bos_token_id
        ):
            embeds.append(
                text_emb(torch.tensor([[tokenizer.bos_token_id]], device=self.device))
            )

        projected_image_embeds = self.mm_projector(image_embeds.to(self.device))
        embeds.append(projected_image_embeds)

        embeds.append(text_emb(tokenized_prompt))

        return torch.cat(embeds, dim=1)

    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    def generate(
        self,
        image_embeds,
        prompt,
        tokenizer,
        max_new_tokens=128,
        **kwargs,
    ):
        generate_config = {
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
            **kwargs,
        }

        with torch.no_grad():
            inputs_embeds = self.input_embeds(prompt, image_embeds, tokenizer)
            output_ids = self.text_model.generate(
                inputs_embeds=inputs_embeds, **generate_config
            )

        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    def answer_question(self, image, question, tokenizer, **kwargs):
        image_embeds = self.encode_image(image)

        chat = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant that can see images and answer questions about them.",
            },
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        # Generate the answer
        with torch.no_grad():
            output = self.generate(
                image_embeds=image_embeds,
                prompt=prompt,
                tokenizer=tokenizer,
                **kwargs,
            )[0]

        # Clean and return the answer
        cleaned_answer = output.strip()
        return cleaned_answer
