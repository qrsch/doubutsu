# doubutsu

A family of smol VLMs.

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

model_id = "qresearch/doubutsu-2b-pt-756"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

model.load_adapter("qresearch/doubutsu-2b-lora-756-docci")

image = Image.open("IMAGE")

print(
    model.answer_question(
        image, "Describe the image", tokenizer, max_new_tokens=128, temperature=0.1
    ),
)
```

## Training

You can train a doubutsu on your own use-case via one of the notebooks provided in this repo.
