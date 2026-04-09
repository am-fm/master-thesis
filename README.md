# gradthesis

Inference
```
from diffusers import StableDiffusionPipeline
import torch

model_path = "sd-model-finetuned-b2c"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt_embeds=encoder_hidden_states[0].unsqueeze(0)).images[0]
image.save("generated.png")
```
