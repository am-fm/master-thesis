# gradthesis
I referred to [huggingface/diffusers](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py).
### Training
```
python train_brain_to_context_to_image.py
```

### Inference
```python
from diffusers import StableDiffusionPipeline
import torch

model_path = "sd-model-finetuned"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt_embeds=encoder_hidden_states[0].unsqueeze(0)).images[0]
image.save("generated.png")
```
