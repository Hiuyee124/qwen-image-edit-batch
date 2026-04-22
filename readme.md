# Qwen-Image-Edit-MultiRef

This project is developed based on the **diffusion-nft** library, and extends the **Qwen-Image-Edit-2511** pipeline to support:

- ✅ Multi-reference image input  
- ✅ Batch inference  
- ✅ Simple and practical interface for downstream tasks  

It enables running image editing with multiple reference images and multiple samples in a single forward pass.

---

## 🚀 Example

A minimal batch example with multi-reference images:

```python
from PIL import Image
from diffusers import QwenImageEditPlusPipeline
import torch

model_path = "Qwen/Qwen-Image-Edit"

pipeline = QwenImageEditPlusPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

# batch of prompts
prompts = [
    "Edit the image using the reference",
    "Apply the same transformation"
]

# each sample uses multiple reference images
images = [
    [Image.open("img1_a.jpg").convert("RGB"),
     Image.open("img1_b.jpg").convert("RGB")],

    [Image.open("img2_a.jpg").convert("RGB"),
     Image.open("img2_b.jpg").convert("RGB")]
]

results = pipeline(
    prompt=prompts,
    image=images,
    num_inference_steps=30,
    true_cfg_scale=4.0,
).images

for i, img in enumerate(results):
    img.save(f"output_{i}.jpg")
