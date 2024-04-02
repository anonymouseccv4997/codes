<div align="center">

<h1> ResAdapter: Domain Consistent Resolution Adapter for Diffusion Models  </h1>

Anonymous ECCV4997

**We have deleted any information about authors. [ResAdapter weights](models/res_adapter) are privided in this repo.**

**We propose ResAdapter, a plug-and-play resolution adapter for enabling any diffusion model generate resolution-free images: no additional training, no additional inference and no style transfer.**

<img src="assets/misc/dreamlike1.png" width="49.9%"><img src="assets/misc/dreamlike2.png" width="50%">
Comparison examples between resadapter and [dreamlike-diffusion-1.0](https://civitai.com/models/1274/dreamlike-diffusion-10).

</div>

## Quicktour

We provide a standalone [example code](quicktour.py) to help you quickly use resadapter with diffusion models.

<div align=center>

<img src="assets/misc/dreamshaper_resadapter.png" width="100%">
<img src="assets/misc/dreamshaper_baseline.png" width="100%">

Comparison examples (640x384) between resadapter and [dreamshaper-xl-1.0](https://huggingface.co/Lykon/dreamshaper-xl-1-0). Top: with resadapter. Bottom: without resadapter.

</div>

```python
# pip install diffusers, transformers, accelerate, safetensors, huggingface_hub
import torch
from torchvision.utils import save_image
from safetensors.torch import load_file
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

generator = torch.manual_seed(0)
prompt = "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
width, height = 640, 384

# Load baseline pipe
model_name = "lykon-models/dreamshaper-xl-1-0"
pipe = AutoPipelineForText2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16").to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")

# Inference baseline pipe
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_baseline.png", normalize=True, padding=0)

# Load resadapter for baseline
resadapter_model_name = "models/res_adapter/resadapter_v1_sdxl"
pipe.load_lora_weights(
    f"{resadapter_model_name}/pytorch_lora_weights.safetensors", 
    adapter_name="res_adapter",
    ) # load lora weights
pipe.set_adapters(["res_adapter"], adapter_weights=[1.0])
pipe.unet.load_state_dict(
    load_file(f"{resadapter_model_name}/diffusion_pytorch_model.safetensors"),
    strict=False,
    ) # load norm weights

# Inference resadapter pipe
image = pipe(prompt, width=width, height=height, num_inference_steps=25, num_images_per_prompt=4, output_type="pt").images
save_image(image, f"image_resadapter.png", normalize=True, padding=0)
```

## Download

### Models

For anonymous version, we directly provide resadapter weights in this repo.

|Models  | Parameters | Resolution Range | Ratio Range | Links |
| --- | --- |--- | --- | --- |
|resadapter_v1_sd1.5| 0.9M | 128 <= x <= 1024 | 0.5 <= r <= 2 | ... |
|resadapter_v1_sd1.5_extrapolation| 0.9M | 512 <= x <= 1024 | 0.5 <= r <= 2  | ...|
|resadapter_v1_sd1.5_interpolation| 0.8M | 128 <= x <= 512 | 0.5 <= r <= 2  | ... |
|resadapter_v1_sdxl| 0.5M | 256 <= x <= 1536 | 0.5 <= r <= 2  | ... |
|resadapter_v1_sdxl_extrapolation| 0.5M | 1024 <= x <= 1536 | 0.5 <= r <= 2  | ... |
|resadapter_v1_sdxl_interpolation| 0.4M | 256 <= x <= 1024 | 0.5 <= r <= 2  | ... |

Hint1: We update the resadapter name format according to [controlnet](https://github.com/lllyasviel/ControlNet-v1-1-nightly).

Hint2: If you want use resadapter with personalized diffusion models, you should download them from [CivitAI](https://civitai.com/).

Hint3: If you want use resadapter with ip-adapter, controlnet and lcm-lora, you should download them from [Huggingface](https://huggingface.co/welcome).

Hint4: Here is an [installation guidance](models/README.md) for preparing environment and downloading models.

## Inference

If you want generate images in our inference script, you should download [related models](models/README.md) and fill in [configs](configs). Then you can directly run this script.

```bash
python main.py --config /path/to/file
```

### ResAdapter with Personalized Models for Text to Image

<div align=center>

<img src="assets/misc/dreamshaper-1024/resadapter1.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter2.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter3.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/resadapter4.jpg" width="25%">
<img src="assets/misc/dreamshaper-1024/baseline1.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline2.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline3.jpg" width="25%"><img src="assets/misc/dreamshaper-1024/baseline4.jpg" width="25%">

Comparison examples (960x1104) between resadapter and [dreamshaper-7](https://civitai.com/models/1274/dreamlike-diffusion-10). Top: with resadapter. Bottom: without resadapter.

</div>

### ResAdapter with ControlNet for Image to Image

<div align=center>

<img src="assets/misc/controlnet/condition_bird.jpg" width="20%"><img src="assets/misc/controlnet/bird_1_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_2_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_3_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet/bird_4_ResAdapter.jpg" width="20%">
<img src="assets/misc/controlnet/condition_bird.jpg" width="20%"><img src="assets/misc/controlnet/bird_1_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_5_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_3_Baseline.jpg" width="20%"><img src="assets/misc/controlnet/bird_4_Baseline.jpg" width="20%">

Comparison examples (840x1264) between resadapter and [lllyasviel/sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny). Top: with resadapter, bottom: without resadapter.

</div>

### ResAdapter with ControlNet-XL for Image to Image

<div align=center>

<img src="assets/misc/controlnet-xl/condition_man.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_0_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_1_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_2_ResAdapter.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_3_ResAdapter.jpg" width="20%">
<img src="assets/misc/controlnet-xl/condition_man.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_0_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_1_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_2_Baseline.jpg" width="20%"><img src="assets/misc/controlnet-xl/man_3_Baseline.jpg" width="20%">

Comparison examples (336x504) between resadapter and [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0). Top: with resadapter, bottom: without resadapter.

</div>

### ResAdapter with IP-Adapter for Face Variance

<div align=center>
<img src="assets/ip_adapter/ai_face2.png" width="20%"><img src="assets/misc/ip-adapter/resadapter3.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter4.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter5.jpg" width="20%"><img src="assets/misc/ip-adapter/resadapter7.jpg" width="20%">
<img src="assets/ip_adapter/ai_face2.png" width="20%"><img src="assets/misc/ip-adapter/baseline3.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline4.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline5.jpg" width="20%"><img src="assets/misc/ip-adapter/baseline7.jpg" width="20%">

Comparison examples (864x1024) between resadapter and [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). Top: with resadapter, bottom: without resadapter.


</div>


### ResAdapter with LCM-LoRA for Speeding up

<div align=center>

<img src="assets/misc/lcm-lora/resadapter5.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter3.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter2.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter4.jpg" width="20%"><img src="assets/misc/lcm-lora/resadapter1.jpg" width="20%">
<img src="assets/misc/lcm-lora/baseline5.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline3.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline2.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline4.jpg" width="20%"><img src="assets/misc/lcm-lora/baseline1.jpg" width="20%">

Comparison examples (512x512) between resadapter and [dreamshaper-xl-1.0](https://huggingface.co/Lykon/dreamshaper-xl-1-0) with [lcm-sdxl-lora](https://huggingface.co/latent-consistency/lcm-lora-sdxl). Top: with resadapter, bottom: without resadapter.


</div>

## Community Resource

### Gradio
For anonymous version, we delete gradio website.

### ComfyUI

https://github.com/anonymouseccv4997/codes/assets/162449909/9211d6a0-544d-49e1-ad29-4c1b1cf6c099


## Usage Tips

- We recommend users to use **interpolation** version to generate lower-resolution images.
- We recommend users to use **extrapolation** version to generate higher-resolution images.
- We recommend users to use `resadapter_v1_sd1.5` and `resadapter_v1_sdxl` for deploying resadapter to generate images with broader resolution.
- We strongly recommend that you use the prompt corresponding to the personalized model, which helps to enhance the quality of the image.

## Acknowledgements

- Thanks to the [HuggingFace](https://huggingface.co/) gradio team for their free GPU support!
- Thanks to the [IP-Adapter](), [ControlNet](), [LCM-LoRA]() for their nice work.
