from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d_inpaint import StableDiffusionLDM3DInpaintPipeline
from PIL import Image
import numpy as np
from diffusers import UNet2DConditionModel

# Cargar con weights inicializados random
unet = UNet2DConditionModel.from_pretrained("Intel/ldm3d-4c", cache_dir="cache", subfolder="unet", in_channels=9, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

pipe = StableDiffusionLDM3DInpaintPipeline.from_pretrained("Intel/ldm3d-4c", cache_dir="cache" , unet=unet)

pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
input_image = Image.open("input_rgb.png")
depth_image = Image.open("input_depth.png")
mask_image = Image.open("input_mask.png")
#mask_image = np.zeros_like(np.array(depth_image))
#dummy threshold
#mask_image[np.array(depth_image) < 10000] = 65535
#mask_image = Image.fromarray(mask_image).convert("L")

output = pipe(prompt=prompt, image=input_image, mask_image=mask_image, depth_image=depth_image, num_inference_steps=20)

rgb = output["rgb"][0]
depth = output["depth"][0]
rgb.save("output_rgb.png")
depth.save("output_depth.png")