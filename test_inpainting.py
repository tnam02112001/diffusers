from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d_inpaint import StableDiffusionLDM3DInpaintPipeline
from PIL import Image
import numpy as np

pipe = StableDiffusionLDM3DInpaintPipeline.from_pretrained("Intel/ldm3d-4c", cache_dir="cache")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
input_image = Image.open("astronaut_ldm3d_rgb.jpg")
depth_image = Image.open("astronaut_ldm3d_depth.png")
mask_image = np.zeros_like(np.array(depth_image))
#dummy threshold
mask_image[np.array(depth_image) < 10000] = 65535
mask_image = Image.fromarray(mask_image).convert("L")

output = pipe(prompt=prompt, image=input_image, mask_image=mask_image, depth_image=depth_image)