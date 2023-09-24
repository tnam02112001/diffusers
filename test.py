#from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d_inpaint import StableDiffusionLDM3DInpaintPipeline

#pipeline = StableDiffusionLDM3DInpaintPipeline()
from src.diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_ldm3d import StableDiffusionLDM3DPipeline

pipe = StableDiffusionLDM3DPipeline.from_pretrained("Intel/ldm3d-4c", cache_dir="cache")
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
output = pipe(prompt, num_inference_steps=15)
rgb_image, depth_image = output.rgb, output.depth
rgb_image[0].save("output_rgb.jpg")
depth_image[0].save("output_depth.png")