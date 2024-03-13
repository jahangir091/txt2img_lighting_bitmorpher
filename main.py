import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import time
import uvicorn

from fastapi import FastAPI, Body

from utils import get_img_path

app = FastAPI()

stime = time.time()
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!



# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", return_cached_folder=True).to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

app.pipe = pipe

@app.post("/ai/api/v1/txt2img")
def txt2img_lighting(
        prompt: str = Body("", title='prompt'),
):
    start_time = time.time()

    # output_file_name = uuid.uuid4().hex[:20] + '.png'
    output = pipe(prompt, num_inference_steps=4, guidance_scale=0)

    out_image_directory_name = '/out_lighting_images/'
    out_image_path = get_img_path(out_image_directory_name)

    # save the resulting image
    output.images[0].save(out_image_path)
    print(len(output.images))

    return {
        "success": True,
        "message": "Returned output successfully",
        "server_process_time": time.time() - start_time,
        "output_media_url": '/media' + out_image_directory_name + out_image_path.split('/')[-1]
    }


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8006)

# Ensure using the same inference steps as the loaded model and CFG set to 0.


