import torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import time
import uvicorn
import requests

from fastapi import FastAPI, Body

from utils import get_img_path

app = FastAPI()

stime = time.time()
base = "stabilityai/stable-diffusion-xl-base-1.0"
repo = "ByteDance/SDXL-Lightning"
ckpt = "sdxl_lightning_4step_unet.safetensors" # Use the correct ckpt for your step setting!


txt2img_styles_res = requests.get('https://photolab-ai.com/media/giff/ai/txt2img_styles/txt2img_styles.json')
styles_dict = txt2img_styles_res.json()

# Load model.
unet = UNet2DConditionModel.from_config(base, subfolder="unet").to("cuda", torch.float16)
unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device="cuda"))
pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float16, variant="fp16", return_cached_folder=True).to("cuda")

# Ensure sampler uses "trailing" timesteps.
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

app.pipe = pipe
app.txt2img_styles = styles_dict

@app.post("/ai/api/v1/txt2img")
def txt2img_lighting(
        prompt: str = Body(title='user prompt'),
        negative_prompt: str = Body('', title='user prompt'),
        model_id: int = Body(1, title='model unique id'),
        seed: int = Body(-1, title="seed value"),
        batch_count: int = Body(1, title="no of batch to produce at a time"),
        steps: int = Body(4, title="steps for image generation"),
        cfg_scale: float = Body(0.0, title="cfg scale"),
        style_id: int = Body(1, title='selected style of user'),
        height: int = Body(1024, title='height of generated image'),
        width: int = Body(1024, title='width of generated image')

):
    start_time = time.time()

    global_style_dict = next((d for d in app.txt2img_styles if d.get("id") == 1), None)
    if style_id != 1:
        style_dict = next((d for d in app.txt2img_styles if d.get("id") == style_id), None)
        if style_dict:
            prompt = style_dict['prompt'].format(prompt=prompt)
    prompt += global_style_dict['prompt']
    negative_prompt += global_style_dict['negative_prompt']

            # 768, width 416
    output = pipe(prompt, num_inference_steps=4, guidance_scale=0, num_images_per_prompt=batch_count, height=1024, width=1024, negative_prompt=negative_prompt, seed=seed)

    out_image_directory_name = '/out_lighting_images/'
    out_image_paths = []
    for img in output.images:
        out_image_path = get_img_path(out_image_directory_name)
        # save the resulting image
        img.save(out_image_path)
        out_image_paths.append('/media' + out_image_directory_name + out_image_path.split('/')[-1])

    print('total generated imaged: {0}, height {1}, width {2}'.format(len(output.images), height, width))
    torch.cuda.empty_cache()

    return {
        "success": True,
        "message": "Returned output successfully",
        "server_process_time": time.time() - start_time,
        "output_media_urls": out_image_paths
    }

@app.get("/ai/api/v1/txt2img-lighting-server-test")
def illusion_server_test():
    return {"server is working fine. OK!"}


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8006)

# Ensure using the same inference steps as the loaded model and CFG set to 0.


