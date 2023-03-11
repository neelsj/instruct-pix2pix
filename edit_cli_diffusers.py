import sys
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from PIL import Image


#model_id= "runwayml/stable-diffusion-v1-5"
#output_dir = "E:/Source/instruct-pix2pix/checkpoints/pokemon"
##output_dir = "sayakpaul/sd-model-finetuned-lora-t4"

#pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#pipe.unet.load_attn_procs(output_dir)
#pipe.to("cuda")

#prompt = "A pokemon with blue eyes."

#images = pipe(prompt).images
#images[0].show()

model_id = "timbrooks/instruct-pix2pix"

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

path = 'E:/Research/HoloAssist/holoassist_samples/z047-june-25-22-nespresso/1_Water_0065.png'

prompt = "pour water into the coffee machine"

image = Image.open(path)

images = pipe(prompt, image=image).images
images[0].show()