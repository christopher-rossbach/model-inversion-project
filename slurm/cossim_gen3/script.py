# Add the parent directory to sys.path so 'inversion' can be imported
import torch

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from inversion.utils import *
from inversion.clip_guided_stable_diffusion import *
from inversion.clip_utils import *
from inversion.run_config import RunConfig
from inversion.reconstruction import run_reconstructions


# Load the individual components of Stable Diffusion
diffusion_model_name = "CompVis/stable-diffusion-v1-4"
clip_model_name = "openai/clip-vit-large-patch14"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_suite = CLIPSuite(diffusion_model_name, clip_model_name)

configs = []
configs.append(RunConfig(
                loss_scale_rn=1.0,
                loss_scale_tv=1.0,
                loss_scale_classification_1k=0.0,
                loss_scale_clip=0.0,
                loss_calculate_clip=True,
                loss_calculate_rn=True,
                loss_calculate_tv=True,
                loss_calculate_lpips_alex=True,
                loss_calculate_lpips_vgg=True,
                loss_calculate_lpips_squeeze=True,
                loss_calculate_classification_1k=True,
                loss_function_rn="cosine_similarity",
                loss_function_tv="squared_relu",
                optimization_space="image",
                iterations=3000,
))


if len(sys.argv) > 1:
    img_paths = sys.argv[1:]
else:
    raise Exception("No command line arguments provided.")

#img_paths = ["Images/14042.jpg", "Images/14158.jpg", "Images/14034.jpg"]
img_paths = img_paths

for i, img_path in enumerate(img_paths):
    for ii, config in enumerate(configs):
        print(f"Running reconstruction {i+1}/{len(img_paths)} with config {ii+1}/{len(configs)}: {config.get_as_dict()}")
        # Initialize a wand
        run_reconstructions(img_path=img_path, config=config,
                            img_logging_freq=100, reconstructions_per_img=4,
                            clip_suite=clip_suite, run_tags=["cossim"])