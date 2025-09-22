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

if len(sys.argv) > 1:
    img_paths = sys.argv[1:]
else:
    raise Exception("No command line arguments provided.")


for i, img_path in enumerate(img_paths):
    for model_name in ["resnet18", "resnet50", "resnet152", "vgg16", "clip"]:
        config = RunConfig(
                    loss_scale_rn=1.0,
                    loss_scale_tv=1.0,
                    loss_scale_avg_clip=1.0 if model_name == "clip" else 0.0,
                    loss_scale_avg_vgg16=1.0 if model_name == "vgg16" else 0.0,
                    loss_scale_avg_resnet18=1.0 if model_name == "resnet18" else 0.0,
                    loss_scale_avg_resnet50=1.0 if model_name == "resnet50" else 0.0,
                    loss_scale_avg_resnet152=1.0 if model_name == "resnet152" else 0.0,
                    loss_calculate_clip=True,
                    loss_calculate_rn=True,
                    loss_calculate_tv=True,
                    loss_calculate_lpips_alex=True,
                    loss_calculate_lpips_vgg=True,
                    loss_calculate_lpips_squeeze=True,
                    loss_function_rn="mse",
                    loss_function_tv="squared_relu",
                    optimization_space="image",
                    apply_relu_on_ref_and_avg=model_name != "clip",
                    iterations=3000,
        )

        run_reconstructions(img_path=img_path, config=config,
                            img_logging_freq=100, reconstructions_per_img=8,
                            clip_suite=clip_suite, run_tags=["avg_rl", "avg_rl_" + model_name])