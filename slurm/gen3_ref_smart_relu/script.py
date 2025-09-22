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

network_inits = {
    'resnet18': init_torch_resnet18,
    'resnet34': init_torch_resnet34,
    'resnet50': init_torch_resnet50,
    'resnet101': init_torch_resnet101,
    'resnet152': init_torch_resnet152,
    'vgg16': init_torch_vgg16,
    'vgg19': init_torch_vgg19,
    'clip': lambda: (lambda x: clip_suite.image_projector()(x).image_embeds),
}

for i, img_path in enumerate(img_paths):
    init_conf = RunConfig(
        loss_scale_rn = 1.0,
        )

    init_conf.loss_scale["tv"] = 1
    init_conf.loss_function["tv"] = "squared_relu"
    init_conf.loss_calculate["clip"] = True
    init_conf.loss_calculate["lpips_alex"] = True
    init_conf.loss_calculate["lpips_vgg"] = True
    init_conf.loss_calculate["lpips_squeeze"] = True
    init_conf.loss_calculate["rn"] = True
    init_conf.loss_calculate["tv"] = True
    init_conf.loss_function["rn"] = "mse"
    init_conf.optimization_space = "image"
    init_conf.iterations = 3000


    result = run_reconstructions(img_path=img_path, config=init_conf,
                        img_logging_freq=100, reconstructions_per_img=8,
                        clip_suite=clip_suite, run_tags=["ref_srl", "init"])


    for rn_scale in [0.0, 1.0]:
        for model_name in ['clip', 'vgg16', 'resnet18', 'resnet50', 'resnet152']:
            conf = RunConfig(
                )
            #configs.append(RunConfig( loss_scale_clip = 1.0))

            conf.loss_scale["tv"] = 1
            conf.loss_function["tv"] = "squared_relu"
            conf.loss_calculate["clip"] = True
            conf.loss_calculate["lpips_alex"] = True
            conf.loss_calculate["lpips_vgg"] = True
            conf.loss_calculate["lpips_squeeze"] = True
            conf.loss_calculate["rn"] = True
            conf.loss_calculate["tv"] = True
            conf.loss_function["rn"] = "mse"
            conf.loss_calculate["avg"]["vgg16"] = True
            conf.loss_calculate["ref"][model_name] = True
            conf.loss_scale["ref"][model_name] = 1.0
            conf.loss_scale["rn"] = rn_scale
            conf.optimization_space = "image"
            conf.apply_smart_relu_on_ref_and_avg = model_name != "clip"
            conf.iterations = 3000

            model = network_inits[model_name]()
            embeds = model(result["reconstructions"])
            embeds_avg = embeds.mean(dim=0, keepdim=False)
            conf.ref_embedding[model_name] = embeds_avg

            run_reconstructions(img_path=img_path, config=conf,
                                img_logging_freq=100, reconstructions_per_img=8,
                                clip_suite=clip_suite, run_tags=["ref_srl", f"ref_srl_{model_name}", "rerun"])