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
    'resnet18': lambda: init_torch_resnet18(no_relu=False),
    'resnet34': lambda: init_torch_resnet34(no_relu=False),
    'resnet50': lambda: init_torch_resnet50(no_relu=False),
    'resnet101': lambda: init_torch_resnet101(no_relu=False),
    'resnet152': lambda: init_torch_resnet152(no_relu=False),
    'vgg16': lambda: init_torch_vgg16(no_relu=False),
    'vgg19': lambda: init_torch_vgg19(no_relu=False),
    'clip': lambda: (lambda x: clip_suite.image_projector()(x).image_embeds),
}

for i, img_path in enumerate(img_paths):
    init_conf = RunConfig(
        loss_scale_rn = 1.0,
        no_relu_for_ref_and_avg=False,
        )

    init_conf.loss_scale["tv"] = 1
    init_conf.loss_function["tv"] = "squared_relu"
    init_conf.loss_calculate["clip"] = True
    init_conf.loss_calculate["lpips_alex"] = True
    init_conf.loss_calculate["lpips_vgg"] = True
    init_conf.loss_calculate["lpips_squeeze"] = False
    init_conf.loss_calculate["rn"] = True
    init_conf.loss_calculate["tv"] = True
    init_conf.loss_function["rn"] = "mse"
    init_conf.optimization_space = "image"
    init_conf.iterations = 3000


    result = run_reconstructions(img_path=img_path, config=init_conf,
                        img_logging_freq=100, reconstructions_per_img=16,
                        clip_suite=clip_suite, run_tags=["ffhq", "only_16", "ref", "init"])


    for rn_scale in [1.0]:
        for model_name in ["clip", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "vgg16", "vgg19"]:
            conf = RunConfig(
                no_relu_for_ref_and_avg=False
            )
            #configs.append(RunConfig( loss_scale_clip = 1.0))

            conf.loss_scale["tv"] = 1
            conf.loss_function["tv"] = "squared_relu"
            conf.loss_calculate["clip"] = True
            conf.loss_calculate["lpips_alex"] = True
            conf.loss_calculate["lpips_vgg"] = True
            conf.loss_calculate["lpips_squeeze"] = False
            conf.loss_calculate["rn"] = True
            conf.loss_calculate["tv"] = True
            conf.loss_function["rn"] = "mse"
            conf.loss_calculate["ref"][model_name] = True
            conf.loss_scale["ref"][model_name] = 1.0
            conf.loss_scale["rn"] = rn_scale
            conf.optimization_space = "image"
            conf.iterations = 3000

            model = network_inits[model_name]()
            embeds = model(result["reconstructions"])
            embeds_avg = embeds.mean(dim=0, keepdim=False)
            conf.ref_embedding[model_name] = embeds_avg

            run_reconstructions(img_path=img_path, config=conf,
                                img_logging_freq=100, reconstructions_per_img=16,
                                clip_suite=clip_suite, run_tags=["ffhq", "only_16", "ref", f"ref_{model_name}", "rerun"])