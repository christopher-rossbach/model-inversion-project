from .utils import *
from .run_config import RunConfig
from .clip_utils import *
from .GAN import *

import wandb
import PIL

from pytorch_msssim import ssim

def collect_losses(
        reconstruction,
        surrogate_clip,
        clip_projection,
        resnet_proj,
        resnet_classifications,
        clip_suite: CLIPSuite,
        config,
        iteration,
        original_image,
        embedding_models,
        ):
    clip_loss = torch.zeros((reconstruction.size(0),), device=clip_suite.device)
    sclip_loss = torch.zeros((reconstruction.size(0),), device=clip_suite.device)
    rn_loss = torch.zeros_like(clip_loss)
    tv_loss = torch.zeros_like(clip_loss)
    classification_1k_loss = torch.zeros_like(clip_loss)
    lpips_loss_alex = torch.zeros_like(clip_loss)
    lpips_loss_vgg = torch.zeros_like(clip_loss)
    lpips_loss_squeeze = torch.zeros_like(clip_loss)
    ssim_loss = {win_size: torch.zeros_like(clip_loss) for win_size in config.loss_calculate["ssim"].keys()}


    distance_measures = {
        "mse": F.mse_loss,
        "mse_dim_normalized": mse_dim_normalized,
        "cosine_similarity": cosine_distance,
    }

    penalty_functions = {
        "linear": lambda x: x,
        "squared_relu": lambda x: F.relu(x-0.3) + F.relu(x - 0.3)**2 * 10
    }

    if config.loss_calculate["classification_1k"]:
        classification_1k = clip_suite.resnet_classification_1k_layer()(clip_suite.resnet()(reconstruction))
        classification_1k_loss = distance_measures[config.loss_function['classification_1k']](classification_1k, resnet_classifications)
    if config.loss_calculate["rn"]:
        rn_loss = resnet_projection_loss(reconstruction, surrogate_clip, clip_projection, resnet_proj, clip_suite, distance_measures[config.loss_function['rn']])
    if config.loss_calculate["sclip"]:
        sclip_loss= surrogate_clip_loss(reconstruction, surrogate_clip, clip_projection, resnet_proj, clip_suite, distance_measures[config.loss_function['sclip']])
    if config.loss_calculate["clip"]:
        clip_loss= clip_projection_loss(reconstruction, surrogate_clip, clip_projection, resnet_proj, clip_suite, distance_measures[config.loss_function['clip']])
    if config.loss_calculate["tv"]:
        tv_loss = total_variation(reconstruction)
    if config.loss_calculate["lpips_alex"]:
        lpips_loss_alex = clip_suite.lpips_loss_alex()(reconstruction, original_image)
    if config.loss_calculate["lpips_vgg"]:
        lpips_loss_vgg = clip_suite.lpips_loss_vgg()(reconstruction, original_image)
    if config.loss_calculate["lpips_squeeze"]:
        lpips_loss_squeeze = clip_suite.lpips_loss_squeeze()(reconstruction, original_image)
    for win_size in config.loss_calculate["ssim"].keys():
        ssim_loss[win_size] = 1. - ssim(denormalize_fast(reconstruction), denormalize_fast(original_image), data_range=1., win_size=win_size)

    avg_loss = {}
    for name, calc in config.loss_calculate["avg"].items():
        avg_loss[name] = torch.zeros_like(clip_loss)
        if calc:
            embeds = embedding_models[name](reconstruction)
            embeds_avg = torch.mean(embeds, dim=0, keepdim=True)
            embeds_avg = embeds_avg.repeat(embeds.size(0), 1)
            if config.apply_relu_on_ref_and_avg:
                avg_loss[name] = distance_measures[config.loss_function["avg"][name]](F.relu(embeds_avg), F.relu(embeds))
            elif config.apply_smart_relu_on_ref_and_avg:
                relued_embeds_avg = F.relu(embeds_avg, inplace=False)
                relued_embeds = F.relu(embeds, inplace=False)
                used_relued_embeds = torch.zeros_like(relued_embeds)
                used_relued_embeds[relued_embeds_avg != 0] = embeds[relued_embeds_avg != 0]
                used_relued_embeds[relued_embeds_avg == 0] = relued_embeds[relued_embeds_avg == 0]
                avg_loss[name] = distance_measures[config.loss_function["avg"][name]](relued_embeds_avg, used_relued_embeds)
            else:
                avg_loss[name] = distance_measures[config.loss_function["avg"][name]](embeds_avg, embeds)

    ref_loss = {}
    for name, calc in config.loss_calculate["ref"].items():
        ref_loss[name] = torch.zeros_like(clip_loss)
        if calc:
            embeds = embedding_models[name](reconstruction)
            embeds_ref = config.ref_embedding[name].unsqueeze(0).repeat_interleave(embeds.size(0), dim=0)
            if config.apply_relu_on_ref_and_avg:
                ref_loss[name] = distance_measures[config.loss_function["ref"][name]](F.relu(embeds_ref), F.relu(embeds))
            elif config.apply_smart_relu_on_ref_and_avg:
                relued_embeds_ref = F.relu(embeds_ref, inplace=False)
                relued_embeds = F.relu(embeds, inplace=False)
                used_relued_embeds = torch.zeros_like(relued_embeds)
                used_relued_embeds[relued_embeds_ref != 0] = embeds[relued_embeds_ref != 0]
                used_relued_embeds[relued_embeds_ref == 0] = relued_embeds[relued_embeds_ref == 0]
                ref_loss[name] = distance_measures[config.loss_function["ref"][name]](relued_embeds_ref, used_relued_embeds)
            else:
                ref_loss[name] = distance_measures[config.loss_function["ref"][name]](embeds_ref, embeds)

    # Compute mean losses for logging
    classification_1k_loss_mean = classification_1k_loss.mean()
    rn_loss_mean = rn_loss.mean()
    clip_loss_mean = clip_loss.mean()
    sclip_loss_mean = sclip_loss.mean()
    tv_loss_mean = tv_loss.mean()
    lpips_loss_alex_mean = lpips_loss_alex.mean()
    lpips_loss_vgg_mean = lpips_loss_vgg.mean()
    lpips_loss_squeeze_mean = lpips_loss_squeeze.mean()
    ssim_loss_means = {win_size: loss.mean() for win_size, loss in ssim_loss.items()}
    avg_loss_means = {name: loss.mean() for name, loss in avg_loss.items()}
    avg_loss_sum = sum([avg_loss[name] * config.loss_scale["avg"][name] for name in avg_loss.keys()])
    ref_loss_means = {name: loss.mean() for name, loss in ref_loss.items()}
    ref_loss_sum = sum([ref_loss[name] * config.loss_scale["ref"][name] for name in ref_loss.keys()])
    loss_sum = (
        rn_loss * config.loss_scale["rn"] +
        clip_loss * config.loss_scale["clip"] +
        sclip_loss * config.loss_scale["sclip"] +
        penalty_functions[config.loss_function["tv"]](tv_loss) * config.loss_scale["tv"] +
        classification_1k_loss * config.loss_scale["classification_1k"] +
        avg_loss_sum +
        ref_loss_sum +
        0
    )
    wandb.log(
        {
            "losses": {
                "classification_1k_loss": classification_1k_loss_mean *  config.loss_scale["classification_1k"],
                "rn_loss": rn_loss_mean *  config.loss_scale["rn"],
                "clip_loss": clip_loss_mean * config.loss_scale["clip"],
                "sclip_loss": sclip_loss_mean * config.loss_scale["sclip"],
                "tv_loss": penalty_functions[config.loss_function["tv"]](tv_loss).mean() * config.loss_scale["tv"],
                "avg": {
                    name: avg_loss_means[name] * config.loss_scale["avg"][name]
                    for name in avg_loss_means.keys()
                },
                "ref": {
                    name: ref_loss_means[name] * config.loss_scale["ref"][name]
                    for name in ref_loss_means.keys()
                },
                "loss_sum":  loss_sum.mean(),
            },
            "losses_unscaled": {
                "avg": avg_loss_means,
                "ref": ref_loss_means,
                "classification_1k_loss_unscaled": classification_1k_loss_mean,
                "rn_loss_unscaled": rn_loss_mean,
                "clip_loss_unscaled": clip_loss_mean,
                "sclip_loss_unscaled": sclip_loss_mean,
                "tv_loss_unscaled": tv_loss_mean,
                "lpips_loss_alex": lpips_loss_alex_mean,
                "lpips_loss_vgg": lpips_loss_vgg_mean,
                "lpips_loss_squeeze": lpips_loss_squeeze_mean,
                "ssim_loss": ssim_loss_means,
            },
        }, step=iteration)

    loss = loss_sum.sum()
    return loss


def init_run(img_name, config, run_tags, img, reconstructions_per_img):
    run_name_parts = []
    run_name_parts.extend(run_tags)
    if config.loss_scale['clip'] != 0:
        run_name_parts.append(f"clip-{config.loss_scale['clip']}")
    if config.loss_scale['rn'] != 0:
        run_name_parts.append(f"rn-{config.loss_function['rn']}_{config.loss_scale['rn']}")
    if config.loss_scale['tv'] != 0:
        run_name_parts.append(f"tv-{config.loss_function['tv']}_{config.loss_scale['tv']}")
    if config.loss_scale['sclip'] != 0:
        run_name_parts.append(f"sclip-{config.loss_function['sclip']}_{config.loss_scale['sclip']}")
    if config.loss_scale['classification_1k'] != 0:
        run_name_parts.append(f"c1k-{config.loss_scale['classification_1k']}")
    for name, scale in config.loss_scale["avg"].items():
        if scale != 0:
            run_name_parts.append(f"avg-{name}-{scale}")
    for name, scale in config.loss_scale["ref"].items():
        if scale != 0:
            run_name_parts.append(f"ref-{name}-{scale}")
    run_name_extra = "_".join(run_name_parts) + f"_{img_name}"
    run_tags.append("gen3")
    run = init_wandb_run(
        run_name=run_name_extra,
        config=None,
        tags=run_tags
    )
    run_dir = f'./out/reconstructions/{run.id}'
    os.makedirs(run_dir, exist_ok=True)
    wandb_input = {}
    plot(denormalize(img[0]), save_path=f"{run_dir}/input_{img_name}.png", show=False)
    wandb_input[f"img_{img_name}"] = wandb.Image(f"{run_dir}/input_{img_name}.png")
    run.config.update({
            **config.get_as_dict(),
            "use_scale_steps": len(config.scale_steps) > 1,
            "reconstructions_per_img": reconstructions_per_img,
            "image_name": img_name,
            })
    run.log(wandb_input, step=0)
    return run, run_dir

def run_reconstructions(
        img_path,
        config: RunConfig,
        clip_suite: CLIPSuite,
        img_logging_freq=250,
        reconstructions_per_img=4,
        run_tags:list=None,
        ):
    img = clip_suite.image_processor()([PIL.Image.open(img_path)], return_tensors="pt").pixel_values.to(clip_suite.device)
    img_name = os.path.basename(img_path)
    img = img.repeat_interleave(reconstructions_per_img, dim=0)
    clip_projection = clip_suite.image_projector()(img).image_embeds
    surrogate_clip = None
    if config.loss_calculate["sclip"]:
        mapping_model = get_simple_sclip_model()[0].to(clip_suite.device)
        mapping_model.eval()
        surrogate_clip = mapping_model(resnet_proj)
    resnet_proj = clip_suite.resnet()(img)
    resnet_classifications = clip_suite.resnet_classification_1k_layer()(resnet_proj)
    augs = K.AugmentationSequential(
        K.RandomAffine(degrees=30, translate=[0.1, 0.1], scale=[0.7, 1.5], p=0.5, padding_mode="border"),
        same_on_batch=False
    )

    embedding_models = {}

    network_inits = {
        'clip': lambda: (lambda x: clip_suite.image_projector()(x).image_embeds),
        'resnet18': lambda: init_torch_resnet18(no_relu=config.no_relu_for_ref_and_avg),
        'resnet34': lambda: init_torch_resnet34(no_relu=config.no_relu_for_ref_and_avg),
        'resnet50': lambda: init_torch_resnet50(no_relu=config.no_relu_for_ref_and_avg),
        'resnet101': lambda: init_torch_resnet101(no_relu=config.no_relu_for_ref_and_avg),
        'resnet152': lambda: init_torch_resnet152(no_relu=config.no_relu_for_ref_and_avg),
        'vgg16': lambda: init_torch_vgg16(no_relu=config.no_relu_for_ref_and_avg),
        'vgg19': lambda: init_torch_vgg19(no_relu=config.no_relu_for_ref_and_avg),
        'identity': lambda: init_torch_identity(),
    }
    for name, calc in config.loss_calculate["avg"].items():
        if calc:
            embedding_models[name] = network_inits[name]()

    for name, calc in config.loss_calculate["ref"].items():
        if calc and name not in embedding_models:
            embedding_models[name] = network_inits[name]()

    run, run_dir = init_run(img_name, config, run_tags, img, reconstructions_per_img)

    scale_steps = config.scale_steps
    if config.optimization_space == "image":
        if config.init_method == "gaussian":
            optimization_target = torch.randn((img.size(0), 3, scale_steps[-1], scale_steps[-1]), device=clip_suite.device)
        elif config.init_method == "gan_v1":
            generator = DCGenerator(input_size=2048, base_features=128).to('cuda')
            generator_id = "DCGen_64_128_2048"
            get_latest_checkpoint(generator, model_id=generator_id)
            encoder_initial = init_torch_resnet()

            for param in generator.parameters():
                param.requires_grad = False
            for param in encoder_initial.parameters():
                param.requires_grad = False

            mapper = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(0.2),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
            ).to('cuda')

            get_latest_checkpoint(mapper, model_id="mapper-fc-bn-relu-fc-bn")

            optimization_target = (generator(mapper(resnet_proj)) / (1 + config.init_noising_factor)).detach()
            optimization_target = optimization_target + config.init_noising_factor * torch.randn_like(reconstruction) / (1 + config.init_noising_factor)
    if config.optimization_space == "gan_latent":
        generator = DCGenerator(input_size=2048, base_features=128).to('cuda')
        generator_id = "DCGen_64_128_2048"
        get_latest_checkpoint(generator, model_id=generator_id)
        if config.init_method == "gaussian":
            optimization_target = torch.randn((img.size(0), 2048), device=clip_suite.device)
        elif config.init_method == "resnet_to_latent_mapper":
            mapper = nn.Sequential(
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
                nn.LeakyReLU(0.2),
                nn.Linear(2048, 2048),
                nn.BatchNorm1d(2048),
            ).to('cuda')
            get_latest_checkpoint(mapper, model_id="mapper-fc-bn-relu-fc-bn")
            optimization_target = mapper(resnet_proj).detach()

    optimization_target.requires_grad_()
    optimization_space = config.optimization_space
    learning_rate = config.learning_rate
    for i in range(config.iterations):
        if i in config.optimization_space_changes.keys():
            if config.optimization_space_changes[i] == "image":
                optimization_space = "image"
                optimization_target = reconstruction
                optimization_target.requires_grad_()
        if i in config.learning_rate_changes.keys():
            learning_rate = config.learning_rate_changes[i]

        if optimization_space == "image":
            if i in scale_steps.keys():
                optimization_target = F.interpolate(optimization_target, size=(scale_steps[i], scale_steps[i]), mode="bicubic")
            reconstruction = optimization_target
            batch = augs(optimization_target)
        elif optimization_space == "gan_latent":
            reconstruction = generator(optimization_target)
            batch = reconstruction
        batch = F.interpolate(batch, size=(224, 224), mode="bicubic")

        loss = collect_losses(batch, surrogate_clip, clip_projection, resnet_proj, resnet_classifications, clip_suite, config, i, img, embedding_models)
        wandb.log({
            "current_resolution": reconstruction.size(2),
            "current_learning_rate": learning_rate,
            }, step=i)

        grad = torch.autograd.grad([loss], [optimization_target])[0]
        batch_norms = grad.view(grad.size(0), -1).norm(dim=1)
        grad = grad / batch_norms.view(-1, *([1] * (grad.dim() - 1)))

        optimization_target = optimization_target - learning_rate*grad
        if i % img_logging_freq == 0 or i == config.iterations - 1:
            img_path = f"{run_dir}/reconstruction_{img_name}_{i:04d}.png"
            plot(denormalize(reconstruction), cols=reconstructions_per_img, show=False, save_path=img_path)
            wandb.log({f"reconstruction_{img_name}": wandb.Image(img_path)}, step=i)
            for idx in range(reconstruction.size(0)):
                wandb.log({f"reconstruction_{img_name}_{idx}": wandb.Image(denormalize(reconstruction[idx]).cpu())}, step=i)
    
    run.finish()
    return {"reconstructions": reconstruction}