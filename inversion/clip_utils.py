from .utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPModel
import PIL

class CLIPSuite:
    def __init__(self, diffusion_model_name: str = "CompVis/stable-diffusion-v1-4", clip_model_name: str = "openai/clip-vit-large-patch14", dtype: torch.dtype = torch.float32):
        self.clip_model_name = clip_model_name
        self.diffusion_model_name = diffusion_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.__vae = None
        self.__unet =  None
        self.__text_encoder =  None
        self.__text_projector =  None
        self.__image_encoder =  None
        self.__image_projector =  None
        self.__clip_model =  None
        self.__tokenizer =  None
        self.__scheduler =  None
        self.__image_processor =  None
        self.__resnet =  None
        self.__classification_1k_layer = None
        self.__lpips_loss_alex = None
        self.__lpips_loss_vgg = None
        self.__lpips_loss_squeeze = None
    
    def vae(self):
        if self.__vae is None:
            self.__vae = AutoencoderKL.from_pretrained(self.diffusion_model_name, subfolder="vae", torch_dtype=self.dtype).to(self.device)
        return self.__vae
    def unet(self):
        if self.__unet is None:
            self.__unet = UNet2DConditionModel.from_pretrained(self.diffusion_model_name, subfolder="unet", torch_dtype=self.dtype).to(self.device)
        return self.__unet
    def text_encoder(self):
        if self.__text_encoder is None:
            self.__text_encoder = CLIPTextModel.from_pretrained(self.clip_model_name, torch_dtype=self.dtype).to(self.device)
        return self.__text_encoder
    def text_projector(self):
        if self.__text_projector is None:
            self.__text_projector = CLIPTextModelWithProjection.from_pretrained(self.clip_model_name, torch_dtype=self.dtype).to(self.device)
        return self.__text_projector
    def image_encoder(self):
        if self.__image_encoder is None:
            self.__image_encoder = CLIPVisionModel.from_pretrained(self.clip_model_name, torch_dtype=self.dtype).to(self.device)
        return self.__image_encoder
    def image_projector(self):
        if self.__image_projector is None:
            self.__image_projector = CLIPVisionModelWithProjection.from_pretrained(self.clip_model_name, torch_dtype=self.dtype).to(self.device)
        return self.__image_projector
    def clip_model(self):
        if self.__clip_model is None:
            self.__clip_model = CLIPModel.from_pretrained(self.clip_model_name, torch_dtype=self.dtype).to(self.device)
        return self.__clip_model
    def tokenizer(self):
        if self.__tokenizer is None:
            self.__tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        return self.__tokenizer
    def scheduler(self):
        if self.__scheduler is None:
            self.__scheduler = DDIMScheduler.from_pretrained(self.diffusion_model_name, subfolder="scheduler")
        return self.__scheduler
    def image_processor(self):
        if self.__image_processor is None:
            # Configuration parameters
            # see https://huggingface.co/openai/clip-vit-large-patch14
            config = {
                "crop_size": 224,
                "do_center_crop": True,
                "do_normalize": True,
                "do_resize": True,
                "feature_extractor_type": "CLIPFeatureExtractor",
                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                "image_std": [0.26862954, 0.26130258, 0.27577711],
                "resample": 3,
                "size": 224
            }
            self.__image_processor = CLIPImageProcessor.from_pretrained(self.clip_model_name, config=config)
        return self.__image_processor

    def resnet(self):
        if self.__resnet is None:
            self.__resnet = init_torch_resnet().to(self.device)
        return self.__resnet

    def resnet_classification_1k_layer(self):
        if self.__classification_1k_layer is None:
            self.__classification_1k_layer = get_torch_restnet_classification_layer().to(self.device)
        return self.__classification_1k_layer
    
    def lpips_loss_alex(self):
        if self.__lpips_loss_alex is None:
            self.__lpips_loss_alex = lpips_loss('alex')
        return self.__lpips_loss_alex
    
    def lpips_loss_vgg(self):
        if self.__lpips_loss_vgg is None:
            self.__lpips_loss_vgg = lpips_loss('vgg')
        return self.__lpips_loss_vgg
    
    def lpips_loss_squeeze(self):
        if self.__lpips_loss_squeeze is None:
            self.__lpips_loss_squeeze = lpips_loss('squeeze')
        return self.__lpips_loss_squeeze
    
    def perform_diffusion(self, latents, steps, diffusion_interval=None):
        if diffusion_interval is None:
            diffusion_interval = [0, steps]
        scheduler = self.scheduler()
        unet = self.unet()
        tokenizer = self.tokenizer()
        text_encoder = self.text_encoder()
        latents = latents * scheduler.init_noise_sigma
        latent_noise = torch.randn(latents.shape, device=self.device, dtype=torch.float32)  # Use float32 for lower memory consumption
        latent_noise = latent_noise * scheduler.init_noise_sigma
        latents = latents #+ latent_noise*(diffusion_interval[1]-diffusion_interval[0])

        # Set scheduler parameters
        scheduler.set_timesteps(steps)
        timesteps = scheduler.timesteps
        with torch.no_grad():
            text_tokens= tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt").input_ids.to(self.device)
            uncond_embeddings = text_encoder(input_ids=text_tokens).last_hidden_state
            uncond_embeddings = uncond_embeddings.repeat(latents.size(0), 1, 1)

        # Denoising loop
        for t in timesteps[diffusion_interval[0]:diffusion_interval[1]]:
            # Predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latents, t, encoder_hidden_states=uncond_embeddings).sample
            
            # Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, eta=0.0).prev_sample
            
            # Free up memory
            del noise_pred
            torch.cuda.empty_cache()

        del uncond_embeddings
        # Decode the latent space to images
        with torch.no_grad():
            return  latents
    
    def get_preprocessed_images(self, image_paths):
        return self.image_processor()([PIL.Image.open(img_path) for img_path in image_paths], return_tensors="pt").pixel_values.to(self.device)

def get_simple_sclip_model():
    mapping_model = nn.Sequential(
        nn.Linear(2048, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024, 1024),
    )
    mapping_model_id = "rn_to_imclip_linear_mapping_2048_1024_1024"
    get_latest_checkpoint(mapping_model, model_id=mapping_model_id)
    return mapping_model, mapping_model_id

def mse_dim_normalized(x, y):
    return F.mse_loss(x, y) / (x.shape[-1] ** 0.5)

def cosine_distance(x, y):
    return 1 - F.cosine_similarity(x, y)

def surrogate_clip_loss(reconstructions, surrogate_clip, clip, rn, clip_suite: CLIPSuite, distance_measure=cosine_distance):
    return distance_measure(clip_suite.image_encoder()(reconstructions).pooler_output, surrogate_clip)
    
def clip_projection_loss(reconstructions, surrogate_clip, clip, rn, clip_suite: CLIPSuite, distance_measure=cosine_distance):
    return distance_measure(clip_suite.image_projector()(reconstructions).image_embeds, clip)

def resnet_projection_loss(reconstructions, surrogate_clip, clip, rn, clip_suite: CLIPSuite, distance_measure=cosine_distance):
    return distance_measure(clip_suite.resnet()(reconstructions), rn)