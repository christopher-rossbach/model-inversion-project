import contextlib
import glob
from io import StringIO
import os
import types
import time
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torchvision import models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import lpips

from Model.image_embedding_model import ImageEmbeddingModel

def load_feature_vector_files():
    file_paths = ["Data/feature_vectors_1.txt","Data/feature_vectors_2.txt","Data/feature_vectors_3.txt"]
    dfs = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep="\t",names=["id", "embedding"])
        dfs.append(df)
    data = pd.concat(dfs, ignore_index=True)

    data['embedding'] = data['embedding'].apply(lambda e: np.array(eval(e), np.float32))
    return data

def get_torch_restnet_classification_layer():
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    fc_layer = resnet50.fc
    return fc_layer

def init_torch_identity():
    identity = torch.nn.Identity()
    identity.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity.to(device)
    return identity

def init_torch_resnet():
    resnet50_v1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50_v1.fc = torch.nn.Identity()
    resnet50_v1.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50_v1.to(device)
    return resnet50_v1

def init_torch_vgg16(no_relu=True):
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    if no_relu:
        vgg16.classifier[4] = torch.nn.Identity()
    vgg16.classifier[6] = torch.nn.Identity()
    vgg16.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg16.to(device)
    return vgg16

def init_torch_vgg19(no_relu=True):
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    if no_relu:
        vgg19.classifier[4] = torch.nn.Identity()
    vgg19.classifier[6] = torch.nn.Identity()
    vgg19.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg19.to(device)
    return vgg19

def base_block_no_relu(bb):
    def foreward_pass(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out
    bb.forward = types.MethodType(foreward_pass, bb)
    return bb

def init_torch_resnet18(no_relu=True):
    resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    if no_relu:
        resnet18.layer4[1] = base_block_no_relu(resnet18.layer4[1])
    resnet18.fc = torch.nn.Identity()
    resnet18.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet18.to(device)
    return resnet18

def init_torch_resnet34(no_relu=True):
    resnet34 = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    if no_relu:
        resnet34.layer4[2] = base_block_no_relu(resnet34.layer4[2])
    resnet34.fc = torch.nn.Identity()
    resnet34.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet34.to(device)
    return resnet34

def bottleneck_block_no_relu(bb):
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #out = self.relu(out)
        return out
    bb.forward = types.MethodType(forward, bb)
    return bb

def resnet_no_relu(rn):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return x
    rn.forward = types.MethodType(forward, rn)
    return rn

def init_torch_resnet50(with_classifier=False, no_relu=True):
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet50.eval()
    if not with_classifier:
        if no_relu:
            resnet50.layer4[2] = bottleneck_block_no_relu(resnet50.layer4[2])
        resnet50.fc = torch.nn.Identity()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet50.to(device)
    return resnet50

def init_torch_resnet101(no_relu=True):
    resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    if no_relu:
        resnet101.layer4[2] = bottleneck_block_no_relu(resnet101.layer4[2])
    resnet101.fc = torch.nn.Identity()
    resnet101.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet101.to(device)
    return resnet101

def init_torch_resnet152(no_relu=True):
    resnet152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    if no_relu:
        resnet152.layer4[2] = bottleneck_block_no_relu(resnet152.layer4[2])
    resnet152.fc = torch.nn.Identity()
    resnet152.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet152.to(device)
    return resnet152

def load_image(image_path: str, resize=None):
    if resize:
        return np.array(Image.open(image_path).convert("RGB").resize(resize, Image.BICUBIC), np.float32)
    else:
        return np.array(Image.open(image_path).convert("RGB"), np.float32)

def plot(tensor, texts=None, cols=None, show=True, save_path=None):
    """
    Plots a tensor as an image or a grid of images using matplotlib.
    Parameters:
        tensor (torch.Tensor): A 3D or 4D tensor to be plotted. If the tensor is 3D, it is assumed to be a single image.
                           If the tensor is 4D, it is assumed to be a batch of images.
        show (bool): Whether to display the plot using plt.show(). Default is True.
        save_path (str or None): If provided, saves the plot to this path.
    Returns:
        None
    """
    tensor = tensor.detach().cpu().clone()
    if len(tensor.shape) == 3:
        plt.imshow(tensor.permute(1, 2, 0).cpu())
    elif tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
    else:
        n = tensor.shape[0]
        grid_size = min(tensor.shape[0], max(4, int(np.ceil(np.sqrt(n)))))
        if cols is None:
            cols = grid_size
        rows = int(np.ceil((tensor.shape[0])/cols))
        fig_width = min(3 * cols, 12)
        fig_height = fig_width * rows / cols
        if texts is not None:
            fig_height *= 1.2
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        for i, im in enumerate(tensor):
            row, col = divmod(i, cols)
            if rows == 1:
                axes[col].imshow(im.permute(1, 2, 0).cpu())
                axes[col].axis('off')
                if texts is not None:
                    axes[col].set_title(texts[i])
            else:
                axes[row, col].imshow(im.permute(1, 2, 0).cpu())
                axes[row, col].axis('off')
                if texts is not None:
                    axes[row, col].set_title(texts[i])
        # Hide any unused subplots
        for j in range(i + 1, cols * rows):
            row, col = divmod(j, cols)
            if rows == 1:
                axes[col].axis('off')
            else:
                axes[row, col].axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

def denormalize_fast(imgs):
    dm_device = dm.detach().to(imgs.device)
    ds_device = ds.detach().to(imgs.device)
    old_shape = imgs.shape
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)
    return (imgs * ds_device.view(1, -1, 1, 1) + dm_device.view(1, -1, 1, 1)).clamp_(0,1).reshape(old_shape)

def denormalize(tensor_in):
    tensor= tensor_in.detach().cpu()
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
    return (tensor * ds.view(1, -1, 1, 1) + dm.view(1, -1, 1, 1)).clamp_(0,1).reshape(tensor_in.shape)

def normalize(tensor):
    if len(tensor.shape) == 4:
        return (tensor - dm.view(1, -1, 1, 1).to(tensor.device)) / ds.view(1, -1, 1, 1).to(tensor.device)
    return (tensor - dm.to(tensor.device)) / ds.to(tensor.device)

dm = torch.as_tensor([0.485, 0.456, 0.406])
ds = torch.as_tensor([0.229, 0.224, 0.225])

def get_image_embedding_model():
    MODEL_FILE_PATH = "Model/resnet_mac_model.onnx"
    IMAGE_DIR = "Images/"
    iem = ImageEmbeddingModel(model_path=MODEL_FILE_PATH, execution_provider_list=["CUDAExecutionProvider"])
    return iem


def get_latest_checkpoint(model, model_id):
    checkpoint_dir = f'./checkpoints/{model_id}/'
    checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pth')))
    if checkpoints:
        print(f"Loading model from {checkpoints[-1]}")
        model.load_state_dict(torch.load(checkpoints[-1]))
    else:
        print(f"model {model_id} not found")
        return model

def save_model(model, model_id, checkpoint_suffix=None):
    checkpoint_dir = f'./checkpoints/{model_id}/'
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = time.strftime('%Y-%m-%dT%H:%M:%S')
    if checkpoint_suffix is None:
        checkpoint_path = os.path.join(checkpoint_dir, f'{timestamp}.pth')
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f'{timestamp}_{checkpoint_suffix}.pth')
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved model to {checkpoint_path}")

def get_imagenet_loader(batch_size, resolution):
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize(resolution),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    # Load the ImageNet training dataset
    train_data = datasets.ImageFolder('/home/woody/iwi1/iwi1106h/data/imagenet/train', transform=transform)
    #train_data = datasets.ImageFolder('/tmp/922711.tinygpu/imagenet/train', transform=transform)

    # Load the ImageNet validation dataset
    val_data = datasets.ImageFolder('/home/woody/iwi1/iwi1106h/data/imagenet/val', transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=8)
    return train_loader, val_loader

def total_variation(img):
    return torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

def init_wandb_run(
    run_name: str,
    project: str = "model-inversion-gen3",
    config: dict = None,
    notes: str = "",
    tags: list = None,
    team: str = "team-cr"
):
    with contextlib.redirect_stdout(StringIO()):
        run = wandb.init(
            project=project,
            entity=team,
            name=run_name,
            config=config,
            notes=notes,
            tags=tags
        )
        return run

def lpips_loss(net_type='alex'):
    """
    Available networks: alex, vgg, squeeze
    """
    loss_fn = lpips.LPIPS(net=net_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = loss_fn.to(device)
    return loss_fn