import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from inversion.GAN import *
from inversion.utils import *

train_loader, test_loader = get_imagenet_loader(batch_size=256, resolution=64)

# Instantiate the generator and discriminator
base_features = 512
input_size = 128
generator = DCGenerator(input_size=input_size, base_features=base_features).to('cuda')
discriminator = DCDiscriminator(base_features=base_features).to('cuda')

# Define the optimizer and loss function
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()
criterionD = nn.MSELoss()

generator_id = "DCGen_64_512_128"
discriminator_id = "DCDis_64_512"
get_latest_checkpoint(generator, model_id=generator_id)
get_latest_checkpoint(discriminator, model_id=discriminator_id)

fake_history = None

run = init_wandb_run(
    project="inversion-gan",
    run_name=f"{generator_id}_{discriminator_id}",
    config={
        "base_features": base_features,
        "gan_latent_size": input_size,
        "batch_size": 256,
        "resolution": 64,
        "generator_id": generator_id,
        "discriminator_id": discriminator_id,
        "learning_rate": 0.0002,
        "betas": (0.5, 0.999)
    },
    tags=["DCGAN", "64x64", "gen1"]
)

run_dir = f"./out/gan/{run.id}"
os.makedirs(run_dir, exist_ok=True)

# Training loop
num_epochs = 100
for epoch in range(0, num_epochs):
    for i, (images, _) in enumerate(train_loader):
        start_time = time.time()
        # Move images to CUDA
        images = images.to('cuda')

        # Create labels
        valid = torch.full((images.size(0), 1), 0.9, device='cuda', dtype=torch.float32).view(-1, 1, 1, 1)
        fake = torch.zeros(images.size(0), 1, device='cuda', dtype=torch.float32).view(-1, 1, 1, 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Generate a batch of images
        z = torch.randn(images.size(0), generator.nz, device='cuda')
        gen_images = generator(z)
        if fake_history is None:
            fake_history = gen_images.detach().clone()
        else: 
            indices = torch.randint(low=0, high=gen_images.size(0), size=(1,))
            fake_history[indices] = gen_images[indices].detach().clone()
        # Calculate the percentage of correctly detected fake images
        with torch.no_grad():
            prediction = discriminator(gen_images.detach()).round()
            detected_fakes = (prediction == 0).sum().item()
            total_fake = fake.numel()
            d_detection_accuracy = detected_fakes / total_fake
        optimizer_D.zero_grad()
        # Real images
        d_loss_on_real = criterionD(discriminator(images), valid)
        # Fake images
        d_values_on_generated = discriminator(gen_images.detach())
        d_loss_on_generated = criterionD(d_values_on_generated, fake)
        d_values_on_history = discriminator(fake_history)
        d_loss_on_history = criterionD(d_values_on_history, fake)

        # Total loss
        d_loss = (d_loss_on_real/2 + (d_loss_on_generated)/2 + d_loss_on_history/2)

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Loss measures generator's ability to fool the discriminator
        g_loss = criterion(torch.sigmoid(discriminator(gen_images)-0.5), valid/0.9)

        g_loss.backward()
        optimizer_G.step()

        run.log({
            "time_per_step": time.time() - start_time if i > 0 else 0,
            "d_loss": d_loss.item(),
            "g_loss": g_loss.item(),
            "d_detection_accuracy": d_detection_accuracy,
            "d_values_on_generated_avg": d_values_on_generated.mean().item(),
            "d_values_on_history_avg": d_values_on_history.mean().item()
        }, step=epoch * len(train_loader) + i)

        if i % 200 == 0:
            print(f"[Epoch {epoch:4d}/{num_epochs:2d}]\n" +
                f" [Batch {i:5d}/{len(train_loader):5d}\n" + 
                f" [D loss: {d_loss.item()} = {d_loss_on_real.item()} + {d_loss_on_generated.item()}]\n" + 
                f" [G loss: {g_loss.item()}]\n" + 
                f" [Fake accuracy: {d_detection_accuracy:.2f}]")
            image_path_generated = f"{run_dir}/generated_{i:04d}.png"
            image_path_history = f"{run_dir}/history_{i:04d}.png"
            plot(denormalize(fake_history[:100]),
                    texts=[f"{d_values_on_history[i].item():.2f}" for i in range(100)],
                    show=False,
                    save_path=image_path_history)
            plot(denormalize(gen_images[:100]),
                    texts=[f"{d_values_on_generated[i].item():.2f}" for i in range(100)],
                    show=False,
                    save_path=image_path_generated)
            run.log({
                "generated_images": wandb.Image(image_path_generated),
                "history_images": wandb.Image(image_path_history)
            }, step=epoch * len(train_loader) + i)

            
        if i % 3000 == 0:
            checkpoint_suffix = f"epoch_{epoch:02d}"
            save_model(generator, generator_id, checkpoint_suffix=checkpoint_suffix)
            save_model(discriminator, discriminator_id, checkpoint_suffix=checkpoint_suffix)

run.finish()