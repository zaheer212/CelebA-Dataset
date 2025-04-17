# CelebA-Dataset

Here’s a rephrased version of your DCGAN implementation documentation. I’ve made it clearer and slightly more concise while keeping the technical accuracy intact:

---

# DCGAN

## DCGAN for Face Image Generation

This repository presents a PyTorch-based implementation of a Deep Convolutional Generative Adversarial Network (DCGAN) to generate realistic face images using the CelebA dataset. It includes data preprocessing, model training, and testing workflows.

---

## Dataset Preparation

### 1. Download the CelebA Dataset
- Obtain the CelebA dataset from its [official source](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or other trusted repositories. Save it as `celeba.zip`.

### 2. Extract the Dataset
- Place `celeba.zip` in your working directory and extract it using the following snippet:

```python
import zipfile
import os

base_dir = "path/to/your/directory"
zip_path = os.path.join(base_dir, "celeba.zip")
extract_path = base_dir
dataset_path = os.path.join(base_dir, "img_align_celeba")

if not os.path.exists(dataset_path):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
```

### 3. Image Transformation
- Images are resized, center-cropped, converted to tensors, and normalized to the range [-1, 1]:

```python
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### 4. Load the Dataset
- Load the dataset using `ImageFolder` and prepare a DataLoader for training:

```python
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
```

---

## Model Training

### 1. Initialize Generator and Discriminator
- Define and initialize the Generator and Discriminator with random weights:

```python
latent_dim = 100
ngf, ndf, n_channel = 64, 64, 3

netG = Generator(latent_dim, ngf, n_channel).to(device)
netD = Discriminator(n_channel, ndf).to(device)
```

### 2. Set Loss Function and Optimizers
- Use Binary Cross-Entropy Loss and Adam optimizers:

```python
criterion = nn.BCELoss().to(device)
lr, beta1 = 0.0002, 0.5
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
```

### 3. Training Loop
- Alternate training between Discriminator and Generator:

```python
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Update Discriminator
        netD.zero_grad()
        labels = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        output = netD(real_images).view(-1)
        lossD_real = criterion(output, labels)
        lossD_real.backward()

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        labels.fill_(fake_label)
        output = netD(fake_images.detach()).view(-1)
        lossD_fake = criterion(output, labels)
        lossD_fake.backward()

        lossD = lossD_real + lossD_fake
        optimizerD.step()

        # Update Generator
        netG.zero_grad()
        labels.fill_(real_label)
        output = netD(fake_images).view(-1)
        lossG = criterion(output, labels)
        lossG.backward()
        optimizerG.step()
```

### 4. Save Checkpoints
- Save model weights and generated images at the end of each epoch:

```python
torch.save(netG.state_dict(), os.path.join(output_dir, "generator.pth"))
torch.save(netD.state_dict(), os.path.join(output_dir, "discriminator.pth"))
```

---

## Model Testing

### 1. Load Trained Generator
- Load the saved Generator model for inference:

```python
netG = Generator(latent_dim, ngf, n_channel).to(device)
netG.load_state_dict(torch.load(generator_path, map_location=device))
netG.eval()
```

### 2. Generate New Images
- Create fake images from random noise vectors:

```python
noise = torch.randn(num_samples, latent_dim, 1, 1, device=device)
with torch.no_grad():
    fake_images = netG(noise)
```

### 3. Save and Visualize Outputs
- Save the generated images and visualize them using Matplotlib:

```python
save_path = os.path.join(output_dir, "generated_faces.png")
vutils.save_image(fake_images, save_path, normalize=True)

plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(vutils.make_grid(fake_images_cpu, padding=2, normalize=True).permute(1, 2, 0))
plt.show()
```

---

## Output Expectations

- **Generated Images**: Realistic face images that mimic those in the CelebA dataset.
- **Training Metrics**: Discriminator and Generator losses should steadily decrease.
- **Saved Models**: Generator and Discriminator checkpoints (`generator.pth`, `discriminator.pth`) will be saved for future use.

---

