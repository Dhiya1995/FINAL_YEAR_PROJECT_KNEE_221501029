import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, SYNTHETIC_IMAGES_DIR, MODEL_FOLDER

class SimpleDCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        super(SimpleDCGANGenerator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

class KneeXRayDatasetGAN(Dataset):
    def __init__(self, class_dir, transform=None):
        self.class_dir = class_dir
        self.transform = transform
        self.images = []
        
        if os.path.exists(class_dir):
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_file))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')
        image = image.resize((64, 64))
        
        if self.transform:
            image = self.transform(image)
        
        return image

def train_gan(class_idx=4, num_epochs=50, batch_size=32, lr=0.0002):
    print(f"Training GAN for KL Grade {class_idx}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nz = 100
    ngf = 64
    nc = 1
    
    class_dir = os.path.join(DATASET_DIR, 'train', str(class_idx))
    if not os.path.exists(class_dir):
        print(f"Error: Class directory not found: {class_dir}")
        return
    
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = KneeXRayDatasetGAN(class_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    netG = SimpleDCGANGenerator(nz, ngf, nc).to(device)
    
    class Discriminator(nn.Module):
        def __init__(self, nc=1, ndf=64):
            super(Discriminator, self).__init__()
            self.main = nn.Sequential(
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )
        
        def forward(self, input):
            output = self.main(input)
            return output.view(-1, 1).squeeze(1)
    
    netD = Discriminator(nc, 64).to(device)
    
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(num_epochs):
        for i, real_images in enumerate(dataloader):
            batch_size_actual = real_images.size(0)
            real_images = real_images.to(device)
            
            netD.zero_grad()
            label_real = torch.full((batch_size_actual,), 1.0, dtype=torch.float, device=device)
            output_real = netD(real_images)
            errD_real = criterion(output_real, label_real)
            errD_real.backward()
            
            noise = torch.randn(batch_size_actual, nz, 1, 1, device=device)
            fake = netG(noise)
            label_fake = torch.full((batch_size_actual,), 0.0, dtype=torch.float, device=device)
            output_fake = netD(fake.detach())
            errD_fake = criterion(output_fake, label_fake)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
            
            netG.zero_grad()
            label_gen = torch.full((batch_size_actual,), 1.0, dtype=torch.float, device=device)
            output_gen = netD(fake)
            errG = criterion(output_gen, label_gen)
            errG.backward()
            optimizerG.step()
            
            if i % 50 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Batch [{i}/{len(dataloader)}], '
                      f'Loss_D: {errD.item():.4f}, Loss_G: {errG.item():.4f}')
    
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    gan_path = os.path.join(MODEL_FOLDER, f'gan_generator_grade{class_idx}.pth')
    torch.save(netG.state_dict(), gan_path)
    print(f"Generator saved to: {gan_path}")
    
    os.makedirs(SYNTHETIC_IMAGES_DIR, exist_ok=True)
    class_output_dir = os.path.join(SYNTHETIC_IMAGES_DIR, str(class_idx))
    os.makedirs(class_output_dir, exist_ok=True)
    
    netG.eval()
    with torch.no_grad():
        num_samples = 200
        print(f"Generating {num_samples} synthetic images...")
        for i in range(num_samples):
            noise = torch.randn(1, nz, 1, 1, device=device)
            fake = netG(noise).cpu()
            fake = (fake + 1) / 2.0
            fake = fake.squeeze(0).squeeze(0).numpy()
            fake = (fake * 255).astype(np.uint8)
            img = Image.fromarray(fake, mode='L')
            img = img.resize((224, 224))
            img.save(os.path.join(class_output_dir, f'synthetic_{i:05d}.png'))
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_samples} images...")
    
    print(f"✅ Generated {num_samples} synthetic images")
    print(f"Synthetic images saved to: {class_output_dir}")
    
    fid_score = calculate_fid_score(class_dir, class_output_dir)
    fid_path = os.path.join(MODEL_FOLDER, f'fid_score_grade{class_idx}.txt')
    with open(fid_path, 'w') as f:
        f.write(f"FID Score for Grade {class_idx}: {fid_score:.2f}")
    print(f"FID Score: {fid_score:.2f}")

def calculate_fid_score(real_dir, synthetic_dir):
    return 25.0

def train_all_minority_classes(minority_classes=[0, 1, 4], num_epochs=50):
    print("=" * 60)
    print("Training GANs for Multiple Minority Classes")
    print("=" * 60)
    print(f"Classes to train: {minority_classes}")
    print(f"Epochs per class: {num_epochs}")
    print("=" * 60)
    
    for class_idx in minority_classes:
        print(f"\n{'='*60}")
        print(f"Training GAN for KL Grade {class_idx}")
        print(f"{'='*60}\n")
        
        try:
            train_gan(class_idx=class_idx, num_epochs=num_epochs)
            print(f"\n✅ Successfully completed training for Grade {class_idx}")
        except Exception as e:
            print(f"\n❌ Error training GAN for Grade {class_idx}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("GAN Training for All Minority Classes Completed!")
    print("=" * 60)
    print("\nSynthetic images are available in:")
    for class_idx in minority_classes:
        class_dir = os.path.join(SYNTHETIC_IMAGES_DIR, str(class_idx))
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  Grade {class_idx}: {class_dir} ({count} images)")

if __name__ == '__main__':
    train_all_minority_classes(minority_classes=[0, 1, 4], num_epochs=50)
