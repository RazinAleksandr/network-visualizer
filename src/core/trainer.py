import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import os
import numpy as np

class ShapesDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, img_size=64):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate random image
        img = np.zeros((self.img_size, self.img_size), dtype=np.float32)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        
        # Add random shapes
        num_shapes = np.random.randint(1, 4)
        for _ in range(num_shapes):
            shape_type = np.random.randint(0, 2) # 0: rect, 1: circle
            
            if shape_type == 0: # Rect
                w, h = np.random.randint(5, 20, 2)
                x = np.random.randint(0, self.img_size - w)
                y = np.random.randint(0, self.img_size - h)
                img[y:y+h, x:x+w] = 1.0
                mask[y:y+h, x:x+w] = 1
            else: # Circle
                r = np.random.randint(5, 15)
                cx = np.random.randint(r, self.img_size - r)
                cy = np.random.randint(r, self.img_size - r)
                y, x = np.ogrid[:self.img_size, :self.img_size]
                mask_circle = ((x - cx)**2 + (y - cy)**2 <= r**2)
                img[mask_circle] = 1.0
                mask[mask_circle] = 1
                
        # Noise
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        
        return torch.from_numpy(img).float().unsqueeze(0), torch.from_numpy(mask).long()

class PlatformTrainer:
    def __init__(self, model, model_name, device, task_type="classification"):
        self.model = model
        self.model_name = model_name
        self.device = device
        self.task_type = task_type
        self.history = {'loss': [], 'metric': []} # Renamed accuracy to metric
        self.weight_snapshots = [] 
        
        # Diffusion specific params
        if task_type == "diffusion":
            self.T = 200 # Total time steps (small for speed)
            self.beta = torch.linspace(0.0001, 0.02, self.T).to(device)
            self.alpha = 1.0 - self.beta
            self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def train(self, epochs=1, progress_callback=None):
        """
        Train the model and snapshot weights.
        """
        if self.task_type == "classification":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            criterion = nn.CrossEntropyLoss()
        
        elif self.task_type == "super_resolution":
             transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
             train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
             criterion = nn.MSELoss()
             
        elif self.task_type == "segmentation":
            train_dataset = ShapesDataset(size=500, img_size=64) 
            criterion = nn.CrossEntropyLoss()
            
        elif self.task_type == "diffusion":
            # Train on MNIST
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) # Range [-1, 1] for diffusion
            ])
            train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            criterion = nn.MSELoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        self.weight_snapshots = []
        self._snapshot_weights()
        
        for epoch in range(epochs):
            running_loss = 0.0
            metric_accum = 0.0
            total = 0
            
            for i, (inputs, target) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                
                optimizer.zero_grad()
                
                if self.task_type == "super_resolution":
                    hr_imgs = inputs
                    lr_imgs = torch.nn.functional.interpolate(hr_imgs, scale_factor=0.5, mode='bicubic')
                    lr_imgs = torch.nn.functional.interpolate(lr_imgs, scale_factor=2.0, mode='bicubic')
                    target = hr_imgs
                    outputs = self.model(lr_imgs)
                    loss = criterion(outputs, target)
                    
                elif self.task_type == "diffusion":
                    # 1. Sample t uniformally
                    t = torch.randint(0, self.T, (inputs.shape[0],), device=self.device).long()
                    
                    # 2. Get noise epsilon
                    noise = torch.randn_like(inputs)
                    
                    # 3. Add noise (Forward diffusion)
                    # sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
                    a_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)
                    noisy_images = torch.sqrt(a_bar_t) * inputs + torch.sqrt(1 - a_bar_t) * noise
                    
                    # 4. Predict noise (Reverse process step approximation)
                    # Model predicts the noise that was added
                    predicted_noise = self.model(noisy_images, t.view(-1, 1).float())
                    
                    loss = criterion(predicted_noise, noise)
                    outputs = predicted_noise # For metric logging logic below
                    target = noise
                    
                elif self.task_type == "segmentation":
                    target = target.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, target)
                else: 
                     target = target.to(self.device)
                     outputs = self.model(inputs)
                     loss = criterion(outputs, target)

                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Metrics
                batch_size = inputs.size(0)
                if self.task_type == "classification":
                     _, predicted = outputs.max(1)
                     acc = predicted.eq(target).sum().item() / batch_size * 100
                     metric_accum += acc
                     metric_name = "Accuracy"
                elif self.task_type == "super_resolution":
                    mse = loss.item()
                    if mse == 0: psnr = 100
                    else: psnr = 10 * np.log10(1.0 / mse)
                    metric_accum += psnr
                    metric_name = "PSNR"
                elif self.task_type == "segmentation":
                    pred_mask = torch.argmax(outputs, dim=1) # (B, H, W)
                    target_mask = target
                    
                    intersection = (pred_mask & target_mask).float().sum((1, 2))
                    union = (pred_mask | target_mask).float().sum((1, 2))
                    
                    iou = (intersection + 1e-6) / (union + 1e-6)
                    metric_accum += iou.mean().item()
                    metric_name = "IoU"
                elif self.task_type == "diffusion":
                    # Metric for diffusion is just Loss (MSE) really, or we could check something else
                    # Just use Loss * 1000 for visibility
                    metric_accum += loss.item()
                    metric_name = "MSE Loss"

                total += 1 # Batch count
                
                if i % 10 == 0:
                    current_metric = metric_accum / total if total > 0 else 0
                    if progress_callback:
                        progress_callback(epoch + 1, loss.item(), current_metric, metric_name)
                    
                    self.history['loss'].append(loss.item())
                    self.history['metric'].append(current_metric)
            
            self._snapshot_weights()
            
        print("Training complete.")

    def _snapshot_weights(self):
        # We save CPU state dict
        state = copy.deepcopy(self.model.state_dict())
        for k, v in state.items():
            state[k] = v.cpu()
        self.weight_snapshots.append(state)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
