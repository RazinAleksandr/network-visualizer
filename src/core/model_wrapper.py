import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
from PIL import Image
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 1 input channel (grayscale), 16 output channels, 3x3 conv
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2) # 28x28 -> 14x14
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2) # 14x14 -> 7x7
        
        self.flatten = nn.Flatten()
        
        # 32 channels * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Linear(128, 10) # 10 Output classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class MnistWrapper:
    def __init__(self, model_path='mnist_cnn.pth'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN().to(self.device)
        self.model_path = model_path
        
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print("Model not found. Training new model...")
            self._train_model()
            
        self.model.eval()
        
        self.activations = {}
        self.hooks = []
        
        # Define layers to partial visualization
        self.layer_mapping = {
            'conv1': 'Layer 1 (Conv)', 
            'pool1': 'Layer 1 (Pool)',
            'conv2': 'Layer 2 (Conv)',
            'pool2': 'Layer 2 (Pool)',
            'fc1': 'Layer 3 (Dense)',
            'fc2': 'Output (Class)'
        }
        
        self._register_hooks()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _train_model(self):
        # Quick training loop
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        # Use substring of data for speed if needed, but MNIST is small enough
        loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        print("Starting training...")
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
                
        print("Training complete.")
        torch.save(self.model.state_dict(), self.model_path)

    def _register_hooks(self):
        self.activations = {}
        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach()
            return hook

        for name, layer in self.model.named_children():
            # We want specific layers
            if name in self.layer_mapping:
                 readable_name = self.layer_mapping[name]
                 layer.register_forward_hook(get_activation(readable_name))

    def process_image(self, image_input):
        # image_input can be tensor or PIL
        if isinstance(image_input, torch.Tensor):
             input_tensor = image_input
             if input_tensor.ndim == 2:
                 input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
             elif input_tensor.ndim == 3:
                 input_tensor = input_tensor.unsqueeze(0)
        else:
             # Assume PIL
             input_tensor = self.preprocess(image_input).unsqueeze(0)
             
        input_tensor = input_tensor.to(self.device)
        
        self.activations = {}
        with torch.no_grad():
            output = self.model(input_tensor)
            
        return self.activations, output

    def get_random_sample(self):
        test_dataset = datasets.MNIST('./data', train=False, download=True, 
                                     transform=transforms.Compose([transforms.ToTensor()])) # No norm for visualization
        idx = np.random.randint(0, len(test_dataset))
        return test_dataset[idx] # Returns (tensor, label)
