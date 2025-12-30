import torch
import numpy as np
import copy
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

class LandscapeUtils:
    @staticmethod
    def compute_loss_surface(trainer, grid_size=10):
        """
        Computes 3D loss surface around the training trajectory using PCA directions.
        Returns:
            X, Y grids (2D)
            Z grid (Loss values)
            traj_X, traj_Y (Trajectory projected onto 2D plane)
            traj_Z (Actual loss values of trajectory)
        """
        snapshots = trainer.weight_snapshots
        if len(snapshots) < 2:
            print("Not enough snapshots for landscape.")
            return None, None, None, None, None, None

        # 1. Vectorize weights
        # We flatten all trainable parameters into a single vector W
        def vectorize(state_dict):
            vec = []
            for k, v in state_dict.items():
                 if v.numel() > 0: # Ensure valid tensor
                     vec.append(v.view(-1).numpy())
            return np.concatenate(vec)

        weight_vectors = [vectorize(s) for s in snapshots]
        weight_matrix = np.stack(weight_vectors) # (T, D)
        
        # 2. PCA calculation on the trajectory
        # We want to find the 2 main directions of movement
        # Center the data first? Yes, usually around the final point or mean.
        # Let's use final point as origin for the plot
        w_final = weight_vectors[-1]
        
        # Compute differences (directions moved)
        # Using PCA on the trajectory points themselves
        pca = PCA(n_components=2)
        pca.fit(weight_matrix)
        
        direction1 = pca.components_[0] # (D,)
        direction2 = pca.components_[1] # (D,)
        
        # 3. Project trajectory onto these 2 directions
        # Coordinate x = dot(W_t - W_mean, d1)
        # We can just transform the matrix
        traj_2d = pca.transform(weight_matrix) # (T, 2)
        traj_x = traj_2d[:, 0]
        traj_y = traj_2d[:, 1]
        
        # Traj Z is just the loss history (sampled at snapshots)
        # We don't have exact loss at snapshot method, let's assume end-of-epoch loss
        # Or re-evaluate. Re-evaluation is safer.
        
        # 4. Generate Grid
        x_min, x_max = traj_x.min(), traj_x.max()
        y_min, y_max = traj_y.min(), traj_y.max()
        
        padding = 0.5 * max(x_max-x_min, y_max-y_min)
        if padding == 0: padding = 1.0
        
        x_range = np.linspace(x_min - padding, x_max + padding, grid_size)
        y_range = np.linspace(y_min - padding, y_max + padding, grid_size)
        
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.zeros_like(X)
        
        # 5. Evaluate Loss on Grid
        # Need a fixed batch of data
        device = trainer.device
        model = trainer.model
        
        if hasattr(trainer, 'task_type') and trainer.task_type == "super_resolution":
             criterion = torch.nn.MSELoss()
             # Use CIFAR10 for consistency (same as training)
             transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
             dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
             is_sr = True
             is_seg = False
        elif hasattr(trainer, 'task_type') and trainer.task_type == "segmentation":
             # Need to import ShapesDataset or define minimal one here (or import from trainer)
             # Better to import. 
             from src.core.trainer import ShapesDataset
             criterion = torch.nn.CrossEntropyLoss()
             dataset = ShapesDataset(size=100, img_size=64)
             is_sr = False
             is_seg = True
             is_diff = False
        elif hasattr(trainer, 'task_type') and trainer.task_type == "diffusion":
             criterion = torch.nn.MSELoss()
             transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)) 
             ])
             dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
             is_sr = False
             is_seg = False
             is_diff = True
        else:
             criterion = torch.nn.CrossEntropyLoss()
             transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
             dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
             is_sr = False
             is_seg = False
             is_diff = False
             
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        inputs, labels = next(iter(loader))
        
        # Prepare fixed inputs/targets for the whole grid
        if is_sr:
             hr_imgs = inputs.to(device)
             lr_imgs = torch.nn.functional.interpolate(hr_imgs, scale_factor=0.5, mode='bicubic')
             lr_imgs = torch.nn.functional.interpolate(lr_imgs, scale_factor=2.0, mode='bicubic')
             inputs = lr_imgs
             targets = hr_imgs
        elif is_seg:
            inputs, targets = inputs.to(device), labels.to(device)
        elif is_diff:
            # Need strict inputs (noisy x) and targets (noise) + t
            inputs = inputs.to(device)
            B = inputs.shape[0]
            # Pick a fixed t for landscape visualization (e.g. t=100)
            t = torch.full((B,), 100, device=device).long()
            noise = torch.randn_like(inputs)
            
            # Add noise manually assuming linear schedule approximation or use trainer properties if possible
            # To be safe and self-contained, let's use the same schedule as trainer
            T_val = 200
            beta = torch.linspace(0.0001, 0.02, T_val).to(device)
            alpha = 1.0 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            
            a_bar_t = alpha_bar[t].view(-1, 1, 1, 1)
            noisy_images = torch.sqrt(a_bar_t) * inputs + torch.sqrt(1 - a_bar_t) * noise
            
            # For visualization, we evaluate loss = ||eps_pred - eps||
            inputs = noisy_images
            targets = noise
            # We must pass 't' to model forward
        else:
             inputs, targets = inputs.to(device), labels.to(device)
        
        # Pre-calculate shapes to reconstruct vectors back to state_dict
        ref_state = snapshots[0]
        shapes = {k: v.shape for k, v in ref_state.items()}
        sizes = {k: v.numel() for k, v in ref_state.items()}
        keys = list(ref_state.keys())
        
        # Helper to set weights from vector
        def set_weights_from_vec(w_vec):
            offset = 0
            new_state = {}
            for k in keys:
                numel = sizes[k]
                if numel == 0: continue
                # Slice and reshape
                chunk = w_vec[offset:offset+numel]
                # Convert back to tensor
                new_state[k] = torch.tensor(chunk.reshape(shapes[k])).to(device)
                offset += numel
            # Load into model
            # We use load_state_dict with strict=False if anything missing, but shouldn't be
            model.load_state_dict(new_state, strict=False)

        # Base mean usually used in PCA, but here we reconstructed relative to PCA center
        # PCA inverse transform does: X @ components + mean
        
        print("Computing Loss Landscape (this might take a moment)...")
        for i in range(grid_size):
            for j in range(grid_size):
                # Construct weight vector
                # coord (X[i,j], Y[i,j]) in PC space
                pc_coord = np.array([[X[i,j], Y[i,j]]])
                w_recon = pca.inverse_transform(pc_coord)[0]
                
                set_weights_from_vec(w_recon)
                
                with torch.no_grad():
                    if is_diff:
                        output = model(inputs, t.view(-1, 1).float()) # pass t
                    else:
                        output = model(inputs)
                    loss = criterion(output, targets)
                Z[i,j] = loss.item()

        # Restore original model
        model.load_state_dict(snapshots[-1])
        model.to(device) # Ensure back on device
        
        # Calculate trajectory Z (loss) using the same batch for consistency
        traj_z = []
        for vec in weight_vectors:
             set_weights_from_vec(vec)
             with torch.no_grad():
                 if is_diff:
                     output = model(inputs, t.view(-1, 1).float())
                 else:
                     output = model(inputs)
                 loss = criterion(output, targets)
             traj_z.append(loss.item())

        return X, Y, Z, traj_x, traj_y, np.array(traj_z)
