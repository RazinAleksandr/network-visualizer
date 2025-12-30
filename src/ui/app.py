import streamlit as st
import sys
import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
from PIL import Image
import importlib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import components
import src.core.models
import src.core.landscape
import src.core.trainer
importlib.reload(src.core.models)
importlib.reload(src.core.landscape)
importlib.reload(src.core.trainer)

from src.core.models import get_model
from src.core.trainer import PlatformTrainer
from src.core.landscape import LandscapeUtils
from src.visualization.visualizer import NetworkVisualizer
from src.visualization.training_viz import TrainingVisualizer

st.set_page_config(page_title="Neural Network Platform", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# SESSION STATE MANAGEMENT
# ----------------------------------
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'Inference'

if 'current_model_name' not in st.session_state:
    st.session_state['current_model_name'] = 'SimpleCNN'

if 'trainer' not in st.session_state:
    st.session_state['trainer'] = None

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.title("CV Platform")
st.sidebar.markdown("---")

task = st.sidebar.selectbox("Task", ["MNIST Classification", "Super Resolution", "Segmentation", "Image Generation"])

# Filter models by task
if task == "MNIST Classification":
    available_models = ["SimpleCNN", "ResNet18", "AlexNet"]
    task_type = "classification"
elif task == "Super Resolution":
    available_models = ["SRCNN"]
    task_type = "super_resolution"
elif task == "Segmentation":
    available_models = ["SimpleUNet"]
    task_type = "segmentation"
else:
    available_models = ["SimpleDiffUNet"]
    task_type = "diffusion"

model_name = st.sidebar.selectbox("Model Architecture", available_models)
st.session_state['current_model_name'] = model_name

mode = st.sidebar.radio("Mode", ["Training", "Inference"])

# ----------------------------------
# TRAINING TAB
# ----------------------------------
if mode == "Training":
    st.title(f"Training Studio: {model_name}")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.info("Train a fresh model from scratch.")
        epochs = st.slider("Epochs", 1, 20, 2)
        lr = st.select_slider("Learning Rate", [0.01, 0.001, 0.0001], value=0.001)
        
        if st.button("Start Training"):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = get_model(model_name).to(device)
            trainer = PlatformTrainer(model, model_name, device, task_type=task_type)
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            chart_placeholder = st.empty()
            
            def progress_callback(epoch, loss, metric, metric_name="Accuracy"):
                status_text.text(f"Epoch {epoch} | Loss: {loss:.4f} | {metric_name}: {metric:.2f}")
                fig = TrainingVisualizer.plot_loss_curve(trainer.history, metric_name)
                chart_placeholder.plotly_chart(fig, use_container_width=True)
                
            trainer.train(epochs=epochs, progress_callback=progress_callback)
            
            st.success("Training Complete!")
            st.session_state['trainer'] = trainer
            
            # Save for inference
            if not os.path.exists("./models"): os.makedirs("./models")
            trainer.save_model(f"./models/{model_name}_trained.pth")
            
    with col2:
        if st.session_state['trainer'] is not None:
            # Show history
            if task_type == "super_resolution":
                m_name = "PSNR"
            elif task_type == "segmentation":
                m_name = "IoU"
            elif task_type == "diffusion":
                m_name = "MSE Loss"
            else:
                m_name = "Accuracy"
                
            fig = TrainingVisualizer.plot_loss_curve(st.session_state['trainer'].history, m_name)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            st.subheader("3D Loss Landscape Analysis")
            if st.button("Compute Loss Landscape (High Compute)"):
                with st.spinner("Calculating 3D Surface via PCA..."):
                    X, Y, Z, tx, ty, tz = LandscapeUtils.compute_loss_surface(st.session_state['trainer'])
                    
                if X is not None:
                    fig_3d = TrainingVisualizer.plot_loss_landscape(X, Y, Z, tx, ty, tz)
                    st.plotly_chart(fig_3d, use_container_width=True, height=700)
                else:
                    st.error("Not enough snapshots to generate landscape.")

# ----------------------------------
# INFERENCE TAB
# ----------------------------------
elif mode == "Inference":
    st.title(f"Inference Lab: {model_name}")
    
    @st.cache_resource
    def load_inference_model(name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(name).to(device)
        path = f"./models/{name}_trained.pth"
        loaded = False
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            loaded = True
        
        model.eval()
        return model, device, loaded, path

    
    # Reload button to handle cache invalidation after training
    col_a, col_b = st.columns([3, 1])
    with col_b:
        if st.button("🔄 Reload Model Weights"):
            st.cache_resource.clear()
            st.rerun()

    model, device, is_loaded, ckpt_path = load_inference_model(model_name)
    
    # Checkpoint Status
    with col_a:
        if is_loaded:
            st.success(f"✅ **Model Ready**: Loaded `{ckpt_path}`")
        else:
            st.error(f"⚠️ **Untrained**: No file at `{ckpt_path}`") 
            st.info("train first or click reload if file exists")

    # Helper for hooks
    activations = {}
    def get_activation(name, clean_name):
        def hook(model, input, output):
            activations[clean_name] = output.detach()
        return hook

    # Register hooks based on architecture
    for name, layer in model.named_children():
        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.MaxPool2d, torch.nn.Sequential)):
            layer.register_forward_hook(get_activation(name, name))
            if isinstance(layer, torch.nn.Sequential):
                for sub_name, sub_layer in layer.named_children():
                     sub_layer.register_forward_hook(get_activation(f"{name}_{sub_name}", f"{name}_{sub_name}"))

    # UI Controls
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if task_type == "diffusion":
             btn_label = "Generate New Image"
        else:
             btn_label = "Generate Random Sample"
             
        if st.button(btn_label):
             from torchvision import datasets, transforms
             if task_type == "classification":
                 test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
                 idx = np.random.randint(0, len(test_dataset))
                 img, label = test_dataset[idx]
                 st.session_state['inf_sample'] = (img, label)
             elif task_type == "super_resolution":
                 transform = transforms.Compose([transforms.Grayscale(), transforms.ToTensor()])
                 test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
                 idx = np.random.randint(0, len(test_dataset))
                 img, _ = test_dataset[idx]
                 st.session_state['inf_sample'] = (img, "Super Resolution Input")
             elif task_type == "segmentation":
                 from src.core.trainer import ShapesDataset
                 ds = ShapesDataset(size=10, img_size=64)
                 img, mask = ds[0]
                 st.session_state['inf_sample'] = (img, mask)
             elif task_type == "diffusion":
                 # Generate from noise
                 T = 200 
                 beta = torch.linspace(0.0001, 0.02, T).to(device)
                 alpha = 1.0 - beta
                 alpha_bar = torch.cumprod(alpha, dim=0)
                 
                 x = torch.randn(1, 1, 28, 28).to(device)
                 
                 frames = []
                 progress_bar = st.progress(0)
                 
                 with torch.no_grad():
                     for t in reversed(range(T)):
                         t_tensor = torch.tensor([t], device=device).float().unsqueeze(1)
                         predicted_noise = model(x, t_tensor)
                         
                         if t > 0:
                             noise = torch.randn_like(x)
                         else:
                             noise = torch.zeros_like(x)
                         
                         alpha_t = alpha[t]
                         alpha_bar_t = alpha_bar[t]
                         
                         coeff = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)
                         
                         x = (1 / torch.sqrt(alpha_t)) * (x - coeff * predicted_noise) + torch.sqrt(beta[t]) * noise
                         
                         if t % 20 == 0 or t == 0:
                             view_img = (x.clamp(-1, 1) + 1) / 2
                             frames.append(view_img.cpu())
                         
                         progress_bar.progress((T - t) / T)
                 
                 st.session_state['inf_sample'] = (frames[-1], "Generated Digit")
                 st.session_state['diffusion_frames'] = frames

        if 'inf_sample' in st.session_state:
            img, label = st.session_state['inf_sample']
            
            if task_type == "diffusion":
                 st.success("Generation Complete")
                 if 'diffusion_frames' in st.session_state:
                     frames = st.session_state['diffusion_frames']
                     st.write("Denoising Process:")
                     
                     cols = st.columns(min(len(frames), 5))
                     step_indices = np.linspace(0, len(frames)-1, len(cols)).astype(int)
                     
                     for idx, i in enumerate(step_indices):
                         f = frames[i]
                         with cols[idx]:
                             st.image(f.squeeze().numpy(), caption=f"Step", width=100)
                             
                 st.image(img.squeeze().numpy(), caption="Final Generation", width=150)
                 
                 # Dummy forward for hook
                 inp_tensor = img.to(device) 
                 t_zero = torch.tensor([0], device=device).float().unsqueeze(1)
                 with torch.no_grad():
                     model(inp_tensor, t_zero)
                     
            elif task_type == "classification":
                inp_tensor = img.unsqueeze(0).to(device)
                st.image(img.squeeze().numpy(), caption=f"Label: {label}", width=150)
                
                normalizer = transforms.Normalize((0.1307,), (0.3081,))
                inp_to_model = normalizer(inp_tensor)
                
                with torch.no_grad():
                    out = model(inp_to_model)
                
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred = np.argmax(probs)
                st.metric("Prediction", f"{pred}", f"{probs[pred]:.1%}")
                st.caption("Class Probabilities")
                st.bar_chart(probs)

            elif task_type == "super_resolution":
                inp_tensor = img.unsqueeze(0).to(device)
                hr = inp_tensor
                lr = torch.nn.functional.interpolate(hr, scale_factor=0.5, mode='bicubic')
                lr_upscaled = torch.nn.functional.interpolate(lr, scale_factor=2.0, mode='bicubic')
                inp_to_model = lr_upscaled
                
                st.image(lr_upscaled.squeeze().cpu().numpy(), caption="Input (Low Res)", width=150, clamp=True)
                
                with torch.no_grad():
                    out = model(inp_to_model)
                
                st.image(out.squeeze().cpu().numpy(), caption="Output (SRCNN)", width=150, clamp=True)
                st.image(hr.squeeze().cpu().numpy(), caption="Ground Truth", width=150, clamp=True)

            elif task_type == "segmentation":
                st.image(img.squeeze().numpy(), caption="Input (Shapes)", width=150, clamp=True)
                
                inp_tensor = img.unsqueeze(0).to(device)
                with torch.no_grad():
                     out = model(inp_tensor)
                
                pred_mask = torch.argmax(out, dim=1).squeeze().cpu().numpy()
                gt_mask = label.numpy()
                
                st.image((pred_mask * 255).astype(np.uint8), caption="Predicted Mask", width=150, clamp=True)
                st.image((gt_mask * 255).astype(np.uint8), caption="Ground Truth Mask", width=150, clamp=True)

    with col2:
        if 'inf_sample' in st.session_state:
            # 3D Viz
            visualizer = NetworkVisualizer()
            th = st.sidebar.slider("Threshold", 0.0, 1.0, 0.2)
            visualizer.layer_spacing = st.sidebar.slider("Spacing", 50, 400, 150)
            
            with st.spinner("Visualizing..."):
                 fig = visualizer.create_network_figure(activations, threshold=th)
            st.plotly_chart(fig, use_container_width=True, height=800)
            
            # 2D Layer Inspector
            st.divider()
            with st.expander("🔍 Layer-wise Feature Map Inspector (2D)", expanded=True):
                st.write("Visualize how each layer transforms the image.")
                
                layer_names = list(activations.keys())
                selected_layer = st.selectbox("Select Layer to Inspect", layer_names)
                
                if selected_layer:
                    act = activations[selected_layer] # (1, C, H, W) or (1, D)
                    st.write(f"Activation Shape: `{tuple(act.shape)}`")
                    
                    if len(act.shape) == 4: # Conv2D (N, C, H, W)
                        act = act.squeeze(0) # (C, H, W)
                        channels = act.shape[0]
                        
                        cols = st.columns(min(channels, 8))
                        limit = min(channels, 32)
                        st.caption(f"Showing first {limit} channels:")
                        
                        for i in range(limit):
                            ch_img = act[i].cpu().numpy()
                            if ch_img.max() > ch_img.min():
                                ch_img = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min())
                            else:
                                ch_img = np.zeros_like(ch_img)
                                
                            with cols[i % 8]:
                                st.image(ch_img, caption=f"Ch {i}", clamp=True, width=80)
                    
                    elif len(act.shape) == 2: # Linear (N, D)
                         vec = act.squeeze(0).cpu().numpy()
                         st.bar_chart(vec)
                         st.caption("Neuron Activations")
