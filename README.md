# Deep Learning Network Visualizer 🧠✨

**An Interactive Educational Platform for Computer Vision & Neural Networks.**

![Platform Banner](assets/banner.png)

This repository contains a powerful, interactive visualizer built with **PyTorch** and **Streamlit**. It is designed to help students, researchers, and enthusiasts understand the inner workings of Convolutional Neural Networks (CNNs), Diffusion Models, and training dynamics.

## 🌟 Key Features

### 1. Multi-Task Support
Explore different domains of Computer Vision in one platform:
*   **🖼️ Image Classification**: Train and visualize `SimpleCNN`, `AlexNet`, `ResNet18` on MNIST.
*   **🔍 Super Resolution**: Restore low-resolution images using `SRCNN`.
*   **🧩 Semantic Segmentation**: Segment geometric shapes using `SimpleUNet`.
*   **🎨 Image Generation (New!)**: Watch a **Diffusion Model** (`SimpleDiffUNet`) create digits from pure noise step-by-step!

### 2. Advanced Visualization Tools
*   **3D Network Explorer**: View your neural network as a 3D layered structure.
*   **2D Feature Map Inspector**: Inspect exactly what each convolutional filter "sees" after every layer. Validates how networks extract edges, textures, and patterns.
*   **Training Dynamics**: Real-time plots of Loss, Accuracy, PSNR, and IoU.
*   **3D Loss Landscape**: Visualize the "terrain" your model traverses during training using PCA-based dimensionality reduction. See local minima and optimization paths!

### 3. Interactive Web UI
Built with Streamlit for a seamless, code-free experience.
*   **Training Studio**: Train models from scratch on your CPU/GPU.
*   **Inference Lab**: Test your trained models on random samples or custom inputs.
*   **Interactive Controls**: Adjust thresholds, layer spacing, and hyperparameters on the fly.

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   Pip or Conda

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/network-visualizer.git
    cd network-visualizer
    ```

2.  **Create an Environment (Recommended):**
    ```bash
    conda create -n network-visualizer python=3.10
    conda activate network-visualizer
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App
launch the interactive dashboard with a single command:
```bash
streamlit run src/ui/app.py
```
Open your browser at `http://localhost:8501`.

## 📚 Modules Overview

| Module | Description |
| :--- | :--- |
| `src.core.models` | Contains PyTorch definitions for `SimpleCNN`, `ResNet`, `SRCNN`, `UNet`, `DiffUNet`. |
| `src.core.trainer` | `PlatformTrainer` class handling training loops, snapshots, and metrics. |
| `src.core.landscape` | `LandscapeUtils` for computing 3D loss surfaces via PCA. |
| `src.visualization` | Plotly-based visualizers for 3D networks and training curves. |
| `src.ui.app` | The main Streamlit application entry point. |

## 🧪 Verification
To run the automated test suite and ensure everything is working correctly:
```bash
python tests/verify_pipeline.py
```

## 🤝 Contributing
Contributions are welcome! Please open an issue or PR to add new models, tasks (e.g., GANs, NLP), or visualization features.

---
*Built with ❤️ for the AI Community.*
