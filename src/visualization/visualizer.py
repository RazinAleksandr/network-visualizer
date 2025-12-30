import plotly.graph_objects as go
import numpy as np
import torch

class NetworkVisualizer:
    def __init__(self):
        # Configuration for spacing
        self.layer_spacing = 150 # Distance between layers along the main axis
        
    def create_network_figure(self, activations_dict, threshold=0.0):
        """
        Creates a 3D Plotly figure representing the network state.
        
        Args:
            activations_dict: Dict of {layer_name: tensor} from ModelWrapper.
            threshold: Float, minimum activation value to show (0.0 to 1.0 usually).
        """
        fig = go.Figure()
        
        current_x_offset = 0
        layer_names = list(activations_dict.keys())
        
        for i, name in enumerate(layer_names):
            tensor = activations_dict[name] 
            
            # Handle different shapes
            # Conv Layers: (Batch, C, H, W) -> Squeeze batch -> (C, H, W)
            # FC Layers: (Batch, N) -> Squeeze batch -> (N)
            
            data = tensor.cpu().detach().numpy()
            if data.ndim == 4: # (B, C, H, W)
                data = data[0]
            elif data.ndim == 2: # (B, N)
                data = data[0]
                
            # Normalize data for visualization color (0-1)
            layer_max = np.abs(data).max()
            if layer_max > 0:
                data_norm = data / layer_max
            else:
                data_norm = data
            
            # ---------------------------
            # VISUALIZATION LOGIC
            # ---------------------------
            
            if data.ndim == 3: # CONV LAYER (C, H, W)
                C, H, W = data.shape
                
                # Filter by threshold
                mask = data_norm > threshold
                
                if not mask.any():
                    current_x_offset += self.layer_spacing + (C * 0.2)
                    continue
                
                c_idxs, h_idxs, w_idxs = np.where(mask)
                values = data_norm[mask]
                
                # Spatial Mapping
                # X = Flow (Global Offset + Channel Depth)
                # Y = Height
                # Z = Width
                
                layer_depth_scale = 1.0 
                x_coords = current_x_offset + (c_idxs * layer_depth_scale)
                y_coords = -h_idxs + (H / 2) # Flip Y for image comparison
                z_coords = w_idxs - (W / 2)
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=values,
                        colorscale='Plasma',
                        showscale=False
                    ),
                    name=name,
                    hoverinfo='text',
                    text=[f"{name}<br>C:{c} H:{h} W:{w}<br>Val:{v:.2f}" for c,h,w,v in zip(c_idxs, h_idxs, w_idxs, values)]
                ))
                
                current_x_offset += (C * layer_depth_scale) + self.layer_spacing
                
            elif data.ndim == 1: # FULLY CONNECTED (N,)
                N = data.shape[0]
                
                # For FC, we can just show all of them, or threshold
                # If N is large (128), maybe show 10x13 grid? Or just a line?
                # Let's do a vertical plane grid to keep it compact.
                
                cols = int(np.ceil(np.sqrt(N)))
                rows = int(np.ceil(N / cols))
                
                # Create grid coordinates
                grid_indices = np.arange(N)
                row_idx = grid_indices // cols
                col_idx = grid_indices % cols
                
                # Layout in Y-Z plane
                # X is fixed at current_offset
                
                x_coords = np.full(N, current_x_offset)
                y_coords = -row_idx + (rows / 2)
                z_coords = col_idx - (cols / 2)
                
                # Adjust spacing for FC visualization
                # If it's the Output layer (size 10), make them bigger/distinct
                node_size = 5
                if N == 10:
                    node_size = 10
                    # For output, maybe just a single vertical line?
                    # Let's stick to grid for consistency, for 10 it's 2x5 or 4x3.
                    # Or just a line of 10?
                    rows = 10
                    cols = 1
                    y_coords = np.arange(10) - 5
                    z_coords = np.zeros(10) 
                
                # Color mapping
                # We show ALL neurons for FC, but color them by intensity
                # We can also threshold if N is huge, but 128 is fine to show all.
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='markers+text' if N == 10 else 'markers',
                    marker=dict(
                        size=node_size,
                        color=data_norm,
                        colorscale='Viridis',
                        cmin=0, cmax=1,
                        showscale=False,
                        symbol='circle'
                    ),
                    text=[str(i) for i in range(10)] if N == 10 else None,
                    textposition="middle right",
                    name=name,
                    hoverinfo='text',
                    hovertext=[f"{name}<br>Idx:{i}<br>Val:{v:.2f}" for i,v in enumerate(data)]
                ))
                
                current_x_offset += self.layer_spacing

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='', showgrid=False, range=[0, current_x_offset+100], zeroline=False, showticklabels=False),
                yaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                zaxis=dict(title='', showgrid=False, zeroline=False, showticklabels=False),
                bgcolor='#0e1117'
            ),
            paper_bgcolor='#0e1117',
            margin=dict(l=0, r=0, b=0, t=0),
            showlegend=True,
            legend=dict(x=0, y=1, font=dict(color='white'))
        )
        
        return fig
