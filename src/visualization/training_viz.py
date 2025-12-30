import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

class TrainingVisualizer:
    
    @staticmethod
    def plot_loss_curve(history, metric_name="Accuracy"):
        """
        Plots 2D line chart for Loss and Metric (Accuracy/PSNR).
        history: dict {'loss': [], 'metric': []}
        """
        steps = list(range(len(history['loss'])))
        df = pd.DataFrame({
            'Step': steps,
            'Loss': history['loss'],
            metric_name: history['metric']
        })
        
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=df['Step'], y=df['Loss'], name="Loss"),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(x=df['Step'], y=df[metric_name], name=metric_name, line=dict(dash='dot')),
            secondary_y=True,
        )
        
        fig.update_layout(
            title_text="Real-time Training Metrics",
            xaxis_title="Step (Batch)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.update_yaxes(title_text="Loss", secondary_y=False)
        fig.update_yaxes(title_text=metric_name, secondary_y=True)
        
        return fig

    @staticmethod
    def plot_loss_landscape(X, Y, Z, traj_x, traj_y, traj_z):
        """
        Plots 3D Loss Surface with Gradient Descent Trajectory.
        """
        fig = go.Figure()
        
        # 1. Surface
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8, name='Loss Surface'))
        
        # 2. Trajectory Line
        fig.add_trace(go.Scatter3d(
            x=traj_x, y=traj_y, z=traj_z,
            mode='lines+markers',
            marker=dict(size=4, color='red'),
            line=dict(color='yellow', width=5),
            name='Gradient Descent Path'
        ))
        
        # 3. Start and End points annotation
        fig.add_trace(go.Scatter3d(
            x=[traj_x[0]], y=[traj_y[0]], z=[traj_z[0]],
            mode='text', text=['Start'], textposition='top center',
            textfont=dict(color='white', size=14)
        ))
        fig.add_trace(go.Scatter3d(
            x=[traj_x[-1]], y=[traj_y[-1]], z=[traj_z[-1]],
            mode='text', text=['Converged'], textposition='top center',
            textfont=dict(color='white', size=14)
        ))

        fig.update_layout(
            title='3D Loss Landscape & Convergence Trajectory (PCA Projection)',
            scene=dict(
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                zaxis_title='Loss',
                bgcolor='#0e1117'
            ),
            paper_bgcolor='#0e1117',
            template="plotly_dark",
            margin=dict(l=0, r=0, b=0, t=40)
        )
        return fig
