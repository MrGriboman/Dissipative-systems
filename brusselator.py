import numpy as np
from scipy.integrate import odeint
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def brusselator(a, b, Da, Db, L=40, N=50, T=1000, dt=0.1):
    """Solve Brusselator PDE"""
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    
    u = np.ones(N) + 0.01 * np.random.rand(N)
    v = np.zeros(N) + 0.01 * np.random.rand(N)
    
    def rhs(u_v, t):
        u, v = u_v[:N], u_v[N:]
        laplacian_u = (np.roll(u,1) + np.roll(u,-1) - 2*u) / dx**2
        laplacian_v = (np.roll(v,1) + np.roll(v,-1) - 2*v) / dx**2
        du = a - (b+1)*u + u**2*v + Da*laplacian_u
        dv = b*u - u**2*v + Db*laplacian_v
        return np.concatenate([du, dv])
    
    t = np.linspace(0, T, int(T/dt))
    sol = odeint(rhs, np.concatenate([u, v]), t)
    u_history = sol[:, :N]
    
    return x, t, u_history

def create_2d_plot(x, u_final, L):
    """Create the 2D line plot of final concentration."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=u_final,
            mode='lines',
            line=dict(color='blue', width=2),
            name='u(x)'
        )
    )
    fig.update_layout(
        title="Activator Concentration at Final Time",
        xaxis_title=f"Position (r) [0, {L}]",
        yaxis_title="x(r)",
        height=500
    )
    return fig

def create_3d_plot(x, t, u_history, L):
    """Create optimized 3D surface plot with correct axis orientation."""
    # Downsample data for better performance
    time_step = max(1, len(t) // 100)
    space_step = max(1, len(x) // 100)
    
    t_downsampled = t[::time_step]
    x_downsampled = x[::space_step]
    u_downsampled = u_history[::time_step, ::space_step]
    
    T_grid, X_grid = np.meshgrid(t_downsampled, x_downsampled)
    
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X_grid,
            y=T_grid,
            z=u_downsampled.T,
            colorscale='Viridis',
            name='u(x,t)',
            showscale=True,
            contours=dict(
                x=dict(show=False),
                y=dict(show=False),
                z=dict(show=False)
            ),
            lighting=dict(
                ambient=0.7,
                diffuse=0.6,
                specular=0.2,
                roughness=0.3
            )
        )
    )
    fig.update_layout(
        title="Brusselator Dynamics",
        scene=dict(
            xaxis_title=f"Position (x) [0, {L}]",
            yaxis_title="Time (t)",
            zaxis_title="Concentration u(x,t)",
            aspectratio=dict(x=1, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
        ),
        height=700
    )
    return fig

if __name__ == "__main__":
    # Parameters
    a, b = 1, 2.1
    Da, Db = 1, 2
    L = 36
    N = 100
    
    x, t, u_history = brusselator(a, b, Da, Db, L=L, N=N)
    
    fig_2d = create_2d_plot(x, u_history[-1], L)
    fig_3d = create_3d_plot(x, t, u_history, L)
    
    # Show plots (will appear as separate tabs in browser)
    fig_2d.show()
    fig_3d.show()
