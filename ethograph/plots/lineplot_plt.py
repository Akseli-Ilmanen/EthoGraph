from typing import Optional
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from ethograph.utils.data_utils import sel_valid






def plot_multidim(ax, time, data, coord_labels=None):
    """
    Plot multi-dimensional data (e.g., pos, vel) over time.
    data: shape (time, space)
    coord_labels: list of labels for each dimension (e.g., ['x', 'y', 'z'])
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    for i in range(data.shape[1]):
        label = coord_labels[i] if coord_labels is not None else f"dim {i}"
        ax.plot(time, data[:, i], label=label)

    return ax


def plot_singledim(ax, time, data, color_data=None, changepoints_dict=None):
    """
    Plot single-dimensional data (e.g., speed) over time.
    Optionally color the curve and mark changepoints.
    """

    if color_data is not None and color_data.shape[1] == 3:
        # color_data: shape (time, 3)
        for i in range(len(data) - 1):
            x_vals = [time[i], time[i+1]]
            y_vals = [data[i], data[i+1]]
            color = color_data[i] if i < len(color_data) else [0, 0, 0]
            ax.plot(x_vals, y_vals, color=color, linewidth=2)
    else:
        ax.plot(time, data, label="Speed")


    if changepoints_dict is not None:
        cmap = plt.get_cmap('tab10')
        colors = cmap.colors[5:]
        for i, (cp_name, cp_array) in enumerate(changepoints_dict.items()):
            idxs = np.where(cp_array)[0]
            color = colors[i % len(colors)]
            ax.scatter(
            time[idxs],
            data[idxs],
            marker='o',
            label=f"{cp_name}",
            facecolors='none',    
            edgecolors=color,  
            zorder=5,
            s=100             
            )


    return ax

def plot_ds_variable(ax, ds, ds_kwargs, variable, color_variable=None):
    """
    Plot a variable from ds for a given trial and keypoint.
    Handles both multi-dimensional (e.g., pos, vel) and single-dimensional (e.g., speed) variables.

    e.g. ds_kwargs: {trials=20, keypoints="beakTip", individuals="Freddy"}

    """
    # Use ds.sel for direct selection
    var = ds[variable]
    time = ds["time"].values

    
    data, _ = sel_valid(var, ds_kwargs)
    
    
    # (time, XX), e.g. (time, space)
    if data.ndim == 2:

        coord_labels = var.coords[var.dims[-1]].values # var.dims[-1] gets the last coord remaining, e.g. space
        ax = plot_multidim(ax, time, data, coord_labels=coord_labels) # coord_labels -> [x, y, z]

    # (time, )
    elif data.ndim == 1:

        color_data, _ = sel_valid(ds[color_variable], ds_kwargs) if color_variable in ds.data_vars else None

        
        changepoints_dict = {}
        cp_ds = ds.filter_by_attrs(type="changepoints")
        for cp_var_name in cp_ds.data_vars:
            cp_var = cp_ds[cp_var_name]
            cp_data = cp_var.sel(**ds_kwargs).squeeze().values
            if cp_var.attrs.get("target_feature") == variable and not np.isnan(cp_data).all():
                changepoints_dict[cp_var_name] = cp_data


        ax = plot_singledim(ax, time, data, color_data=color_data, changepoints_dict=changepoints_dict)

    else:
        print(f"Variable '{variable}' not supported for plotting.")


    if hasattr(ds, "boundary_events"):
        boundary_events_raw = ds["boundary_events"].values
        valid_events = boundary_events_raw[~np.isnan(boundary_events_raw)]
        eventsIdxs = valid_events.astype(int)
        eventsIdxs = eventsIdxs[(eventsIdxs >= 0) & (eventsIdxs < len(time))]
        
        for event in eventsIdxs:
            ax.axvline(x=time[event], color='k')


    ylabel = var.attrs.get("ylabel", variable)
    title = ", ".join([f"Trial: {ds.attrs['trial']}" ] + [f"{k}={v}" for k, v in ds_kwargs.items()])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right')

    return ax

    

def space_plot(
    ax: Optional[plt.Axes] = None,
    ds: xr.Dataset = None,
    color_variable: Optional[str] = None,
    view_3d: bool = False,
    **ds_kwargs
) -> plt.Axes:
    """Plot trajectory from top view (2D) or 3D view with color coding."""
    
    # Create appropriate axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        if view_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
    else:
        # Verify axes type matches view_3d
        if view_3d and not hasattr(ax, 'zaxis'):
            raise ValueError("For 3D plotting, ax must be created with projection='3d'")
    
    # Extract position data
    spaces = ['x', 'y', 'z'] if view_3d else ['x', 'y']
    position, _ = sel_valid(ds.sel(space=spaces).position, ds_kwargs)
    X, Y = position[:, 0], position[:, 1]
    Z = position[:, 2] if view_3d else None
    
    # Extract and normalize color data
    color_data = None
    if color_variable and color_variable in ds.data_vars:
        color_data, _ = sel_valid(ds[color_variable], ds_kwargs)
        if color_data.max() > 1.0:
            color_data = color_data / 255.0
    
    # Plot trajectory
    n_points = len(X)
    if view_3d:
        if color_data is None:
            ax.plot(X, Y, Z, color='blue', linewidth=3)
        else:
            for i in range(n_points - 1):
                ax.plot(X[i:i+2], Y[i:i+2], Z[i:i+2],
                       color=color_data[i], linewidth=3)
    else:
        if color_data is None:
            ax.plot(X, Y, color='blue', linewidth=3)
        else:
            for i in range(n_points - 1):
                ax.plot(X[i:i+2], Y[i:i+2],
                       color=color_data[i], linewidth=3)
    
    # Define box geometry
    box_xy_base = np.array([
        [-7.00,  0.00],
        [-7.00,  9.80],
        [ 6.80,  9.80],
        [ 6.80,  0.00]
    ])
    z_bot, z_top = 0.65, 2.75
    
    # Plot bounding box
    if view_3d:
        vertices = {
            'box1': [box_xy_base[0][0], box_xy_base[0][1], z_bot],
            'box2': [box_xy_base[1][0], box_xy_base[1][1], z_bot],
            'box3': [box_xy_base[2][0], box_xy_base[2][1], z_bot],
            'box4': [box_xy_base[3][0], box_xy_base[3][1], z_bot],
            'box5': [box_xy_base[0][0], box_xy_base[0][1], z_top],
            'box6': [box_xy_base[1][0], box_xy_base[1][1], z_top],
            'box7': [box_xy_base[3][0], box_xy_base[3][1], z_top],
            'box8': [box_xy_base[2][0], box_xy_base[2][1], z_top]
        }
        
        edges = [
            ('box1', 'box2'), ('box2', 'box3'), ('box3', 'box4'), ('box4', 'box1'),
            ('box1', 'box5'), ('box2', 'box6'), ('box3', 'box8'), ('box4', 'box7'),
            ('box5', 'box6'), ('box6', 'box8'), ('box8', 'box7'), ('box7', 'box5')
        ]
        
        for v1, v2 in edges:
            p1, p2 = vertices[v1], vertices[v2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 
                   'k-', linewidth=2)
        
        # Set 3D-specific properties
        ax.set_zlabel('Z')
        ax.set_box_aspect([1,1,0.5])  # Adjust aspect ratio for better view
    else:
        closed_box = np.vstack([box_xy_base, box_xy_base[0]])
        ax.plot(closed_box[:, 0], closed_box[:, 1], 'k-', linewidth=2)
    
    # Common settings
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal' if not view_3d else 'auto')
    ax.axis('off')
    
    return ax