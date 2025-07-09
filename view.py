import torch
from torch import nn
from typing import Callable, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton
import sys
import math

from nerf import *

device = torch.device('cuda')
d_input = 3
n_freqs = 10
log_space = True
n_layers = 2
d_filter = 128
skip = []
use_viewdirs = True
n_freqs_views = 4
focal = 138.88887889922103

def spherical_to_cartesian_zup(radius, theta, phi):
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.array([x, y, z], dtype=np.float32)

def spherical_camera_matrix_nerf_style(radius, theta, phi):
    # Camera position in Z-up coordinates
    pos = spherical_to_cartesian_zup(radius, theta, phi)

    # Original NeRF convention: Forward points from camera to world origin (0,0,0)
    forward = (np.array([0, 0, 0], dtype=np.float32) - pos)
    forward /= np.linalg.norm(forward)

    # Guess an "up" that's roughly Z+
    up_guess = np.array([0, 0, 1], dtype=np.float32)

    # Right vector: cross of forward and up_guess
    right = np.cross(forward, up_guess)
    right /= np.linalg.norm(right)

    # Recompute up to be orthogonal
    up = np.cross(right, forward)

    # Build matrix (row-major, like your original)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = right
    m[1, :3] = up
    m[2, :3] = -forward
    #m[:3, 3] = pos
    m[3, :3] = pos

    return m.T

# Example parameters (radius, theta, phi):
radius = 4.031129
theta = 0.7425274
phi = -2.3809996

view_matrix = spherical_camera_matrix_nerf_style(radius, theta, phi)
np.set_printoptions(precision=7, suppress=True)

testpose = torch.tensor(view_matrix).to(device)
#testpose = torch.tensor([[ 6.8935126e-01, 5.3373039e-01, -4.8982298e-01, -1.9745398e+00],
#            [-7.2442728e-01, 5.0788772e-01, -4.6610624e-01, -1.8789345e+00],
#            [ 1.4901163e-08, 6.7615211e-01,  7.3676193e-01,  2.9699826e+00],
#            [ 0.0000000e+00, 0.0000000e+00,  0.0000000e+00,  1.0000000e+00]]).to(device)

near, far = 2., 6.
n_samples = 8
perturb = True
inverse_depth = False
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}
n_samples_hierarchical = 64
chunksize = 2**14


encoder = PositionalEncoder(d_input, n_freqs, log_space=log_space)
encode = lambda x: encoder(x)

# View direction encoders
if use_viewdirs:
  encoder_viewdirs = PositionalEncoder(d_input, n_freqs_views,
                                      log_space=log_space)
  encode_viewdirs = lambda x: encoder_viewdirs(x)
  d_viewdirs = encoder_viewdirs.d_output
else:
  encode_viewdirs = None
  d_viewdirs = None

model = NeRF(encoder.d_output,
             n_layers=n_layers,
             d_filter=d_filter,
             skip=skip,
             d_viewdirs=d_viewdirs)
state_dict = torch.load('nerf.pt', map_location='cuda')
model.load_state_dict(state_dict)
model.to(device)

fine_model = NeRF(encoder.d_output,
             n_layers=n_layers,
             d_filter=d_filter,
             skip=skip,
             d_viewdirs=d_viewdirs)
fine_state_dict = torch.load('nerf-fine.pt', map_location='cuda')
fine_model.load_state_dict(fine_state_dict)
fine_model.to(device)

model.eval()
height, width = 100, 100

rays_o, rays_d = get_rays(height, width, focal, testpose)
rays_o = rays_o.reshape([-1, 3])
rays_d = rays_d.reshape([-1, 3])
outputs = nerf_forward(rays_o, rays_d,
                       near, far, encode, model,
                       kwargs_sample_stratified=kwargs_sample_stratified,
                       n_samples_hierarchical=n_samples_hierarchical,
                       kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                       fine_model=fine_model,
                       viewdirs_encoding_fn=encode_viewdirs,
                       chunksize=chunksize)

rgb_predicted = outputs['rgb_map']
image = rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()

fig, ax = plt.subplots()
im = plt.imshow(image)

stored_x = 0
stored_y = 0
t = 0.0

def on_move(event):
    if not event.inaxes:
        return

    global stored_x, stored_y, t
    global phi
    dx = float(event.x - stored_x) / 100.0
    stored_x = event.x
    phi -= dx

    view_matrix = spherical_camera_matrix_nerf_style(radius, theta, phi)
    testpose = torch.tensor(view_matrix).to(device)
    rays_o, rays_d = get_rays(height, width, focal, testpose)
    rays_o = rays_o.reshape([-1, 3])
    rays_d = rays_d.reshape([-1, 3])
    outputs = nerf_forward(rays_o, rays_d,
                           near, far, encode, model,
                           kwargs_sample_stratified=kwargs_sample_stratified,
                           n_samples_hierarchical=n_samples_hierarchical,
                           kwargs_sample_hierarchical=kwargs_sample_hierarchical,
                           fine_model=fine_model,
                           viewdirs_encoding_fn=encode_viewdirs,
                           chunksize=chunksize)

    rgb_predicted = outputs['rgb_map']
    image = rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy()

    im.set_data(image)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_move)

plt.show()

