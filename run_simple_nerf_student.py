'''
READ THIS

you should run this python code from Command Prompt.

To install pytorch-cpu: In anaconda run the following :conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 cpuonly -c pytorch

if you have gpu you can run also your code in GPU:
you can install pytorch-gpu (based on cuda version)
For example 

# CUDA 11.6
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

For more information: 
https://pytorch.org/get-started/previous-versions/


There are a bunch of video on Youtube that show how to install pytorch.

'''

import os

import imageio
import torch
from einops import rearrange
from tqdm import tqdm, trange

from load_blender import load_blender_data
from run_nerf_helpers import *


def create_pointcloud(N, min_val, max_val):
    '''
    N - number of points along the axis
    min_vaL and max_val: points are sampled between these two points
    '''

    t = np.linspace(min_val, max_val, N)
    t_index = np.arange(N)
    query_pts = np.stack(np.meshgrid(t, t, t), -1).astype(np.float32)
    query_indices = np.stack(np.meshgrid(t_index, t_index, t_index), -1).astype(np.int16)

    flat = query_pts.reshape([-1, 3])
    flat_indices = query_indices.reshape([-1, 3])

    return torch.tensor(flat), t, flat_indices


def raw2outputs(raw, z_vals, rays_d):
    rays_d_magnitude = torch.norm(rays_d, dim=1).unsqueeze(1)

    # delta = t_n+1 - t_n
    z_vals = torch.diff(z_vals)
    z_vals = torch.concatenate((z_vals, z_vals[-1].unsqueeze(0)), dim=0)
    z_vals = z_vals.repeat(rays_d.shape[0], 1)

    delta_n = z_vals * rays_d_magnitude

    # first three columns in tensor raw are dedicated to RGB channels
    # should be normalized
    c_n = raw[..., :3].sigmoid()

    # the fourth column in tensor raw is dedicated to density
    # sigma values should be only positive, the relu function sets to 0 all negative values (ReLU(x) = max(0,x))
    sigma_n = raw[..., 3].relu()

    # calculation of T_n that is part of the objective function: T_n = e^^(- SUM (sigma_m * delta_m))
    T_n = torch.exp(- torch.cumsum(sigma_n * delta_n, dim=-1))

    # calculation of weights with the given equation: weights = Tn * (1 - e^^(- sigma_n * delta_n))
    weights = (T_n * (1.0 - torch.exp(-sigma_n * delta_n))).unsqueeze(2)

    # calculation of the rgb_map using the aforementioned objective function C_r = SUM (weights) * c_n
    rgb_map = torch.sum(weights * c_n, dim=1)

    acc_map = torch.sum(weights.squeeze(2), -1)

    if True:  # this MUST BE ALWAYS TRUE in your case
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map


def find_nn(pts, ptscloud):
    """
    pts: points along the ray of size (M,3)
    ptscloud: points in the pointcloud (KX3), where K=200X200X200
    :returns nn_index: the nearest index for every point in pts
    """
    scaling_factor = 7.45 / 199.0
    global_shift = torch.tensor([-3.75, -3.75, -3.75])

    # apply shifting and scaling
    rounded_points = torch.round((pts + global_shift) / scaling_factor)

    # apply the relationship equation
    nn_index = rounded_points[..., 0] * 200 + rounded_points[..., 1] * 200 * 200 + rounded_points[..., 2]

    # print("nn_index.long().shape", nn_index.long().shape)
    return nn_index.long()


def render_rays_discrete(ray_steps, rays_o, rays_d, N_samples, pt_cloud, rgb_val, sigma_val):
    # reshape the rays distance, rasys origin and rays steps
    rays_d_reshape = rays_d.unsqueeze(1).repeat(1, N_samples, 1)
    rays_o_reshape = rays_o.unsqueeze(1).repeat(1, N_samples, 1)
    ray_steps_reshape = ray_steps.repeat(3, 1).transpose(0, 1)

    # create new points using the given rays
    points = rays_o_reshape + ray_steps_reshape * rays_d_reshape

    # find the nearest neighbors for those points
    nearest_neighbors = find_nn(points, pt_cloud)
    # initialize c_n with the rgb values that we extract from nearest_neighbors
    c_n = rgb_val[nearest_neighbors]
    # initialize sigma with the rgb values that we extract from nearest_neighbors
    sigma_n = sigma_val[nearest_neighbors]

    raw = torch.cat([c_n, sigma_n], dim=-1)

    # convert rays to raw output
    rgb_val_rays = raw2outputs(raw, ray_steps, rays_d)

    return rgb_val_rays


def regularize_rgb_sigma(point_cloud, rgb_values, sigma_values):
    # initialize number of points in the subset
    num_points_subset = 10000
    K, _ = point_cloud.shape

    # select indices randomly for the subset
    random_indices = torch.randperm(K)[:num_points_subset]

    # select indices randomly for the neighbors in the subset
    rand_neighbor_indices = torch.randint(-1, 2, (num_points_subset, 3), dtype=torch.long)
    random_neighbor_indices = random_indices.unsqueeze(-1) + rand_neighbor_indices

    # calculate 1D neighbor index using the relationship equation between indices
    neighbor_indices = (random_neighbor_indices[..., 0] * 200 + random_neighbor_indices[..., 1] * (200 ** 2) +
                        random_neighbor_indices[..., 2]).long()

    # calculate smoothness regularization for RGB values
    # l2_rgb = |c[i, j, k] − c[m, n, p]|2
    diff_rgb = rgb_values[random_indices] - rgb_values[neighbor_indices % K]
    diff_sigma = sigma_values[random_indices] - sigma_values[neighbor_indices % K]

    # calculate smoothness regularization for sigma values
    # l2_sigma = |σ[i, j, k] − σ[m, n, p]|2
    l2_rgb = torch.mean(diff_rgb.pow(2))
    l2_sigma = torch.mean(diff_sigma.pow(2))

    return l2_rgb, l2_sigma


def train():
    K = None
    device = 'cpu'

    '''
    Do not change below parameters !!!!!!!!
    '''
    N_rand = 1024  # number of rays that are use during the training, IF YOU DO NOT HAVE ENOUGH RAM YOU CAN DECREASE IT BUT DO NOT NOT FORGET TO INCREASE THE N_iter!!!!
    precrop_frac = 0.9  # do not change
    # start, N_iters = 0, 100001
    start, N_iters = 0, 20001
    N_samples = 200  # number of samples along the ray
    precrop_iters = 0
    lrate = 5e-3  # learning rate
    pts_res = 200  # point resolution of the pooint clooud
    pts_max = 3.725  # boundary of our point cloud
    near = 2.
    far = 6.

    # You can play with this hyperparameters
    lambda_sigma = 1e-3  # regularization for the lambda sigma (see the final loss in the loop)
    lambda_rgb = 1e-3  # regularization for the lambda color (see the final loss in the loop)

    main_folder_name = 'Train_lego'  # folder name where the output images, out variables will be estimated
    # load dataset
    images, poses, render_poses, hwf, i_split = load_blender_data('data/nerf_synthetic/lego', True, 8)
    print('Loaded blender', images.shape, render_poses.shape, hwf)
    i_train, _, _ = i_split
    print('\n i_train: ', i_train)
    # get white background
    images = images[..., :3] * images[..., -1:] + (1. - images[..., -1:])

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5 * W],
            [0, focal, 0.5 * H],
            [0, 0, 1]
        ])

    # generate point cloud
    pt_cloud, t_linspace, pt_cloud_indices = create_pointcloud(N=pts_res, min_val=-1 * pts_max, max_val=pts_max)
    pt_cloud = pt_cloud.to(device)
    save_folder_test = os.path.join('logs', main_folder_name)
    os.makedirs(save_folder_test, exist_ok=True)
    torch.save(pt_cloud.cpu(), os.path.join(save_folder_test, 'pts_clous.tns'))
    torch.save(torch.tensor(t_linspace), os.path.join(save_folder_test, 't_linspace.tns'))
    torch.save(torch.tensor(pt_cloud_indices).long(), os.path.join(save_folder_test, 'pt_cloud_indices.tns'))

    sigma_val = torch.ones(pt_cloud.size(0), 1).uniform_(0, 0.5).to(device)
    rgb_val = torch.zeros(pt_cloud.size(0), 3).uniform_().to(device)

    # do not make any change
    sigma_val.requires_grad = True
    rgb_val.requires_grad = True

    optimizer = torch.optim.Adam([{'params': sigma_val},
                                  {'params': rgb_val}],
                                 lr=lrate,
                                 betas=(0.9, 0.999))

    for i in trange(start, N_iters):
        # Random from one image
        img_i = np.random.choice(i_train)
        target = images[img_i]
        target = torch.Tensor(target).to(device)
        pose = poses[img_i, :3, :4]

        if N_rand is not None:
            # for every pixel in the image, get the ray origin and ray direction
            rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

            if i < precrop_iters:
                '''
                if this is True, at the  beginning it will sample rays only from the 'center' of the image to avoid bad local minima
                '''
                dH = int(H // 2 * precrop_frac)
                dW = int(W // 2 * precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
                    ), -1)
                if i == start:
                    print(f"[Config] Center cropping of size {2 * dH} x {2 * dW} is enabled until iter {precrop_iters}")
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                     -1)  # (H, W, 2)

            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
            select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            # select final ray_o and ray_d
            rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

            t_vals = torch.linspace(0., 1., steps=N_samples)
            z_vals = near * (1. - t_vals) + far * (t_vals)

            rgb_map = render_rays_discrete(ray_steps=z_vals,
                                           rays_o=rays_o,
                                           rays_d=rays_d,
                                           N_samples=N_samples,
                                           pt_cloud=pt_cloud,
                                           rgb_val=rgb_val,
                                           sigma_val=sigma_val)  # CHANGE THIS
            # Note that the rgb_map MUST have the same shape as the target_s !!!!

            # do not make any change          
            optimizer.zero_grad()
            img_loss = img2mse(rgb_map, target_s)
            reg_loss_rgb, reg_loss_sigma = regularize_rgb_sigma(point_cloud=pt_cloud,
                                                                rgb_values=rgb_val,
                                                                sigma_values=sigma_val)  # DO NOT FORGET TO CHANGE THIS
            loss = img_loss + lambda_rgb * reg_loss_rgb + lambda_sigma * reg_loss_sigma  # --> this is the loss we minimize
            psnr = mse2psnr(img_loss)

            loss.backward()
            optimizer.step()

        if i % 100 == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()} loss image: {img_loss.item()}")

        if i % 1000 == 0:  #
            '''
            YOU DO NOT NEED TO MAKE ANY CHANGE HERE EXCEPT  render_rays_discrete FUNCTION !!!!!
            at 1000-th iteration the bulldozer should be appeared when trained with the default hyperparameters
            We save some intermediate images.
            The first 100 images are from the training set and the rest are novel views!. To speed up the generation we render every 8th pose/image
            '''
            save_folder_test_img = os.path.join('logs', main_folder_name, f"{i:05d}")
            os.makedirs(save_folder_test_img, exist_ok=True)
            torch.save(rgb_val.detach().cpu(), os.path.join(save_folder_test_img, 'rgb_{:03d}.tns'.format(i)))
            torch.save(sigma_val.detach().cpu(), os.path.join(save_folder_test_img, 'sigma_{:03d}.tns'.format(i)))

            for j in trange(0, poses.shape[0], 8):

                pose = poses[j, :3, :4]
                rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

                coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)),
                                     -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

                chunk = 200
                # N_rand = chunk
                rgb_image = []
                for k in range(int(coords.size(0) / chunk)):
                    select_coords = coords[k * chunk: (k + 1) * chunk].long()  # (N_rand, 2)
                    rays_o_batch = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                    rays_d_batch = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

                    t_vals = torch.linspace(0., 1., steps=N_samples)
                    z_vals = near * (1. - t_vals) + far * (t_vals)

                    with torch.no_grad():
                        rgb_map = render_rays_discrete(ray_steps=z_vals,
                                                       rays_o=rays_o_batch,
                                                       rays_d=rays_d_batch,
                                                       N_samples=N_samples,
                                                       pt_cloud=pt_cloud,
                                                       rgb_val=rgb_val,
                                                       sigma_val=sigma_val)

                    rgb_image.append(rgb_map)

                rgb_image = torch.cat(rgb_image)

                rgb_image = rearrange(rgb_image, '(w h) d -> w h d', w=W)

                rgbimage8 = to8b(rgb_image.cpu().numpy())
                filename = os.path.join(save_folder_test_img, '{:03d}.png'.format(j))
                imageio.imwrite(filename, rgbimage8)


if __name__ == '__main__':
    # torch.set_default_tensor_type('torch.cuda.FloatTensor') # UNCOMMENT THIS IF YOU NEED TO RUN IT IN GPU

    train()
