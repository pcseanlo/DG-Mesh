#  created by Isabella Liu (lal005@ucsd.edu) at 2024/05/29 17:10.
#
#  Rendering the DG-Mesh in a trajectory


import os, sys
import copy
import json
import datetime
import os.path as osp
import torch
import uuid
import datetime
from tqdm import tqdm
import random
from argparse import ArgumentParser, Namespace
import numpy as np
import imageio
import nvdiffrast.torch as dr

from scene import Scene
from scene import GaussianModelDPSRDynamicAnchor as gaussian_model
from scene import DeformModelNormal as deform_model
from scene import DeformModelNormalSep as deform_model_sep
from scene import AppearanceModel as appearance_model
from utils.renderer import mesh_renderer, mesh_shape_renderer, pointcloud_renderer
from utils.general_utils import safe_state
from utils.system_utils import load_config_from_file, merge_config
from utils.camera_utils import get_camera_trajectory_pose
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render


try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def rendering_trajectory(
    dataset,
    opt,
    pipe,
    checkpoint,
    camera_radius,
    camera_elevation,
    camera_lookat,
    total_frames,
    fps=24,
    custom_dxyz_path=None,
):
    args.model_path = dataset.model_path

    # Load models
    ## Gaussian model
    gaussians = gaussian_model(
        dataset.sh_degree,
        grid_res=dataset.grid_res,
        density_thres=opt.init_density_threshold,
        dpsr_sig=opt.dpsr_sig,
    )
    glctx = dr.RasterizeCudaContext()
    scene = Scene(dataset, gaussians, shuffle=False)
    ## Deform forward model
    deform = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform"
    )
    deform_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_normal",
    )
    ## Deform backward model
    deform_back = deform_model(
        is_blender=dataset.is_blender, is_6dof=dataset.is_6dof, model_name="deform_back"
    )
    deform_back_normal = deform_model_sep(
        is_blender=dataset.is_blender,
        is_6dof=dataset.is_6dof,
        model_name="deform_back_normal",
    )
    ## Appearance model
    appearance = appearance_model(is_blender=dataset.is_blender)
    ## Load checkpoint
    if checkpoint:
        gaussians.load_ply(checkpoint, iteration=-1)
        deform.load_weights(checkpoint, iteration=-1)
        deform_normal.load_weights(checkpoint, iteration=-1)
        deform_back.load_weights(checkpoint, iteration=-1)
        deform_back_normal.load_weights(checkpoint, iteration=-1)
        appearance.load_weights(checkpoint, iteration=-1)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Load custom d_xyz if provided
    custom_dxyz = None
    if custom_dxyz_path is not None:
        custom_dxyz = np.load(custom_dxyz_path)  # shape (T, N, 3)
        if custom_dxyz.shape[1] != gaussians.get_xyz.shape[0]:
            raise ValueError(f"Custom d_xyz shape {custom_dxyz.shape} does not match gaussians shape {gaussians.get_xyz.shape}")
        custom_dxyz = torch.tensor(custom_dxyz, dtype=torch.float32, device="cuda")
        print(f"Loaded custom d_xyz from {custom_dxyz_path}, with shape {custom_dxyz.shape}")
        # Override total_frames to match custom trajectory length
        total_frames = custom_dxyz.shape[0]
        print(f"Setting total_frames to {total_frames} based on custom d_xyz")

    # Compose camera trajectory
    camera_poses = get_camera_trajectory_pose(
        camera_radius, camera_elevation, total_frames, look_at=camera_lookat
    )
    viewpoint_cam = scene.getTestCameras()[
        0
    ]  # Use the intrinsics from the first camera in the test cameras

    # Create folders
    image_folder = osp.join(dataset.model_path, "images")
    os.makedirs(image_folder, exist_ok=True)
    final_images = []
    # Lists to collect per-frame deformation deltas and deformed positions
    dxyz_list = []
    drot_list = []
    dscale_list = []
    deformed_list = []


    # store the first frame rotation and scaling for reference
    fid = torch.tensor([0.0], device="cuda")
    N = gaussians.get_xyz.shape[0]
    time_input = fid.unsqueeze(0).expand(N, -1)

    _, d_rotation_0, d_scaling_0, _ = deform.step(
        gaussians.get_xyz.detach(), time_input
    )
    d_normal_0 = deform_normal.step(gaussians.get_xyz.detach(), time_input)

    for idx, pose in tqdm(enumerate(camera_poses)):
        render_cam = copy.deepcopy(viewpoint_cam)
        
        # Convert pose (c2w in Blender/OpenGL format) to camera parameters
        c2w = np.array(pose)
        # Change from OpenGL/Blender camera axes (Y up, Z back) to OpenCV (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # Get world-to-camera transform
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        # Update camera extrinsics for Gaussian rendering
        render_cam.reset_extrinsic(R, T)
        # Also set orig_transform for other renderers if needed
        render_cam.orig_transform = pose
        
        # Debug: print camera position to verify it's changing
        if idx % 20 == 0:
            cam_center = render_cam.camera_center.cpu().numpy()
            print(f"Frame {idx}: Camera center = {cam_center}")

        fid = torch.tensor([idx / total_frames], device="cuda")
        N = gaussians.get_xyz.shape[0]
        time_input = fid.unsqueeze(0).expand(N, -1)

        # Query the gaussians or use custom d_xyz
        if custom_dxyz is not None:
            # Use custom d_xyz
            d_xyz = custom_dxyz[idx] - gaussians.get_xyz.detach()
            d_rotation = d_rotation_0
            d_scaling = d_scaling_0
        else:
            # Use deformation model
            d_xyz, _, _, _ = deform.step(
                gaussians.get_xyz.detach(), time_input
            )
            d_rotation = d_rotation_0
            d_scaling = d_scaling_0
        d_normal = d_normal_0

        # Store deformation deltas as CPU numpy arrays for later saving
        try:
            dxyz_np = d_xyz.detach().cpu().numpy()
        except Exception:
            dxyz_np = np.array(d_xyz)
        if isinstance(d_rotation, torch.Tensor):
            drot_np = d_rotation.detach().cpu().numpy()
        else:
            drot_np = np.array(d_rotation)
        if isinstance(d_scaling, torch.Tensor):
            dscale_np = d_scaling.detach().cpu().numpy()
        else:
            dscale_np = np.array(d_scaling)
        dxyz_list.append(dxyz_np)
        drot_list.append(drot_np)
        dscale_list.append(dscale_np)
        # Compute and store deformed gaussian positions (original + delta)
        try:
            base_xyz = gaussians.get_xyz.detach()
        except Exception:
            base_xyz = gaussians.get_xyz
        try:
            deformed_np = (base_xyz + d_xyz).detach().cpu().numpy()
        except Exception:
            # Fallback if tensors aren't torch.Tensors
            deformed_np = np.array(base_xyz) + np.array(d_xyz)
        deformed_list.append(deformed_np)
        
        # Render Gaussian splatting image
        render_pkg = render(
            render_cam,
            gaussians,
            pipe,
            background,
            d_xyz,
            d_rotation,
            d_scaling,
            dataset.is_6dof,
        )
        gs_image = render_pkg["render"]
        gs_image_np = (gs_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        
        # Save Gaussian rendering image
        final_img = gs_image_np
        img_save_path = osp.join(image_folder, f"{idx:04d}.png")
        imageio.imwrite(img_save_path, final_img.astype(np.uint8))

        final_images.append(final_img)

    # Save the final video
    final_images = np.stack(final_images).astype(np.uint8)

    # Save deformation arrays collected per-frame (store deformed positions instead of raw d_xyz)
    if len(dxyz_list) > 0:
        try:
            # Stack arrays: shape will be (frames, N, 3) for positions
            deformed_arr = np.stack(deformed_list)
            drot_arr = np.stack(drot_list)
            dscale_arr = np.stack(dscale_list)
            np.save(osp.join(dataset.model_path, "deformed_xyz.npy"), deformed_arr)
            np.save(osp.join(dataset.model_path, "d_rotation.npy"), drot_arr)
            np.save(osp.join(dataset.model_path, "d_scaling.npy"), dscale_arr)
            print(f"Saved deformed positions and other deformation arrays to {dataset.model_path}")
        except Exception as e:
            print(f"Failed saving deformation arrays: {e}")

    # Save the gif
    with imageio.get_writer(
        osp.join(dataset.model_path, "video.gif"), fps=fps, codec="libx264", loop=0
    ) as writer:
        for img in final_images:
            writer.append_data(img)

    # Save the mp4
    with imageio.get_writer(
        osp.join(dataset.model_path, "video.mp4"), fps=fps, codec="libx264"
    ) as writer:
        for img in final_images:
            writer.append_data(img)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--camera_radius", type=float, default=4.0)
    parser.add_argument("--camera_lookat", type=float, nargs="+", default=[0, 0, 0])
    parser.add_argument("--camera_elevation", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--total_frames", type=int, default=240)
    parser.add_argument("--custom_dxyz_path", type=str, default=None, help="Path to custom d_xyz array (.npy file with shape [T, N, 3])")

    # Fix random seed
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    args = parser.parse_args(sys.argv[1:])

    # Load config file
    if args.config:
        config_data = load_config_from_file(args.config)
        combined_args = merge_config(config_data, args)
        args = Namespace(**combined_args)

    lp = lp.extract(args)
    op = op.extract(args)
    pp = pp.extract(args)

    # Updating save path
    unique_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    data_name = osp.basename(lp.source_path)
    folder_name = f"custom-rendering-traj-{data_name}-{unique_str}"
    if not lp.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        lp.model_path = os.path.join("./output/", unique_str[0:10])
    lp.model_path = osp.join(lp.model_path, folder_name)
    # Set up output folder
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok=True)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Save all parameters into file
    combined_args = vars(Namespace(**vars(lp), **vars(op), **vars(pp)))
    # Convert namespace to JSON string
    args_json = json.dumps(combined_args, indent=4)
    # Write JSON string to a text file
    with open(osp.join(lp.model_path, "cfg_args.txt"), "w") as output_file:
        output_file.write(args_json)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    rendering_trajectory(
        lp,
        op,
        pp,
        args.start_checkpoint,
        args.camera_radius,
        args.camera_elevation,
        args.camera_lookat,
        args.total_frames,
        args.fps,
        args.custom_dxyz_path,
    )

    # All done
    print("\nRendering complete.")
