"""
Author: Donmgmyeong Lee (domlee[at]utexas.edu)
Date:   Feb 11, 2024
Description: Get the poses and pointclouds (result of odometry_saver)
"""
import os
import argparse
from pathlib import Path
from natsort import natsorted
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--odom_dir",
        type=str,
        required=True,
        help="Path to the map directory (result of odometry_saver)",
    )
    parser.add_argument(
        "--pcd",
        action="store_true",
        help="Get the .bin files of the pointclouds corresponding to the keyframes",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="",
        help="sequence name",
    )
    args = parser.parse_args()

    args.dataset_dir = Path("/home/dongmyeong/Projects/AMRL/CODa")
    args.pose_dir = args.dataset_dir / "poses" / "fast-lio"
    args.points_dir = args.dataset_dir / "3d_comp" / "os1" / args.seq
    args.prefix = "3d_comp_os1_" + args.seq
    args.timestamp_file = args.dataset_dir / "timestamps" / f"{args.seq}.txt"
    os.makedirs(args.pose_dir, exist_ok=True)
    os.makedirs(args.points_dir, exist_ok=True)

    return args


def load_pose(pose_file: str) -> np.ndarray:
    """
    Load estimated pose from a odom file (.odom) from the odometry_saver

    Args:
        pose_file: Path to the .odom file
    Returns:
        pose: (7,) estimated pose (x, y, z, qw, qx, qy, qz)
    """
    with open(pose_file, "r") as f:
        lines = f.readlines()

        pose_matrix = np.zeros((4, 4))
        for i, line in enumerate(lines):
            pose_matrix[i] = np.array([float(x) for x in line.split()])

        pose = np.zeros(7)
        pose[:3] = pose_matrix[:3, 3]
        pose[3:] = np.roll(R.from_matrix(pose_matrix[:3, :3]).as_quat(), 1)

    return pose


def pc_to_bin(pcd_file: str, bin_file: str):
    """
    Convert a .pcd file to a .bin file

    Args:
        pcd_file: Path to the .pcd file
        bin_file: Path to the .bin file
    """
    # load the pointcloud
    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)

    # save the pointcloud as a .bin file
    points.astype(np.float32).tofile(bin_file)
    print(f"Saved {bin_file}")


def main():
    args = get_args()

    # Get the pose and pcd files
    odometry_files = natsorted(list(Path(args.odom_dir).glob("[0-9]*.odom")))
    pcd_files = natsorted(list(Path(args.odom_dir).glob("[0-9]*.pcd")))
    assert len(odometry_files) == len(pcd_files)

    ref_timestamps = None
    if args.timestamp_file:
        ref_timestamps = np.loadtxt(args.timestamp_file, delimiter=" ", usecols=0)

    frame = 0
    poses = np.zeros(
        (len(ref_timestamps) if ref_timestamps is not None else len(odometry_files), 8)
    )
    for i, (odom_file, pcd_file) in enumerate(zip(odometry_files, pcd_files)):
        assert odom_file.stem == pcd_file.stem
        sec, nsec = odom_file.stem.split("_")
        ts = float(sec) + float(nsec) * 1e-9

        frame = i
        if args.timestamp_file:
            # ts of odom is last packet ts, so we deduct 1 to get the ts of the frame
            frame = np.searchsorted(ref_timestamps, ts, side="left") - 1
            print(f"frame: {frame}, ts: {ts}, ref_ts: {ref_timestamps[frame]}")

        # Load the pose (Sync with the timestamp file if it exists)
        pose = load_pose(odom_file)
        poses[frame, 0] = ref_timestamps[frame] if ref_timestamps is not None else ts
        poses[frame, 1:] = pose

        # use the first odom pose as the initial pose
        if i == 0 and frame > 0:
            for j in range(frame):
                poses[j, 0] = ref_timestamps[j]
                poses[j, 1:] = poses[frame, 1:]

        # Save the pointcloud
        if args.pcd:
            bin_file = args.points_dir / f"{args.prefix}_{frame}.bin"
            pc_to_bin(str(pcd_file), str(bin_file))


    # Fill the rest of the poses with the last pose
    if ref_timestamps is not None and frame < len(ref_timestamps):
        for j in range(frame, len(ref_timestamps)):
            poses[j, 0] = ref_timestamps[j]
            poses[j, 1:] = poses[frame, 1:]

    # Save the poses
    pose_file = args.pose_dir / f"{args.seq}.txt"
    with open(pose_file, "w") as f:
        for i in range(len(poses)):
            ts = poses[i, 0]
            pose = poses[i, 1:]
            f.write(f"{ts:.6f} " + " ".join(f"{x:.8f}" for x in pose) + "\n")


if __name__ == "__main__":
    main()
