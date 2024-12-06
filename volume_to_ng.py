import os
import re
import sys
import glob
import json
import argparse
import pickle as pkl
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

import zarr
from cloudvolume import CloudVolume
from igneous import task_creation as tc
from taskqueue import LocalTaskQueue
import neuroglancer

dtypes = {
    "image": ("uint8", "float32"),
    "segmentation": ("uint8", "uint16", "uint32", "uint64")
}


def create_parser():
    parser = argparse.ArgumentParser(
        description="Process electron microscopy data")

    # Data selection arguments
    type_group = parser.add_mutually_exclusive_group(required=True)
    type_group.add_argument("--image", action="store_const", const="image", dest="command",
                            help="Process electron microscopy image data.")
    type_group.add_argument("--segmentation", action="store_const", const="segmentation", dest="command",
                            help="Process electron microscopy segmentation data.")

    # Input data directory
    parser.add_argument("INPUT_DIR", type=str,
                        help="Path to the electron microscopy data (em or segmentation).")

    # Task arguments
    task_group = parser.add_argument_group("Tasks", "Select the desired task")
    task_group.add_argument("--precompute", action="store_true",
                            help="Convert data to Neuroglancer precomputed format.")
    task_group.add_argument("--downsample", action="store_true",
                            help="Downsample data using igneous.")

    # Segmentation specific tasks
    segmentation_group = parser.add_argument_group(
        "Segmentation Tasks", "Tasks specific to segmentation data")
    segmentation_group.add_argument("--skeletons", action="store_true",
                                    help="Generate skeletons from segmentation data.")
    segmentation_group.add_argument("--mesh", action="store_true",
                                    help="Generate meshes from segmentation data.")

    # Neuroglancer visualization
    parser.add_argument("--show", action="store_true",
                        help="Show data in SyConn-Neuroglancer.")

    # Verbose flag
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Display details about the tasks.")

    return parser


def parse_arguments(parser):
    """
    Parses command-line arguments based on the provided parser.

    Args:
        parser (argparse.ArgumentParser): The argument parser.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    args = parser.parse_args()

    # Conditional arguments based on tasks
    if args.precompute:
        parser.add_argument("OUT_DIR", type=str,
                            help="Path to write the precomputed data (em or segmentation).")

    if args.downsample:
        downsample_group = parser.add_argument_group("Downsampling parameters")
        downsample_group.add_argument("--mip", default=0, type=int,
                                        help="Start downsampling from this mip level.")
        downsample_group.add_argument("--num_mips", default=1, type=int,
                                        help="Number of downsamples to produce.")
        downsample_group.add_argument("--volumetric", action="store_true",
                                        help="Do volumetric downsampling.")

    if args.mesh:
        mesh_group = parser.add_argument_group("Meshing parameters")
        mesh_group.add_argument("--res", default=0, type=int,
                                help="Resolution or LOD to mesh at.")
        mesh_group.add_argument("--err", default=40, type=int,
                                help="Maximum physical deviation of mesh vertices during simplification.")
        mesh_group.add_argument("--dust", default=512, type=int,
                                help="Minimum voxel count to mesh.")

    if args.show:
        show_group = parser.add_argument_group("Neuroglancer server parameters")
        show_group.add_argument("--port", default=1337, type=int,
                                    help="Port to run the Neuroglancer server.")

    # Reparse to include dynamic arguments
    return parser.parse_args()


def handle_tasks(args):
    tasks = {
        "Convert zarr": args.precompute,
        "Downsample": args.downsample,
        "Write skeletons": args.skeletons,
        "Generate meshes": args.mesh,
        "Show in Neuroglancer": args.show,
    }

    print(f"Input volume path: {args.INPUT_DIR}")
    print(f"Volume type: {args.command}")
    print("\nSelected tasks:")
    selected_tasks = [task for task, selected in tasks.items() if selected]
    for task in selected_tasks:
        print(f"    - {task}")

    cloud_path = args.INPUT_DIR

    for task in selected_tasks:
        if task == "Convert zarr":
            out_path = args.OUT_DIR
            cloud_path = precompute(args.INPUT_DIR, out_path, args.command)
        elif task == "Downsample":
            downsample(cloud_path, args.command)
        elif task == "Write skeletons":
            skeletonize(cloud_path)
        elif task == "Generate meshes":
            meshing(cloud_path)
        elif task == "Show in Neuroglancer":
            link = show(cloud_path, port=args.port)
            print(f"Link: {link}")
            input("Press any key to exit")
        print()


def process_chunk(zarr_vol, precomputed_vol, vol_type, chunk_size, chunk_offsets):
    for offset in chunk_offsets:
        try:
            x, y, z = offset

            x_start = x * chunk_size[0]
            y_start = y * chunk_size[1]
            z_start = z * chunk_size[2]

            # ignore the channel dimension (if exists)
            zarr_size = zarr_vol.shape[-3:]
            # check for boundary chunk ends for dataset
            x_end = (x_start + chunk_size[0]) if (x_start +
                                                  chunk_size[0]) < zarr_size[0] else zarr_size[0]
            y_end = (y_start + chunk_size[1]) if (y_start +
                                                  chunk_size[1]) < zarr_size[1] else zarr_size[1]
            z_end = (z_start + chunk_size[2]) if (z_start +
                                                  chunk_size[2]) < zarr_size[2] else zarr_size[2]

            if len(zarr_vol.shape) > 3:
                cube = zarr_vol[0, x_start:x_end, y_start:y_end, z_start:z_end]
            else:
                cube = zarr_vol[x_start:x_end, y_start:y_end, z_start:z_end]

            # compressed segmentation encoding only works with 'uint32' and 'uint64'
            if vol_type == 'segmentation':
                cube = cube.astype(np.uint64)

            precomputed_vol[x_start:x_end, y_start:y_end, z_start:z_end] = cube
            assert np.any(
                precomputed_vol[x_start:x_end, y_start:y_end, z_start:z_end]) == True

        except IOError as err:
            errno, strerror = err.args
            print('I/O error({0}): {1}'.format(errno, strerror))
            print(err)
        except:
            print('Unexpected error:', sys.exc_info())
            raise


def precompute(in_path, out_path, vol_type, resolution=[9, 9, 20], chunk_size=[512, 512, 512], batch_size=10):
    data = zarr.open(in_path)
    dataset_name = os.path.basename(os.path.normpath(in_path))
    dataset_name = dataset_name.split('.')[0]  # remove extensions
    out_path = os.path.join(out_path, dataset_name)
    match = re.search(r"v(\d+)_seed(\d+)", in_path)
    if match:
        version = match.group(1)
        seed = match.group(2)
    else:
        raise NameError(f"Could not parse version and seed from dataset path {in_path}.\
                        Please try again using a dataset with 'v<version>_seed<seed>' in the name.\
                        This is required by Nginx to serve statically (with a consistent naming scheme).")

    if isinstance(data, zarr.hierarchy.Group):
        if 'arr_0' in data.array_keys():
            arr = data['arr_0']
        size = arr.shape[-3:]  # ignore the channel dimension (if exists)

        info = {
            "type": vol_type,
            "layer_type": vol_type,
            "num_channels": 1,  # for both image and segmentation
            "data_type": "uint64" if vol_type == 'segmentation' else arr.dtype,
            "scales": [
                {
                    "key": "1_1_1",
                    "size": [size[0], size[1], size[2]],
                    "resolution": resolution,  # TODO: get this from upstream
                    # TODO: get this from upstream (or use the zarr chunk size)
                    "chunk_sizes": [chunk_size],
                    "encoding": "compressed_segmentation" if vol_type == "segmentation" else "raw",
                    "factor": (1, 1, 1),
                }
            ]
        }

        # add block size if 'compressed segmentation' encoding is used, typically [8, 8, 8]
        if vol_type == "segmentation":
            info["scales"][0]["compressed_segmentation_block_size"] = [8, 8, 8]

        precomputed_vol = CloudVolume(
            "file://" + out_path + "/",
            info=info,
            bounded=False,
            non_aligned_writes=True,
            mip=0,
            fill_missing=True
        )
        precomputed_vol.provenance.description = f"Synthetic em {vol_type}: {version}, {seed}"
        precomputed_vol.provenance.owners = [
            'sm2667@cam.ac.uk', 'mclayton@mrc-lmb.cam.ac.uk', 'ac2040@cam.ac.uk']
        precomputed_vol.commit_info()
        precomputed_vol.commit_provenance()

        print(
            f"Writing precomputed dataset at {out_path} from zarr dataset at {in_path}\n")
        offsets = []
        # split dataset
        num_x, num_y, num_z = size[0]//chunk_size[0], size[1]//chunk_size[1], size[2]//chunk_size[2]
        if verbose:
            print(
                f"Splitting dataset into chunks : x{num_x}, y{num_y}, z{num_z}")
        for x in range(num_x + 1):          # +1 to include the boundary
            for y in range(num_y + 1):
                for z in range(num_z + 1):
                    offsets.append((x, y, z))

        batch_size = 10
        if verbose:
            print(
                f"{len(offsets)} offsets downloaded. Batching offsets (batch_size: {batch_size})\n")
        offset_chunks = [offsets[i:i + batch_size]
                         for i in range(0, len(offsets), batch_size)]
        num_workers = cpu_count() - 1  # Leave one CPU free
        with Pool(num_workers) as pool:
            with tqdm(total=len(offset_chunks), desc="Processing chunks") as pbar:
                for _ in pool.imap_unordered(partial(process_chunk, arr, precomputed_vol, vol_type, chunk_size), offset_chunks):
                    pbar.update()

        return os.path.join(out_path, dataset_name)


def downsample(cloud_path, vol_type, mip=0, num_mips=2, compression='gzip', volumetric=False):
    print(f"Downsampling precomputed dataset at {cloud_path}")
    num_workers = cpu_count() - 1  # Leave one CPU free
    with LocalTaskQueue(parallel=num_workers) as tq:
        tasks = tc.create_downsampling_tasks(
            # e.g. 'gs://bucket/dataset/layer'
            f"precomputed://file://{cloud_path}",
            # Start downsampling from this mip level (writes to next level up)
            mip=mip,
            fill_missing=True,  # Ignore missing chunks and fill them with black
            axis='z',
            # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
            num_mips=num_mips,
            # chunk_size=None, # manually set chunk size of next scales, overrides preserve_chunk_size
            # use existing chunk size, don't halve to get more downsamples
            preserve_chunk_size=True,
            sparse=False,  # for sparse segmentation, allow inflation of pixels against background
            bounds=None,  # mip 0 bounding box to downsample
            # e.g. 'raw', 'compressed_segmentation', etc
            encoding='compressed_segmentation' if vol_type == 'segmentation' else 'raw',
            # issue a delete instead of uploading files containing all background
            delete_black_uploads=False,
            background_color=0,  # Designates the background color
            # None, 'gzip', and 'br' (brotli) are options
            compress=compression,
            # common options are (2,2,1) and (2,2,2)
            factor=(2, 2, 2) if volumetric else (2, 2, 1),
        )
        # performs on-line execution (naming is historical)
        tq.insert_all(tasks)


def skeletonize(cloud_path, mip=0):
    print(f"Writing kimimaro skeletons of precomputed dataset at {cloud_path}")
    match = re.search(r"v(\d+)_seed(\d+)", cloud_path)
    if match:
        version = match.group(1)
        seed = match.group(2)
    else:
        raise ValueError(
            f"Could not parse dataset version and seed from path {cloud_path}")

    skels_path = f"/cajal/scratch/projects/misc/riegerfr/synem/syn_cubes/skeleton_v{version}_seed{seed}.pklkimimaro"
    skels = pkl.load(open(skels_path, 'rb'))

    vol = CloudVolume("file://" + cloud_path)

    # write skeletons
    for skel in skels.values():
        vol.skeleton.upload(skel)

    # write info file
    transform = skel.transform.astype(int).reshape(-1).tolist()
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": transform,
        "vertex_attributes": [],
        "spatial_index": None
    }

    with open(cloud_path + "/skeletons/info", "w") as f:
        json.dump(info, f)


def meshing(cloud_path, mip=0, error=40, dust_threshold=512):
    print(f"Generating meshes of precomputed dataset at {cloud_path}")
    num_workers = cpu_count() - 1  # Leave one CPU free
    with LocalTaskQueue(parallel=num_workers) as tq:
        # First Pass
        tasks = tc.create_meshing_tasks(
            f"precomputed://file://{cloud_path}",  # Which data layer
            # Which resolution level to mesh at (we often choose near isotropic resolutions)
            mip,
            # shape=chunk_size,  # Size of a task to mesh, chunk alignment not needed
            simplification=True,  # Whether to enable quadratic edge collapse mesh simplification
            # Maximum physical deviation of mesh vertices during simplification
            max_simplification_error=error,
            mesh_dir=None,  # Optionally choose a non-default location for saving meshes
            dust_threshold=dust_threshold,  # Don't bother meshing below this number of voxels
            object_ids=None,  # Optionally, only mesh these labels.
            # Display a progress bar (more useful locally than in the cloud)
            progress=True if verbose else False,
            # If part of the data is missing, fill with zeros instead of raising an error
            fill_missing=False,
            # 'precomputed' or 'draco' (don't change this unless you know what you're doing)
            encoding='precomputed',
            spatial_index=False,  # generate a spatial index for querying meshes by bounding box
            sharded=False  # generate intermediate shard fragments for later processing into sharded format
        )
        # performs on-line execution (naming is historical)
        tq.insert_all(tasks)

        # Second Pass
        tasks = tc.create_mesh_manifest_tasks(
            f"precomputed://file://{cloud_path}",
            # high magnitude (3-5+) is appropriate for horizontal scaling workloads
            # while small magnitudes (1-2) are more suited for small volumes locally processed
            magnitude=2
        )
        # performs on-line execution (naming is historical)
        tq.insert_all(tasks)


def show(cloud_path, host='localhost', port=1337):
    dataset_name = os.path.basename(os.path.normpath(cloud_path))
    match = re.search(r"v(\d+)_seed(\d+)", cloud_path)
    if match:
        version = match.group(1)
        seed = match.group(2)
    else:
        raise ValueError(
            f"Could not parse dataset version and seed from path {cloud_path}")

    sources = []
    root = f"precomputed://https://syconn.esc.mpcdf.mpg.de/synem/seg/v{version}_seed{seed}"
    sources.append(root)

    skel_path = glob.glob(cloud_path + "/skeletons*")
    mesh_path = glob.glob(cloud_path + "/mesh*")

    if len(skel_path) != 0:
        name = os.path.basename(os.path.normpath(skel_path[-1]))
        sources.append(root + f"/{name}")

    if len(mesh_path) != 0:
        name = os.path.basename(os.path.normpath(mesh_path[-1]))
        sources.append(root + f"/{name}")

    neuroglancer.set_server_bind_address(bind_address=host, bind_port=port)
    viewer = neuroglancer.Viewer(token=dataset_name)

    with viewer.txn() as s:
        layer = neuroglancer.SegmentationLayer if vol_type == "segmentation" else neuroglancer.ImageLayer
        s.layers[vol_type] = layer(
            source=sources
        )

    return viewer.get_viewer_url()


if __name__ == "__main__":
    parser = create_parser()
    args = parse_arguments(parser)
    handle_tasks(args)
