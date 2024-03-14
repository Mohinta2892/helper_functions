#!/Users/cardonagroup/miniconda3/envs/napari-env/bin/python

"""
NB: Ensure you have a good processor and enough RAM because we use `NAPARI`, which loads everything into RAM.
This script has not been rigorously tested, hence may still have underlying bugs when tested on a different OS.

Installation:
Napari:
https://napari.org/stable/tutorials/fundamentals/installation
pip install scikit-image==0.22.0
pip install zarr==2.16.1
pip install dask==2023.12.1
pip install numpy==1.26.0
pip install matplotlib==3.8.0
pip install scipy==1.11.3
pip install tqdm==4.66.1

Functionality:
- This script loads a zarr containing raw, neuron labels and mitochondria and visualises them in napari.
- Click on `seg` layer in napari window. Press `4`. This will let you now hover and choose the colours you want from display.
Click to choose the colour and press `a` after every choice to add them to a list.
- Click on the dock where all layers displayed on your left. Press `p` to only visualise choosen colors.
Pressing `p` also automatically in-paints and displays the in-painted and intermediate volumes.
Moreover, it creates an affinity matrix and display it.
- Press `z` to clear the list.

Run:
path/to/extract_clean_somas.py -f path/to/zarr -d /path/to/output_folder (this is default in the script)
Please comment the shebang in Line1 if you do the below:
$ conda activate napari-env
$ python path/to/extract_clean_somas.py -f path/to/zarr -d /path/to/output_folder

Test ENV:
Distributor ID:	Ubuntu
Description:	Ubuntu 22.04.3 LTS
Release:	22.04
Codename:	jammy
RAM specs: 126GB
Processor: AMD EPYC 7513 32-Core Processor
Author: Samia Mohinta
Affiliation: Cardona lab, University of Cambridge, UK

"""

import argparse
import os.path

import napari
import skimage
import zarr
import numpy as np
import dask.array as da
from skimage.morphology import closing, square, ellipse, area_closing
from skimage.restoration import inpaint
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage import morphology, measure
import time
from datetime import datetime

viewer = napari.Viewer()
picked_labels = []
relab = []
cher_pick_f = ""


def seg_to_affgraph(seg, nhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]]):
    nhood = np.array(nhood)

    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    dims = nhood.shape[1]
    aff = np.zeros((nEdge,) + shape, dtype=np.int32)

    if dims == 2:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            ]
                            > 0
                    )
            )

    elif dims == 3:
        for e in range(nEdge):
            aff[
            e,
            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
            ] = (
                    (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            == seg[
                               max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                               max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                               max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                               ]
                    )
                    * (
                            seg[
                            max(0, -nhood[e, 0]): min(shape[0], shape[0] - nhood[e, 0]),
                            max(0, -nhood[e, 1]): min(shape[1], shape[1] - nhood[e, 1]),
                            max(0, -nhood[e, 2]): min(shape[2], shape[2] - nhood[e, 2]),
                            ]
                            > 0
                    )
                    * (
                            seg[
                            max(0, nhood[e, 0]): min(shape[0], shape[0] + nhood[e, 0]),
                            max(0, nhood[e, 1]): min(shape[1], shape[1] + nhood[e, 1]),
                            max(0, nhood[e, 2]): min(shape[2], shape[2] + nhood[e, 2]),
                            ]
                            > 0
                    )
            )

    else:
        raise RuntimeError(f"AddAffinities works only in 2 or 3 dimensions, not {dims}")
    print(f"aff values: {np.unique(aff)}")
    return aff


@viewer.bind_key('a')
def pick_labels(viewer):
    selected_label = viewer.layers['seg'].selected_label
    global pick_labels
    picked_labels.append(selected_label)
    print(selected_label)
    print(picked_labels)


@viewer.bind_key('z')
def clear_labels(viewer):
    global picked_labels
    picked_labels = []
    print(picked_labels)
    viewer.layers['seg'].visible = True
    viewer.layers.selection.discard('cherry_pick_seg')
    viewer.layers.selection.discard('aff')


def remove_holes_2d(mask):
    mask = mask.astype(np.bool_)
    z_shape = mask.shape[0]
    for i in range(z_shape):
        mask[i] = morphology.remove_small_holes(mask[i], area_threshold=4096, connectivity=4)
    new_mask = (mask.astype(np.uint8) * 255).astype(np.uint8)
    return new_mask


def remove_smalls_2d(mask):
    for i in range(mask.shape[0]):
        new_mask = morphology.remove_small_objects(mask[i].astype(np.bool_), min_size=2048, connectivity=8)
        mask[i] = (new_mask.astype(np.uint8) * 255).astype(np.uint8)
    return mask


def remove_smalls(mask, min_size=20480):
    mask = mask.astype(np.bool_)
    new_mask = morphology.remove_small_objects(mask, min_size=min_size, connectivity=1)
    new_mask = (new_mask.astype(np.uint8) * 255).astype(np.uint8)
    return new_mask


@viewer.bind_key('p')
def pick_in_paint(viewer):
    msg = "cherry-picking display on!!"
    print(msg)
    global picked_labels
    global relab
    cherry_pick_seg = np.zeros_like(relab)  # .astype(np.uint16)
    in_paint_seg = np.zeros_like(relab).astype(np.uint16)

    start = time.time()
    if len(picked_labels):
        for x in picked_labels:
            cherry_pick_seg[relab == x] = 1  # it is mask
            # print(cherry_pick_seg.dtype)
            # viewer.add_image(cherry_pick_seg, name='cherry_inter')

            # # Step1 #
            # thre_mask = remove_holes_2d((cherry_pick_seg * 255).astype(np.uint8))
            # thre_mask = thre_mask.astype(np.uint8)

            # # Step2 #
            # removed_both2d_3d_mask = remove_smalls_2d(thre_mask)
            # removed_both2d_3d_mask = remove_smalls(removed_both2d_3d_mask, min_size=102400)

            # # Step3 #
            # result = measure.label(removed_both2d_3d_mask.astype(bool), connectivity=1)
            # result = result.astype(np.uint16)
            # in_paint_seg += result
            # in_paint_seg[relab == 1] = x

            in_paint_seg += cherry_pick_seg
            in_paint_seg[relab == 1] = x

        viewer.add_labels(in_paint_seg, name='picked_labels')

        # viewer.add_labels(in_paint_seg, name='cherry_pick_seg')
        # viewer.layers['seg'].visible = False
        # once cherry-picked, generate and display the affinity map
        # aff = seg_to_affgraph(in_paint_seg)
        # viewer.add_image(aff, name='aff', opacity=0.7, blending='additive')

        # if np.sum(in_paint_seg) > 0:
        global cher_pick_f
        cher_pick_f["volumes/labels/neuron_ids"] = in_paint_seg
        cher_pick_f["volumes/labels/cherry_pick_seg"] = (cherry_pick_seg*255).astype(np.uint8)

            # we set it here since we do not have access to offset and res in `pick_in_paint()`
        cher_pick_f["volumes/labels/neuron_ids"].attrs["offset"] = (0,0,0)
        cher_pick_f["volumes/labels/neuron_ids"].attrs["resolution"] = (8,8,8)
        cher_pick_f["volumes/labels/cherry_pick_seg"].attrs["offset"] = (0,0,0)
        cher_pick_f["volumes/labels/cherry_pick_seg"].attrs["resolution"] = (8,8,8)


        
        end = time.time()

        print(f"Time spent till saving {end - start}")
        viewer.close()
        return 1

    return 0


def pick_somas(filename, out_dir=None):
    if out_dir is None:
        out_dir = os.path.dirname(filename)

    f = zarr.open(filename, mode='r')
    date_time = datetime.now()
    # format specification
    format = '%Y-%m-%d %H:%M:%S'

    # applying strftime() to format the datetime
    dt_str = date_time.isoformat().strip(" /")


    global cher_pick_f  # a global cherry-picked and inpainted zarr
    cher_pick_f = zarr.open(os.path.join(out_dir,
                                         f"{os.path.basename(filename).split('.zarr')[0]}_picked_{dt_str}.zarr"), mode='a')
    crop_roi=(slice(0,f["volumes/raw"].shape[0],None), slice(0,f["volumes/raw"].shape[1],None), slice(0,f["volumes/raw"].shape[2], None))
    seg = f["volumes/labels/neuron_ids"][crop_roi]
    global relab
    relab, _, _ = skimage.segmentation.relabel_sequential(seg)
    relab = relab.astype(np.uint16)
    relab_mito, _, _ = skimage.segmentation.relabel_sequential(f["volumes/labels/mito_ids"][crop_roi])
    relab_mito = relab_mito.astype(np.uint16)

    viewer.add_labels(relab, name='seg')
    viewer.add_image(f["volumes/raw_clahe"][crop_roi], name='raw', opacity=0.5, blending='additive')
    #viewer.add_labels(relab_mito, name='mito', opacity=0.5, blending='additive')
    #viewer.layers["mito"].visible = False

    cher_pick_f["volumes/raw"] = f["volumes/raw_clahe"][:]
    cher_pick_f["volumes/labels/mito_ids"] = relab_mito

    # set the resolution
    res = f["volumes/raw_clahe"].attrs["resolution"]
    offset = f["volumes/raw_clahe"].attrs["offset"]

    cher_pick_f["volumes/raw"].attrs["offset"] = offset
    cher_pick_f["volumes/raw"].attrs["resolution"] = res

    cher_pick_f["volumes/labels/mito_ids"].attrs["offset"] = offset
    cher_pick_f["volumes/labels/mito_ids"].attrs["resolution"] = res

    # we set it here since we do not have access to offset and res in `pick_in_paint()`
    # cher_pick_f["volumes/labels/neuron_ids"].attrs["offset"] = offset
    # cher_pick_f["volumes/labels/neuron_ids"].attrs["resolution"] = res



    napari.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="Path/to/zarr")
    parser.add_argument("-d", default= "/Users/cardonagroup/Documents/hemibrain_soma_bbox/soma_picked", required=False, help="Path/to/output/folder")

    args = parser.parse_args()

    pick_somas(args.f, args.d)



