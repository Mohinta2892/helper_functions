"""
Program desc:
Splits an n5 directory source into regions of interest parallely.
Checks if the roi actually contains neural data or is completely background (Resin). Discards solely resin rois from
being saved to disk.

Input n5 : An n5 directory source that is read in as zarr N5Filestore
Input mask: A downsampled mask which distinguishes foreground (necessary in roi) and backhground (to be discarded)
Output: Regions of interest of predefined size in the code

Parallel processing: Uses Python multiprocessing Pool
Warning: Can be slow, future rewrite with daisy/dask

Author: Samia Mohinta
Mask generated and implementation logic - Pedro Gómez-Gálvez

"""

import numpy as np
import zarr
from multiprocessing import Pool
from zarr.n5 import N5FSStore
import dask.array as da
from tqdm import tqdm
import tifffile as tf

"""

Notes:
Matrix shape of octo scale 0 in zyx:(34745, 21471, 20486)
Matrix shape of octo scale 6 in zyx:(542, 335, 320)
Ratio changes = 542/34745 = 0.02; 0.02; 0.02

Example:
Patch size per 1568^3 roi is 31^3 in the scale 0 mask
mask_patch = mask[:31, :31, :31]
"""

in_store = N5FSStore('/media/samia/DATA/mounts/zstore1/FIBSEM_L1120_FullCNS_8x8x8nm')
in_file = zarr.Array(in_store, path='s0', read_only=True)
d_arr = da.from_zarr(in_file)
shape = d_arr.shape

# read scale 6 mask to separate foreground from background data
scale_0_mask = tf.imread('/media/samia/DATA/ark/code-experiments/scale_0_mask/ottoMaskBrain_s6.tif')

# Define the shape of the chunks to be saved
chunk_shape = (256, 256, 256)

# calculate the scale factor. used to transform the scale 0 indices to scale 6 level
factors_zyx = [542 / 34745, 335 / 21471, 320 / 20486]


# calculate transformed coordinates of scale 0 ndarray as per scale 6 shape
def multiply_slices(slices, factor=factors_zyx):
    """
    :param slices: tuple of slice objects pointing to the region of interest (ROI) in the larger ndarray
    :param factor: list of floats to be used to scale the indices of the larger ndarray ROI to be mask shape
    :return: tuple of slice objects that have been scaled to the scale 6 mask
    """
    return tuple(slice(int(s.start * factor[0]) if s.start is not None else None,
                       int(s.stop * factor[1]) if s.stop is not None else None,
                       int(s.step * factor[2]) if s.step is not None else None)
                 if s is not None else None for s in slices)


def check_for_actual_data(slices):
    """

    :param slices:
    :return:
    """
    mask_patch = scale_0_mask[slices]

    mask_contains_foreground = np.nonzero(mask_patch)
    print(mask_contains_foreground)

    if np.count_nonzero(mask_patch) > 0:
        return True

    return False


# Define a function that reads a 3D region of interest and saves it as a chunk in a .zarr file
def read_roi_and_save_chunk(filename, region_of_interest):
    # Read the 3D region of interest from the large ndarray
    roi_data = d_arr[region_of_interest]
    # for now check 8 vertices of cube with mask and check if data is there, otherwise reject the roi from being saved
    slices = multiply_slices(region_of_interest, factors_zyx)
    print(region_of_interest, slices)
    if check_for_actual_data(slices):
        roi_data = roi_data.rechunk(chunks='auto')  # replace `auto` with chunk_shape when needed
        # Open the zarr file for writing and write the data to the file
        roi_data.to_zarr(filename, component='volumes/raw')
        with zarr.open(filename, mode='a') as z:
            z['volumes/raw'].attrs['offset'] = (0, 0, 0)
            z['volumes/raw'].attrs['resolution'] = (8,) * 3
    else:
        # skip that roi_data from saving on disk
        print(f"Skipped this: {filename}")

    # update the progress
    pbar.update(1)


# Define the number of workers to use
num_workers = 1

# Define the size of the regions of interest
roi_size = (1568, 1568, 1568)

# Define the filename for the .zarr file
filename = '/media/samia/DATA/ark/code-experiments/sub-rois-scale0/octo-s0-subroi'

# Create a list of regions of interest to process
regions_of_interest = []
for i in range(0, shape[0], roi_size[0]):
    for j in range(0, shape[1], roi_size[1]):
        for k in range(0, shape[2], roi_size[2]):
            region_of_interest = (slice(i, i + roi_size[0]), slice(j, j + roi_size[1]), slice(k, k + roi_size[2]))
            regions_of_interest.append(region_of_interest)

# progress bar
pbar = tqdm(total=len(regions_of_interest))

# Create a pool of workers and use it to process the regions of interest
with Pool(num_workers) as p:
    p.starmap(read_roi_and_save_chunk,
              [(f"{filename}_z{region_of_interest[0].start}-{region_of_interest[0].stop}_"
                f"y{region_of_interest[1].start}-{region_of_interest[1].stop}_"
                f"x{region_of_interest[2].start}-{region_of_interest[2].stop}.zarr", region_of_interest)
               for region_of_interest in regions_of_interest])
