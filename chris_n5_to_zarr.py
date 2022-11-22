"""
Uses dask to load zarr, n5 and hdf files.
Main function - Can save files as zarr or hdf. But note, it can only convert one dataset to a hdf.
Functionality to save a complete zar/n5 to hdf yet to be supported.

Utilities:
transpose_ndarray: Transpose an array from xyz to zyx orientation. Assumes all input arrays to have zyx orientation.
Cut out a small ndarray from the original input array
Average data across all affinity channels
Scale array values to be in the range of 0 to 255 and convert to uint8

"""
import shutil

import dask.array
import zarr
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage


def save_to_zarr(d_arr, out_file, dataset='volumes/raw', overwrite=True,
                 offset=(0, 0, 0), resolution=(8, 8, 8)):
    try:
        if os.path.exists(out_file):
            shutil.rmtree(out_file)
        dask.array.to_zarr(d_arr, out_file, component=dataset, overwrite=overwrite)
        f = zarr.open(out_file, 'a')
        f['volumes/raw'].attrs['offset'] = offset
        f['volumes/raw'].attrs['resolution'] = resolution

    except Exception as e:
        print(e)

    print(f"Saved successfully .hdf here \n {out_file} ")


def zarr_to_hdf(d_arr, out_file, dataset):
    try:
        # if isinstance(d_arr, dask.array.array):
        if os.path.exists(out_file):
            shutil.rmtree(out_file)

        d_arr.to_hdf5(out_file, dataset)
        # else:

    except Exception as e:
        d_arr = dask.array.from_array(d_arr)
        d_arr.to_hdf5(out_file, dataset)
        # print(e)

    print(f"Saved successfully .hdf here \n {out_file} ")


def transpose_ndarray(d_arr, reshape_pattern=(2, 1, 0)):
    """
    Changes the orientation of a ndarray from (x,y,z) --> (z, y, x)
    Args:
        d_arr: ndarray of shape (x,y,z) or (c, x, y, z)
        reshape_pattern: axes for rearrangement, default (z, y, x)

    Returns:
        An ndarray of orientation (z, y, x) or (c, z, y, x) and of the same shape as input d_arr
    """
    if len(d_arr.shape) > 3:
        reshape_pattern = (0, 3, 2, 1)

    d_arr = np.transpose(d_arr, reshape_pattern)

    return d_arr


def drop_channel_dim(d_arr: np.ndarray) -> np.ndarray:
    """
    Averages the data across the channel dimension of the input array (only suitable for predicted affinity maps).

    Args:
        d_arr: ndarray of shape (c,z,y,x)

    Returns:
        An ndarray of shape (z, y, x)
    """
    # average all channel dims
    d_arr = (d_arr[0, ...] + d_arr[1, ...] + d_arr[2, ...]) / 3.0

    return d_arr


def slice_arr(d_arr, slices=None):
    """
    Cuts out a portion of the input ndarray based on user's positional input.
    Args:
        d_arr:  ndarray of shape  (z, y, x) or (c, z, y, x)
        slices: a list of slice type objects

    Returns:
        An ndarray that is smaller or of the same size as the original
    """
    try:
        if len(d_arr.shape) > 3:
            slices = [slice(None)] + slices

        # slices across z, y, x, if channel has to slice, please pass accordingly
        assert slices is not None, "Slices cannot be None"
        d_arr = d_arr[tuple(slices)]

    except Exception as e:
        print(e)

    return d_arr


def invert_arr(d_arr):
    """

    Invert image pixels (black to white, vice versa). Done only for plant-seg's segmentation to work

    Args:
        d_arr: An ndarray of shape (z, y, x)

    Returns:
        An inverted ndarray of the same input shape.


    """
    d_arr = skimage.util.invert(d_arr)

    # sanity check via plotting a slice
    plot_ = d_arr[0, ...]
    plt.imshow(plot_, cmap='gray')

    plt.show()

    return d_arr


def scale_arr_0to255(d_arr):
    # scale 0 to 255
    d_arr = d_arr * 255  # 2**32

    # sanity check - probability map values
    print(np.amax(d_arr), np.amin(d_arr))

    # convert floats to uint8
    d_arr = d_arr.astype(np.uint8)

    # sanity check - probability map values after casting to uint8
    print(np.amax(d_arr), np.amin(d_arr))

    return d_arr


def read_zarr(in_file, dataset):
    container = zarr.open(in_file, mode="r")
    print(container.tree())
    arr = container[dataset]
    d_arr = dask.array.from_zarr(arr)

    return d_arr


def read_hdf(in_file, dataset):
    container = h5py.File(in_file, mode="r")
    arr = container[dataset]

    # lazy loading into a dask array - meaning they do not load any data
    d_arr = dask.array.from_array(arr)
    return d_arr


def main():
    """ Creates dask arrays before any further actions on these"""

    # TODO- make these argparse
    in_file = "/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/G2019S/zarr/data.zarr"
    # note: in-dataset and out dataset will be same as of now
    # Todo- make use of zarr.tree() to generate arrays to save as hdf
    dataset = "volumes/raw"
    # dataset = "volumes/pred_affs"
    # TODO- generate this from slices below
    suffix_outfile = "z120-140_y100-300_x100-300.hdf"
    out_file = f"/mnt/wwn-0x5000c500e0dbd55e/ark/snapshots_3dmtlsd-hemi/{os.path.basename(in_file).split('.')[0]}" \
               f"_{suffix_outfile}"

    # if orientation is zyx then do not transpose, else transpose
    orientation_zyx = True
    drop_channel_dims = False

    # Do you want to:
    slice_array = True
    slices = {'z': [120, 140], 'y': [100, 300], 'x': [100, 300], }

    # scale array values from 0 to 255
    scale_array = False

    # only to do segmentation via plant-seg
    invert_array = False

    # save to hdf; False saves it as zarr
    save_to_hdf = True

    try:
        if in_file.endswith('.n5'):
            store = zarr.N5FSStore(in_file)  # also works with a URL
            d_arr = read_zarr(in_file, dataset)

        elif in_file.endswith('.zarr'):
            d_arr = read_zarr(in_file, dataset)

        elif in_file.endswith('.hdf'):
            d_arr = read_hdf(in_file, dataset)

    except Exception as e:
        print(f"Not implemented {in_file.split('.')[-1]} type yet!")

    d_arr = np.asarray(d_arr)

    # reshape xyz orientation zyx
    if not orientation_zyx:
        d_arr = transpose_ndarray(d_arr, reshape_pattern=(2, 1, 0))

    if drop_channel_dims:
        # averages values of all channel dims
        d_arr = drop_channel_dim(d_arr)
    if slice_array:
        assert isinstance(slices, dict), "Slices must be a dictionary"
        try:

            slices_ = [slice(values[0], values[1]) for keys, values in slices.items()]
            d_arr = slice_arr(d_arr, slices_)

        except Exception as e:
            print(e)

    if scale_array:
        d_arr = scale_arr_0to255(d_arr)

    if invert_array:
        d_arr = invert_arr(d_arr)

    if save_to_hdf:
        zarr_to_hdf(d_arr, out_file, dataset)
    else:
        save_to_zarr(d_arr, out_file, dataset)


if __name__ == '__main__':
    main()
