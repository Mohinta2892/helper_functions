import zarr
from numcodecs import Blosc
import dask.array as da
import numpy as np
from zarr.n5 import N5FSStore
from dask.diagnostics import ProgressBar
from dask.callbacks import Callback
from tqdm.auto import tqdm


# custom progressbar to visualize save progress:  https://github.com/tqdm/tqdm/issues/278#issuecomment-649810339
class ProgressBar(Callback):
    def _start_state(self, dsk, state):
        self._tqdm = tqdm(total=sum(len(state[k]) for k in ['ready', 'waiting', 'running', 'finished']))

    def _posttask(self, key, result, dsk, state, worker_id):
        self._tqdm.update(1)

    def _finish(self, dsk, state, errored):
        pass


# read the n5 dir containing data
in_store = N5FSStore('/media/samia/DATA/mounts/zstore1/FIBSEM_L1120_FullCNS_8x8x8nm')
dataset_scale = 's0'
in_file = zarr.Array(in_store, path=dataset_scale, read_only=True)  # path can be any scale from s0 to s11 for example

# a `zarr.core.Array` containing the data
d_arr = da.from_zarr(in_file)
shape = d_arr.shape

dtype = np.uint8  # replace with dtype of choice
chunks = 1000, 1000, 1000  # replace with chunk shape of choice
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)  # replace with compressor of choice,example gzip

# set up zarr array to store data
out_store = zarr.DirectoryStore(f'/media/samia/DATA/ark/code-experiments/octo-{dataset_scale}.zarr')
root = zarr.group(out_store)

""" Sanity check code, uncomment to test blank runs 
# TB1 = root.zeros('data',
#                  shape=shape,
#                  chunks=chunks,
#                  compressor=compressor,
#                  dtype=dtype)
"""

# create a dataset `volumes/raw` here inside the zarr root group
TB1 = root.create_dataset('volumes/raw',
                          shape=shape,
                          chunks=chunks,
                          compressor=compressor,
                          dtype=dtype,
                          overwrite=False)

TB1.attrs['offset'] = (0, 0, 0)
TB1.attrs['resolution'] = (8, 8, 8)

# uncomment if data not needed to be rechunked into prescribed chunk shape above
z = da.rechunk(d_arr, chunks=chunks)

"""
To be used with sanity check code above
# set up a dask array with random numbers
# d = da.random.randint(0, 3, size=shape, dtype=dtype, chunks=chunks)

"""

# make this a delayed process so that it is only a placeholder and does not initiate the n5 to zarr saving yet!
# we want to use the progress bar when it does.

delayed_arr = z.store(TB1, lock=False, compute=False)

with ProgressBar():
    # compute and store the random numbers
    delayed_arr.compute()
