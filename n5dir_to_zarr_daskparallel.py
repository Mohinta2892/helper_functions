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


in_store = N5FSStore('/media/samia/DATA/mounts/zstore1/FIBSEM_L1120_FullCNS_8x8x8nm')
# shape = 1_000_000, 1_000_000
in_file = zarr.Array(in_store, path='s0', read_only=True)
d_arr = da.from_zarr(in_file)
shape = d_arr.shape
# print(d_arr.shape, in_file.info)

dtype = np.uint8  # 'i2'
chunks = 1000, 1000, 1000  # 20_000, 5_000
compressor = Blosc(cname='lz4', clevel=5, shuffle=Blosc.BITSHUFFLE)

# set up zarr array to store data
out_store = zarr.DirectoryStore('/media/samia/DATA/ark/code-experiments/octo-s0.zarr')
root = zarr.group(out_store)
# TB1 = root.zeros('data',
#                  shape=shape,
#                  chunks=chunks,
#                  compressor=compressor,
#                  dtype=dtype)

TB1 = root.create_dataset('volumes/raw',
                          shape=shape,
                          chunks=chunks,
                          compressor=compressor,
                          dtype=dtype,
                          overwrite=False)

TB1.attrs['offset'] = (0, 0, 0)
TB1.attrs['resolution'] = (8, 8, 8)

z = da.rechunk(d_arr, chunks=chunks)

# set up a dask array with random numbers
# d = da.random.randint(0, 3, size=shape, dtype=dtype, chunks=chunks)
delayed_arr = z.store(TB1, lock=False, compute=False)
with ProgressBar():
    # compute and store the random numbers
    delayed_arr.compute()
