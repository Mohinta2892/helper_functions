import dask.array as da
import zarr
from dask.diagnostics import ProgressBar

# Load the Zarr array
zarr_ds = 'path/to/your/zarr'
dask_array = da.from_zarr(zarr_ds, component='path/to/dataset')

# Rechunk
new_chunks = (256, 256, 256)  # Adjust as needed
rechunked_array = dask_array.rechunk(new_chunks)

# Save to new Zarr store
with ProgressBar():
    rechunked_array.to_zarr('path/to/new/zarr/store', overwrite=True)
