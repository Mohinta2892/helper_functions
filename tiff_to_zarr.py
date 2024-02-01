import zarr
import tifffile
import numpy as np

f = tifffile.imread('path/to/tiff')
outfile = 'path/to/zarr/filename.zarr'
ds = 'volumes/labels/neuron_ids' # example dataset

# if you are dealing with tifs that are saved as `labels.tif` (containing labels), this helps them to load them into napari later
# f = f.astype(np.uint64)

o = zarr.open(outfile, mode='w')
o[ds] = f
o[ds].attrs['resolution'] = (8,8,8)
o[ds].attrs['offset'] = (0,0,0)
