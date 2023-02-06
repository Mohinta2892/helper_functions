import zarr
import tifffile
f = tifffile.imread('path/to/tiff')
outfile = 'path/to/zarr/filename.zarr'
ds = 'volumes/labels/neuron_ids' # example dataset
o = zarr.open(outfile, mode='w')
o[ds] = f
o[ds].attrs['resolution'] = (8,8,8)
o[ds].attrs['offset'] = (0,0,0)
