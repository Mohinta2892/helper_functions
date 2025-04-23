# Very basic for writing zarr datasets back to tiff
import zarr
import tifffile as t
f = zarr.open("/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/test/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.zarr")
raw = f["volumes/raw"]
t.imwrite("/media/samia/DATA/mounts/zstore1/catena/data/OCTO_3CUBES_ZYX/data_3d/test/octo_cube3_calyx_5603_6267_y3254_3890_z7464_8163.tiff", raw)
