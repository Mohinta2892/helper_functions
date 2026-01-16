import h5py
import zarr
import numpy as np

old_h5 = h5py.File("/media/samia/DATA/mounts/zstore1/catena/data/OCTO_CORRECTED_SAME_PREID/data_3d/train/octo_cube2_12485_13164_y6231_6901_z3971_4640.hdf","r")
new_h5 = h5py.File("/media/samia/DATA/mounts/zstore1/catena/data/COMBINED_NEURIPS_SAME_PREID/data_3d/octo_corr_tpfp_19aug2025/octo_cube2_19aug25_tpfp_12485_13164_y6231_6901_z3971_4640.hdf", "a")

seg_v_c = old_h5["volumes/labels/neuron_ids"][...]

new_h5["volumes/labels/neuron_ids"] = seg_v_c

new_h5["volumes/labels/neuron_ids"].attrs["resolution"] = old_h5["volumes/labels/neuron_ids"].attrs["resolution"]
new_h5["volumes/labels/neuron_ids"].attrs["offset"] = old_h5["volumes/labels/neuron_ids"].attrs["offset"]


