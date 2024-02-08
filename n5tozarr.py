#!/usr/bin/env python3

"""
Adapted from Chris Barnes's n5 to mrc script to extend it's functionality to be able to convert n5 to zarr.
TO DO - n5 to hdf5
"""

from __future__ import annotations

import typing as tp
from argparse import ArgumentParser
from contextlib import contextmanager
from pathlib import Path

import dask.array
import pandas as pd
import zarr
from zarr.n5 import N5FSStore


def parse_subvolume(s):
    """
    Does not support negative dims!
    :param s:
    :return:
    """
    out = []
    for dim in s.split(","):
        dim = dim.strip()
        if dim == "...":
            out.append(Ellipsis)
            continue

        parts = []
        for part in dim.split(":"):
            if not part:
                parts.append(None)
            else:
                parts.append(int(part))
        if len(parts) > 2:
            raise ValueError("Each dimension can only have 2 or fewer parts")
        if (
                len(parts) == 2
                and parts[0] is not None
                and parts[1] is not None
                and parts[0] >= parts[1]
        ):
            raise ValueError("Cannot reverse dimensions")
        if ":" in dim:
            out.append(slice(*parts))
        else:
            out.append(parts[0])
    return tuple(out)


def parse_compression(fpath: Path):
    kwargs = dict()
    suff = Path(fpath).suffix
    if suff is None:
        return kwargs
    suff = suff.lower()
    if suff == ".bz2":
        kwargs["compression"] = "bzip2"
    elif suff == ".gz":
        kwargs["compression"] = "gzip"
    if kwargs:
        raise NotImplementedError("Compression is not supported")
    return kwargs


def parse_args(args=None):
    p = ArgumentParser()
    p.add_argument("container", help="path to N5 root")
    p.add_argument("dataset", help="path from N5 root to dataset")
    p.add_argument("outfile", type=Path, help="path to MRC file")
    p.add_argument("zarr", type=int, default=1,
                   help="Pass 1 to convert n5 to zarr, default converts raw to volumes/raw dataset")
    p.add_argument("mrc", type=int, help="Pass 1 to convert n5 to mrc: Credit Chris Barnes")
    p.add_argument(
        "-f", "--force", action="store_true", help="overwrite existing output file"
    )
    p.add_argument(
        "-s",
        "--subvolume",
        type=parse_subvolume,
        help=(
            "subvolume bounds (left-inclusive) as a string like "
            "'0:100,20:60,1000:20000'"
        ),
    )
    return p.parse_args(args)


def get_input(root, dataset):
    store = N5FSStore(root, mode="r")
    arr = zarr.Array(store, dataset, True)
    return dask.array.from_zarr(arr)


def to_hdf(z, outfile):
    # TODO
    pass


def to_zarr(z, outfile, resolution=(8, 8, 8), offset=(0, 0, 0)):
    z = dask.array.rechunk(z, chunks="auto")

    dask.array.to_zarr(z, outfile, component='volumes/raw', overwrite=True)
    f = zarr.open(outfile, 'a')
    f['volumes/raw'].attrs['offset'] = offset
    f['volumes/raw'].attrs['resolution'] = resolution  # (8, 8, 8)


@contextmanager
def get_output(fpath, shape, dtype, force=False):
    _ = parse_compression(fpath)

    with mrcfile.new_mmap(fpath, shape, mode_from_dtype(dtype), overwrite=force) as f:
        yield f


def main(
        container,
        dataset,
        outfile,
        subvolume: tp.Optional[tuple[slice, ...]] = None,
        force=False,
        parsed=None,
        resolution=(8, 8, 8),
        offset=(0, 0, 0)
):
    z = get_input(container, dataset)
    # Be careful to think of the orientation of the volume, meaning pass zyx if the n5 is oriented as zyx.
    if subvolume:
        z = z[subvolume]

    # if int(parsed.zarr):
    if True:  # crude
        to_zarr(z, outfile, resolution=resolution, offset=offset)

    # if int(parsed.mrc):
    #     with get_output(outfile, z.shape, z.dtype, force) as out:
    #         dask.array.store(z, out.data)


def _main(args=None):
    # parsed = parse_args(args)

    # parsed.container = '/media/samia/DATA/mounts/zstore1/FIBSEM_L1120_FullCNS_8x8x8nm'
    # parsed.dataset = 's0'
    # parsed.outfile = '/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/octo/' \
    #                  'zarr/crop_A1_z10030-10150_y13300-16336_x15987-17114.zarr'
    # parsed.subvolume = "10030:10150,13300:16336,15987:17114"

    # main(
    #     parsed.container, parsed.dataset, outfile, subvolume,  # parsed.force
    # )

    # OCTO - this is a crude local versions
    # container = '/media/samia/DATA/mounts/zstore1/FIBSEM_L1120_FullCNS_8x8x8nm'
    # dataset = 's0'
    # outfile = '/media/samia/DATA/ark/connexion/data/OCTO/data_3d/test/octo_lhemibrain_z6000-8500_y5000-9507_x5000-8189.zarr'
    # # subvolume = "28079:28640,13695:14239,9460:10116" # A7
    # # subvolume = "22471:23032,13095:13759,8700:9388" # A4
    # # subvolume = "16655:17216,13231:13903,7650:8468"  # A1
    # subvolume = "6000:8500,5000:9507,5000:8189"  # left hemi
    # resolution = (8, 8, 8)
    # offset = (0, 0, 0)

    # LEONARDO crop one hemisphere: voxel size is 30,12, 12 in zyx 42gb size
    # container = "/media/samia/DATA/ark/dan-samia/lsd/funke/leornado/s0_cut_for_segmentation.n5"
    # dataset = "s0"
    # outfile = "/media/samia/DATA/ark/connexion/data/LEONARDO/data_3d/test/leornado_mclayton_z512-1859_y0-4194_x200-4527.zarr"
    # subvolume = "512:,0:4194,200:"
    # resolution = (30, 12, 12)
    # offset = (0, 0, 0)

    # PARKER - 16,8,8
    # container = "/media/samia/DATA/mounts/parker/n5"
    # dataset = 's0'
    # outfile = "/media/samia/DATA/ark/connexion/data/PARKER/data_3d/test/parker_roi512_z9520-10032_y7797-8309_x13740-14252.zarr"
    # subvolume = "4760:5272,7797:8309,13740:14252"
    # resolution = (16, 8, 8)
    # offset = (0, 0, 0)

    # Seymour
    # container = "/media/samia/DATA/mounts/zstore1/0111-8_whole_L1_CNS.n5"
    # dataset = "volumes/raw/c0/s0"
    # outfile = "/media/samia/DATA/ark/dan-samia/lsd/funke/seymour/zarr/seymour_rhemi_z1044-1312_y5000-7000_x15000-19668.zarr"
    # subvolume = "1044:1312,5000:7000,15000:19668"
    # resolution = (54, 4, 4)
    # offset = (0, 0, 0)

    # Popeye ROI scale 2
    # container = "/media/samia/DATA/mounts/squiddata/popeye2"
    # dataset = 's2'
    # outfile = "/media/samia/DATA/ark/dan-samia/lsd/funke/squid/zarr/popeye2_z2800-3500_y3535-9780_x4615-12801.zarr"
    # subvolume = "2800:3500,3535:9780,4615:12801"
    # resolution = (30, 8, 8)
    # offset = (0, 0, 0)
    # main(
    #     container, dataset, outfile, parse_subvolume(subvolume), resolution=resolution, offset=offset  # parsed.force
    # )

    # Parker Roi Scale 0; resolution is in zyx: 16,8,8
    container = "/media/samia/DATA/mounts/zstore1/registration-Albert/n5"
    dataset = 's0'
    subvolume = "3831:4729,5871:6395,5998:6655"
    ranges = subvolume.split(',')
    formatted_suffix = "z{}_y{}_x{}".format(
        ranges[0].replace(':', '-'),  # Format the first range
        ranges[1].replace(':', '-'),  # Format the second range
        ranges[2].replace(':', '-')  # Format the third range
    )
    outfile = f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/zarr/parker_Danf1right_{formatted_suffix}.zarr"
    resolution = (8, 8, 8) # even though it is 16nm in z, download it as 8 to fool the network
    offset = (0, 0, 0)
    main(
        container, dataset, outfile, parse_subvolume(subvolume), resolution=resolution, offset=offset  # parsed.force
    )

    # Downloading synapse locations based on catmaid annotations.
    # We center the synapse locations in the downloaded raw such that
    # z-5:z+5,y-256:y+256,x-256:x+256. Each synapse location is downloaded as a separate zarr file.

    # container = "/media/samia/DATA/mounts/zstore1/registration-Albert/n5"
    # dataset = 's0'
    # resolution = (16, 8, 8)
    # offset = (0, 0, 0)
    #
    # df = pd.read_csv("/media/samia/DATA/ark/dan-samia/lsd/funke/parker/csvs/connectors.csv")
    # df.columns = df.columns.str.replace(' ', '')
    #
    # # Calculate subvolume dynamically for each row
    # subvolumes = []
    # for index, row in df.iterrows():
    #     subvolume = f"{int(row['z'] / resolution[0]) - 5}:{int(row['z'] / resolution[0]) +5}," \
    #                 f"{int(row['y'] / resolution[1]) - 256}:{int(row['y'] / resolution[1]) + 256}," \
    #                 f"{int(row['x'] / resolution[2]) - 256}:{int(row['x'] / resolution[2]) + 256}"
    #
    #     subvolumes.append(subvolume)
    #
    # # Add the subvolume column to the DataFrame
    # df['subvolume'] = subvolumes
    #
    # print(df[['connector_id', 'subvolume']])
    #
    # for i, row in df.iterrows():
    #     subvolume = row["subvolume"]
    #     ranges = subvolume.split(',')
    #     formatted_suffix = "z{}_y{}_x{}".format(
    #         ranges[0].replace(':', '-'),  # Format the first range
    #         ranges[1].replace(':', '-'),  # Format the second range
    #         ranges[2].replace(':', '-')  # Format the third range
    #     )
    #     outfile = f"/media/samia/DATA/ark/dan-samia/lsd/funke/parker/zarr/synapse-subvols/parker_KC_syn_{formatted_suffix}.zarr"
    #     main(
    #         container, dataset, outfile, parse_subvolume(subvolume), resolution=resolution, offset=offset  # parsed.force
    #     )


if __name__ == "__main__":
    _main()

    # OTTO slice - 7392:7904, 6586:7098, 5388:5900

    # this is what you need:
    # import zarr

    # store = N5FSStore(
    #     '/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/G2019S/forSamia_x8000_9023_y7000_8023_z650_1145.n5',
    #     'volumes/raw/c0/s0')
    # from zarr.n5 import N5FSStore
    #
    # store = N5FSStore(
    #     '/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/G2019S/forSamia_x8000_9023_y7000_8023_z650_1145.n5',
    #     'volumes/raw/c0/s0')
    # arr = zarr.Array(store, dataset, True)
    # arr = zarr.Array(store, 'volumes/raw/c0/s0', True)
    # imprt
    # import dask.array
    #
    # d_arr = dask.array.from_zarr(arr)
    # dask.array.to_zarr(d_arr,
    #                    '/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/G2019S/forSamia_x8000_9023_y7000_8023_z650_1145.zarr',
    #                    component='volumes/raw', overwrite=True)
    # outfile = '/mnt/wwn-0x5000c500e0dbd55e/ark/dan-samia/lsd/funke/G2019S/forSamia_x8000_9023_y7000_8023_z650_1145.zarr'
    # f = zarr.open(outfile, 'a')
    # f['volumes/raw'].attrs['offset'] = (0, 0, 0)
    # f['volumes/raw'].attrs['resolution'] = (8, 8, 8)
