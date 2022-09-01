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
import zarr
from zarr.n5 import N5FSStore


def parse_subvolume(s):
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
    p.add_argument("zarr", type=int, help="Pass 1 to convert n5 to zarr, default converts raw to volumes/raw dataset")
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


def to_zarr(z, outfile):
    z = dask.array.rechunk(z, outfile)

    dask.array.to_zarr(z, outfile, component='volumes/raw', overwrite=True)
    f = zarr.open(outfile, 'a')
    f['volumes/raw'].attrs['offset'] = (0, 0, 0)
    f['volumes/raw'].attrs['resolution'] = (8, 8, 8)


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
    parsed=None
):
    z = get_input(container, dataset)
    # Be careful to think of the orientation of the volume, meaning pass zyx if the n5 is oriented as zyx.
    if subvolume:
        z = z[subvolume]

    if int(parsed.zarr):
        to_zarr(z, outfile)

    if int(parsed.mrc):
        with get_output(outfile, z.shape, z.dtype, force) as out:
            # TODO: consider locking
            dask.array.store(z, out.data)


def _main(args=None):
    parsed = parse_args(args)
    # parsed.container = '/mnt/wwn-0x5000c500e0dbd55e/zstore1'
    # parsed.dataset = 'FIBSEM_L1120_FullCNS_8x8x8nm/s0'
    # parsed.outfile = '/mnt/wwn-0x5000c500e0dbd55e/n5mrc/mynew2.zarr'
    # parsed.subvolume = "20000:20776,11000:11776,11000:11776"

    main(
        parsed.container, parsed.dataset, parsed.outfile, parsed.subvolume, parsed.force
    )


if __name__ == "__main__":
    _main()

    # OTTO slice - 7392:7904, 6586:7098, 5388:5900
