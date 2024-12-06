"""
A basic .swc viewer that plots a `directory` of skeletons with matplotlib and saves a png.
NB: Hard-coded paths in the script.

Dependencies (versions - latest):
Kimimaro
cloud-volume
tqdm
matplotlib
numpy
fastremap
"""

import glob
import os

import cloudvolume
from cloudvolume import Skeleton
from cloudvolume.lib import mkdir
import click
import numpy as np

import kimimaro
import fastremap
from tqdm import tqdm


def viewer(
        skeletons,  # Pass a list of skeleton objects to be visualized
        units='nm',
        draw_edges=True, draw_vertices=True,
        color_by='radius'
):
    """
    View multiple skeletons in the same viewer with a radius heatmap.

    Requires the matplotlib library which is
    not installed by default.

    skeletons: list of skeleton objects to be visualized.
    units: label axes with these units
    draw_edges: draw lines between vertices (more useful when skeleton is sparse)
    draw_vertices: draw each vertex colored by its radius.
    color_by:
      'radius': color each vertex according to its radius attribute
        aliases: 'r', 'radius', 'radii'
      'component': color connected components separately
        aliases: 'c', 'component', 'components'
      'cross_section': color each vertex according to its cross sectional area
        aliases: 'x'
      anything else: draw everything black
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        print("Skeleton.viewer requires matplotlib. Try: pip install matplotlib --upgrade")
        return

    RADII_KEYWORDS = ('radius', 'radii', 'r')
    CROSS_SECTION_KEYWORDS = ('cross_section', 'x')
    COMPONENT_KEYWORDS = ('component', 'components', 'c')

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(units)
    ax.set_ylabel(units)
    ax.set_zlabel(units)

    # Set plot axes equal for all skeletons together
    all_X = np.concatenate([skel.vertices[:, 0] for skel in skeletons])
    all_Y = np.concatenate([skel.vertices[:, 1] for skel in skeletons])
    all_Z = np.concatenate([skel.vertices[:, 2] for skel in skeletons])

    max_range = np.array([all_X.max() - all_X.min(), all_Y.max() - all_Y.min(), all_Z.max() - all_Z.min()]).max() / 2.0

    mid_x = (all_X.max() + all_X.min()) * 0.5
    mid_y = (all_Y.max() + all_Y.min()) * 0.5
    mid_z = (all_Z.max() + all_Z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    component_colors = ['k', 'deeppink', 'dodgerblue', 'mediumaquamarine', 'gold']

    def draw_component(i, skel):
        nonlocal units
        component_color = component_colors[i % len(component_colors)]

        if draw_vertices:
            xs = skel.vertices[:, 0]
            ys = skel.vertices[:, 1]
            zs = skel.vertices[:, 2]

            if color_by in RADII_KEYWORDS or color_by in CROSS_SECTION_KEYWORDS:
                colmap = cm.ScalarMappable(cmap=cm.get_cmap('rainbow'))

                axis_label = ''
                if color_by in RADII_KEYWORDS:
                    axis_label = 'radius'
                    colmap.set_array(skel.radii)
                    normed_data = skel.radii / np.max(skel.radii)
                else:
                    axis_label = 'cross sectional area'
                    units += '^2'
                    colmap.set_array(skel.cross_sectional_area)
                    normed_data = skel.cross_sectional_area / np.max(skel.cross_sectional_area)

                ax.scatter(xs, ys, zs, c=cm.rainbow(normed_data), marker='o')
                # cbar = fig.colorbar(colmap, ax=ax)
                # cbar.set_label(f'{axis_label} ({units})', rotation=270)
            elif color_by in COMPONENT_KEYWORDS:
                ax.scatter(xs, ys, zs, color=component_color, marker='.')
            else:
                ax.scatter(xs, ys, zs, color='k', marker='.')

        if draw_edges:
            for e1, e2 in skel.edges:
                pt1, pt2 = skel.vertices[e1], skel.vertices[e2]
                ax.plot(
                    [pt1[0], pt2[0]],
                    [pt1[1], pt2[1]],
                    zs=[pt1[2], pt2[2]],
                    color=(component_color if not draw_vertices else 'silver'),
                    linewidth=1,
                )

    # Iterate over the list of skeletons and plot each one
    for i, skel in enumerate(skeletons):
        draw_component(i, skel)

    # plt.show()

    plt.savefig("/Users/sam/Downloads/plotswc.png", dpi=300)


def view(filename, port):
    """Visualize a .swc or .npy file."""
    # read the swc in a list
    skel_list = []

    if isinstance(filename, list):
        for f in filename:
            basename, ext = os.path.splitext(f)
            if ext == ".swc":
                with open(f, "rt") as swc:
                    skel = Skeleton.from_swc(swc.read())
                skel_list.append(skel)

    viewer(skel_list)


if __name__ == '__main__':
    swc_files = glob.glob("/Users/sam/Documents/random_codebases/ssTEM_data_Acardona/Seg2Link/data/skeletons/*.swc")[
                500:700]

    print(len(swc_files))

    port = 8080
    view(swc_files, port)
