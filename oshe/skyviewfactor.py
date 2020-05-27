import subprocess
from typing import List
import tempfile
import pathlib
from .helpers import find_files


def sky_view_factor(pts_file: str, rad_files: List[str], case_name: str = "svf", output_directory: str = pathlib.Path(tempfile.gettempdir())):
    """
    # Calculate Sky view Factor from a set of Radiance geometry and sample point locations
    Method here from https://www.radiance-online.org/pipermail/radiance-general/2018-April/012481.html

    Parameters
    ----------
    pts_file
    rad_files
    case_name
    output_directory
    run

    Returns
    -------

    """
    # Create output directory if it doesn't exist
    output_directory = pathlib.Path(output_directory) / case_name
    output_directory.mkdir(parents=True, exist_ok=True)

    # Create file paths
    oct_file = output_directory / "scene.oct"
    sky_file = output_directory / "sky.rad"
    skyviewfactor_file = output_directory / "sky_view_factor.dat"

    # Create the sky file
    with open(sky_file, "w") as f:
        f.write("void glow sky_glow\n0\n0\n4 0.318309886 0.318309886 0.318309886 0\nsky_glow source sky\n0\n0\n4 0 0 1 180")

    # Create the octree file
    octree_cmd = "oconv {0:} {1:} > {2:}".format(str(sky_file), " ".join(rad_files), str(oct_file))
    # print(octree_cmd)
    subprocess.call(octree_cmd, shell=True)

    # Calculate the sky view factor for each point in the context geometry
    svf_cmd = "type {0:} | rtrace -h -ab 1 -I {1:} | rcalc -e \"SVF=$1;$1=SVF\" > {2:}".format(str(pts_file), str(oct_file), str(skyviewfactor_file))
    # print(svf_cmd)
    subprocess.call(svf_cmd, shell=True)

    # Load svfs into array
    with open(skyviewfactor_file, "r") as f:
        data = [float(i.strip()) for i in f.readlines()]

    return data

def case_svf(case_directory: str):
    case_directory = pathlib.Path(case_directory)
    svf_file = case_directory.parents[0] / "_sky_view_factors.vf"
    pts_file = find_files(case_directory, endswith=".pts")[0]
    rad_files = []
    for j in ["mat", "rad"]:
        for i in find_files(case_directory / "scene", endswith=j):
            rad_files.append(str(i))
    svfs = sky_view_factor(pts_file=str(pts_file), rad_files=rad_files)
    # print(svf_file)
    with open(svf_file, "w") as f:
        f.writelines(["{0:}".format(i) for i in svfs])
    return svfs
