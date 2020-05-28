import os
import pathlib
import subprocess
import sys
import tempfile
from multiprocessing import Pool

from honeybee.radiance.analysisgrid import AnalysisGrid
from honeybee.radiance.recipe.annual.gridbased import GridBased
from honeybee.radiance.sky.skymatrix import SkyMatrix

from .geometry import Ground, Shade
from .helpers import flatten, chunks
from .helpers import load_radiance_results


class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def run_radiance(epw_file: str, ground: Ground, shades: Shade = None, case_name: str = "openfield",
                 output_directory: str = pathlib.Path(tempfile.gettempdir()), run: bool = False):
    """ Run Radiance for a point in an open-field (with or without shade)

    Parameters
    ----------
    epw_file : str
        Path to EPW file
    ground : Ground
        Ground object to simulate
    ground : Ground
        Ground object to simulate
    case_name : str
        Name of case being simulated
    output_directory : str
        Directory where simulation results will be stored
    run : bool
        Run the simulation

    Returns
    -------
    [direct_radiation, diffuse_radiation] : float
        Annual hourly direct and diffuse radiation

    """
    # Construct case output directory and process variables
    epw_file = pathlib.Path(epw_file)

    # Create ground and shade (if included) context geometry and materials
    context_geometry = []
    context_geometry.append(ground.to_hb())

    if shades is not None:
        for shd in shades:
            context_geometry.append(shd.to_hb())

    # Flatten objects
    context_geometry = flatten(context_geometry)

    with _HiddenPrints():

        # Prepare Radiance case for radiation incident on exposed test-point
        sky_matrix = SkyMatrix.from_epw_file(epw_file)
        analysis_grid = AnalysisGrid.from_points_and_vectors([[0, 0, 1.2]], name=case_name)
        recipe = GridBased(sky_mtx=sky_matrix, analysis_grids=[analysis_grid], simulation_type=1,
                           hb_objects=context_geometry, reuse_daylight_mtx=True)

    # Run annual irradiance simulation
    command_file = recipe.write(target_folder=output_directory, project_name=case_name)

    if run:
        with _HiddenPrints():
            recipe.run(command_file=command_file)

        print("Direct and diffuse solar radiation simulation completed")

        # Read Radiance results
        radiation_direct, radiation_diffuse = load_radiance_results(
            output_directory / case_name / "gridbased_annual" / "result")

        return radiation_direct, radiation_diffuse
    else:
        print("Radiance case written to {}".format(str(output_directory / case_name / "gridbased_annual")))
        return None


class HBRad(object):
    def __init__(self, source_directory, n_processes=1):
        self.source_directory = pathlib.Path(source_directory)
        self.source_pts = self.find_pts()
        self.n_processes = n_processes
        self.sub_pts = self.split_pts_file()
        self.bat_file = self.source_directory / "commands.bat"
        self.bat_init = None
        self.sub_bat_files = self.split_bat_file()

    def find_pts(self):
        return list(pathlib.Path(self.source_directory).glob('**/*{}'.format(".pts")))[0]

    def split_bat_file(self):
        # Get header and find line where calculation starts
        with open(self.bat_file, "r") as f:
            bat_commands = [i.strip() for i in f.readlines()]
        for n, i in enumerate(bat_commands):
            if i == "":
                header_idx = n
            if "start of the calculation for scene" in i:
                split_idx = n
            if " > analemma.oct" in i:
                analemma_idx = n

        # Put the initial smx commands into their own bat_init file
        self.bat_init = self.source_directory / "{0:}_init.bat".format(self.bat_file.stem)
        header = bat_commands[:split_idx] + [bat_commands[analemma_idx]]
        with open(self.bat_init, "w") as f:
            for i in header:
                if "echo" in i:
                    pass
                else:
                    f.write("{0:}\n".format(i))

        # Put the remainder of the bat commands into pts-set respective files
        sub_bat_files = []
        for i in range(self.n_processes):
            base_outstr = bat_commands[:header_idx] + bat_commands[split_idx:analemma_idx] + bat_commands[
                                                                                             analemma_idx + 1:]
            outstr = []
            for j in base_outstr:
                if "echo" in j:
                    pass
                else:
                    outstr.append(
                        j.replace(self.source_pts.name, "{0:}_{1:03d}.pts".format(self.source_pts.stem, i)).replace(".dc", "_{0:03d}.dc".format(i)).replace(".rgb", "_{0:03d}.rgb".format(i)).replace(".ill", "_{0:03d}.ill".format(i)))
            sub_bat_file = self.source_directory / "{0:}_{1:03d}.bat".format(self.bat_file.stem, i)
            with open(sub_bat_file, "w", newline='\n', encoding='utf-8') as f:
                f.write("\n".join(outstr) + "\n")
            sub_bat_files.append(sub_bat_file)
        return sub_bat_files

    def split_pts_file(self):
        sub_pts = []
        with open(self.source_pts, "r") as f:
            src_pts = [i.strip() for i in f.readlines()]
        src_pts_split = chunks(src_pts, self.n_processes)

        for i in range(self.n_processes):
            sub_pts_file = self.source_directory / "{0:}_{1:03d}.pts".format(self.source_pts.stem, i)
            with open(sub_pts_file, "w", newline='\n', encoding='utf-8') as f:
                f.write("\n".join(src_pts_split[i]) + "\n")

            sub_pts.append(sub_pts_file)
        return sub_pts

    def run(self):
        # Create the analemma OCTREE
        subprocess.call(str(self.bat_init), shell=True)

        # Run the composite bat files
        for bat in self.sub_bat_files:
            subprocess.call('{0:}'.format(str(bat)), shell=True)
            # subprocess.Popen(str(bat), shell=True)
            # subprocess.run([str(bat)], stdout=subprocess.PIPE)

    def __repr__(self):
        return "{klass}\n{attrs}".format(
            klass=self.__class__.__name__,
            attrs="\n".join("- {}: {!r}".format(k, v) for k, v in self.__dict__.items()),
        )
