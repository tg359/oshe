import os
import pathlib
import sys
import tempfile

from honeybee.radiance.analysisgrid import AnalysisGrid
from honeybee.radiance.recipe.annual.gridbased import GridBased
from honeybee.radiance.sky.skymatrix import SkyMatrix

from .geometry import Ground, Shade
from .helpers import flatten
from .helpers import load_radiance_results


class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def run_radiance(epw_file: str, ground: Ground, shades: Shade = None, case_name: str = "openfield.py",
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
        analysis_grid = AnalysisGrid.from_points_and_vectors([[0, 0, 1.2]], name="openfield.py")
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
