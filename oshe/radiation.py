import os
import pathlib
import sys
import tempfile

import pandas as pd
from honeybee.hbsurface import HBSurface
from honeybee.radiance.analysisgrid import AnalysisGrid
from honeybee.radiance.material.glass import Glass
from honeybee.radiance.material.plastic import Plastic
from honeybee.radiance.properties import RadianceProperties
from honeybee.radiance.recipe.annual.gridbased import GridBased
from honeybee.radiance.sky.skymatrix import SkyMatrix

from .helpers import load_radiance_results

class _HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def open_field_radiation(epw_file: str, ground_reflectance: float, case_name: str = None, output_directory: str = None, shaded: bool = False):
    """ Run Radiance for a point in an open-field (with or without shade)

    Parameters
    ----------
    epw_file : str
        Path to EPW file
    ground_reflectance : float
        Reflectivity of ground in open field
    case_name : str
        Name of case being simulated
    output_directory : str
        Directory where simulation results will be stored
    shaded : bool
        True if simulated point is shaded. Shade properties are 10% transmissive to allow for reflections from nearby surfaaces as shade object is an enclosure

    Returns
    -------
    [direct_radiation, diffuse_radiation] : float
        Annual hourly direct and diffuse radiation

    """
    # Construct case output directory and process variables
    case_name = "open_field" if case_name is None else case_name
    output_directory = pathlib.Path(tempfile.gettempdir()) if output_directory is None else pathlib.Path(output_directory)
    epw_file = pathlib.Path(epw_file)

    # Create ground and shade (if included) context geometry and materials
    context_geometry = []

    ground_material = Plastic("ground", r_reflectance=ground_reflectance, g_reflectance=ground_reflectance, b_reflectance=ground_reflectance)
    ground_properties = RadianceProperties(material=ground_material)
    ground_surface = HBSurface(name="ground", sorted_points=[[-100, -100, 0], [-100, 100, 0], [100, 100, 0], [100, -100, 0]], surface_type=2, rad_properties=ground_properties)
    context_geometry.append(ground_surface)

    if shaded:
        shade_material = Glass(name="shade", r_transmittance=0.1, g_transmittance=0.1, b_transmittance=0.1)
        shade_properties = RadianceProperties(material=shade_material)
        shade_surfaces = [
            HBSurface(name="shade_side_0", sorted_points=[[-100, -100, 0], [-100, 100, 0], [0, 0, 3]], surface_type=6, rad_properties=shade_properties),
            HBSurface(name="shade_side_1", sorted_points=[[-100, 100, 0], [100, 100, 0], [0, 0, 3]], surface_type=6, rad_properties=shade_properties),
            HBSurface(name="shade_side_2", sorted_points=[[100, 100, 0], [100, -100, 0], [0, 0, 3]], surface_type=6, rad_properties=shade_properties),
            HBSurface(name="shade_side_3", sorted_points=[[100, -100, 0], [-100, -100, 0], [0, 0, 3]], surface_type=6, rad_properties=shade_properties),
        ]
        for shd_srf in shade_surfaces:
            context_geometry.append(shd_srf)

    with _HiddenPrints():
        # Prepare Radiance case for radiation incident on exposed test-point
        sky_matrix = SkyMatrix.from_epw_file(epw_file)
        analysis_grid = AnalysisGrid.from_points_and_vectors([[0, 0, 1.2]], name="openfield")
        recipe = GridBased(sky_mtx=sky_matrix, analysis_grids=[analysis_grid], simulation_type=1, hb_objects=context_geometry, reuse_daylight_mtx=True)

        # Run annual irradiance simulation
        command_file = recipe.write(target_folder=output_directory, project_name=case_name)
        recipe.run(command_file=command_file)
    print("Direct and diffuse solar radiation simulation completed")

    # Read Radiance results
    radiation_direct, radiation_diffuse = load_radiance_results(output_directory / case_name / "gridbased_annual" / "result")

    return radiation_direct, radiation_diffuse
