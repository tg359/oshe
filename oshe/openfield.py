
from .material import material_dict
from .geometry import Ground, Shade
from .energyplus import run_energyplus
from .radiance import run_radiance
from .mrt import mean_radiant_temperature
from .utci import universal_thermal_climate_index
from .helpers import ANNUAL_DATETIME, load_weather

import pandas as pd

def openfield(epw_file: str, idd_file: str, material: str = "CONCRETE", shaded: bool = False):

    # Load weatherfile
    epw = load_weather(epw_file=epw_file)

    # Define ground material
    gnd_mat = material_dict[material]
    ground_zone = Ground(gnd_mat, xy=5, depth=1.5, subsurface_size=5)

    # Define VERY LARGE horizontal shade
    if shaded:
        # Single shade
        shades = [
            Shade(vertices=[[-20, -20, 3], [-20, 20, 0], [20, 20, 0], [20, -20, 3]])
        ]
        
        # Pyramid shade
        # shades = [
        #     Shade(vertices=[[2.5, -2.5, 0], [2.5, 2.5, 0], [0, 0, 3]]),
        #     Shade(vertices=[[2.5, 2.5, 0], [-2.5, 2.5, 0], [0, 0, 3]]),
        #     Shade(vertices=[[-2.5, 2.5, 0], [-2.5, -2.5, 0], [0, 0, 3]]),
        #     Shade(vertices=[[-2.5, -2.5, 0], [2.5, -2.5, 0], [0, 0, 3]])
        # ]
        
        # Box shade
        # shades = [
        #     Shade(vertices=[[5, 5, 3], [5, 5, 0], [5, -5, 0], [5, -5, 3]]),
        #     Shade(vertices=[[5, -5, 3], [5, -5, 0], [-5, -5, 0], [-5, -5, 3]]),
        #     Shade(vertices=[[-5, -5, 3], [-5, -5, 0], [-5, 5, 0], [-5, 5, 3]]),
        #     Shade(vertices=[[-5, 5, 3], [-5, 5, 0], [5, 5, 0], [5, 5, 3]]),
        #     Shade(vertices=[[-5, -5, 3], [-5, 5, 3], [5, 5, 3], [5, -5, 3]]),
        # ]
    else:
        shades = None

    # Calculate ground surface temperature
    of_srf_temp = run_energyplus(epw_file, idd_file, ground=ground_zone, shades=shades, case_name="shaded" if shaded else "unshaded", run=True).values

    # Calculate incident solar direct and diffuse radiation
    of_dir_rad, of_dif_rad = run_radiance(epw_file, ground=ground_zone, shades=shades, case_name="shaded" if shaded else "unshaded", run=True)

    # Calculate MRT
    of_mrt = mean_radiant_temperature(
        surrounding_surfaces_temperature=of_srf_temp[0],
        horizontal_infrared_radiation_intensity=0 if shaded else epw.hir.values,
        diffuse_horizontal_solar=of_dif_rad.T[0],
        direct_normal_solar=of_dir_rad.T[0],
        sun_altitude=epw.sun_altitude.values,
        ground_reflectivity=gnd_mat.reflectivity,
        sky_exposure=0 if shaded else 1)[0]  # TODO - replace exposure with TRUE exposure value

    # Calculate UTCI
    of_utci = universal_thermal_climate_index(epw.dbt.values, of_mrt, epw.ws.values, epw.rh.values)

    # Join important results
    d = {
        "hir": epw.hir.values * 0 if shaded else epw.hir.values * 1,
        "dbt": epw.dbt.values,
        "dir": of_dir_rad.T[0],
        "dif": of_dif_rad.T[0],
        "gnd_srftemp": of_srf_temp[0],
        "mrt": of_mrt,
        "utci": of_utci
    }

    return pd.DataFrame.from_dict(d).set_index(ANNUAL_DATETIME)