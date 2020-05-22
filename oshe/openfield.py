
from .material import material_dict
from .geometry import Ground, Shade
from .energyplus import run_energyplus
from .radiance import run_radiance
from .mrt import mean_radiant_temperature
from .utci import universal_thermal_climate_index
from .helpers import ANNUAL_DATETIME, load_weather

import pandas as pd

def openfield(epw_file: str, idd_file: str, material: str = "CONCRETE", shaded: bool = False):

    df = pd.DataFrame(index=ANNUAL_DATETIME)

    # Load weatherfile
    epw = load_weather(epw_file=epw_file)

    # Define ground material
    gnd_mat = material_dict["CONCRETE"]
    ground_zone = Ground(gnd_mat, xy=5, depth=1.5, subsurface_size=5)

    # Define VERY LARGE horizontal shade
    if shaded:
        shades = [
            Shade(vertices=[[-1000, -1000, 3], [-1000, 1000, 0], [1000, 1000, 0], [1000, -1000, 3]])
        ]
        # Box shade - currently not utilised
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
    of_srf_temp = run_energyplus(epw_file, idd_file, ground=ground_zone, shades=shades, run=True)

    # Calculate incident solar direct and diffuse radiation
    of_dir_rad, of_dif_rad = run_radiance(epw_file, ground=ground_zone, shades=None, run=True)

    # Calculate MRT
    of_mrt = mean_radiant_temperature(
        surrounding_surfaces_temperature=of_srf_temp[0],
        horizontal_infrared_radiation_intensity=epw.hir.values,
        diffuse_horizontal_solar=of_dif_rad.T[0],
        direct_normal_solar=of_dir_rad.T[0],
        sun_altitude=epw.sun_altitude.values,
        ground_reflectivity=gnd_mat.reflectivity,
        sky_exposure=0 if shaded else 1)[0]

    # Calculate UTCI
    of_utci = universal_thermal_climate_index(epw.dbt.values, of_mrt, epw.ws.values, epw.rh.values)

    # Join important results
    d = {
        "dbt": epw.dbt.values,
        "gnd_srftemp": of_srf_temp[0],
        "mrt": of_mrt,
        "utci": of_utci
    }

    return pd.DataFrame.from_dict(d).set_index(ANNUAL_DATETIME)