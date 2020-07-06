import json
import pathlib
import shutil
import uuid

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath
from ladybug.psychrometrics import wet_bulb_from_db_rh, humid_ratio_from_db_rh, enthalpy_from_db_hr


ANNUAL_DATETIME = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")

MASKS = {
    "Daily": ((ANNUAL_DATETIME.hour >= 0) & (ANNUAL_DATETIME.hour <= 24)),
    "Morning": ((ANNUAL_DATETIME.hour >= 5) & (ANNUAL_DATETIME.hour <= 10)),
    "Midday": ((ANNUAL_DATETIME.hour >= 11) & (ANNUAL_DATETIME.hour <= 13)),
    "Afternoon": ((ANNUAL_DATETIME.hour >= 14) & (ANNUAL_DATETIME.hour <= 18)),
    "Evening": ((ANNUAL_DATETIME.hour >= 19) & (ANNUAL_DATETIME.hour <= 22)),
    "Night": ((ANNUAL_DATETIME.hour >= 23) | (ANNUAL_DATETIME.hour <= 4)),
    "MorningShoulder": ((ANNUAL_DATETIME.hour >= 7) & (ANNUAL_DATETIME.hour <= 10)),
    "AfternoonShoulder": ((ANNUAL_DATETIME.hour >= 16) & (ANNUAL_DATETIME.hour <= 19)),

    "Annual": ((ANNUAL_DATETIME.month >= 1) & (ANNUAL_DATETIME.month <= 12)),
    "Spring": ((ANNUAL_DATETIME.month >= 3) & (ANNUAL_DATETIME.month <= 5)),
    "Summer": ((ANNUAL_DATETIME.month >= 6) & (ANNUAL_DATETIME.month <= 8)),
    "Autumn": ((ANNUAL_DATETIME.month >= 9) & (ANNUAL_DATETIME.month <= 11)),
    "Winter": ((ANNUAL_DATETIME.month <= 2) | (ANNUAL_DATETIME.month >= 12)),
    "Shoulder": ((ANNUAL_DATETIME.month == 3) | (ANNUAL_DATETIME.month == 10)),
}

def load_weather(epw_file: str, index: pd.DatetimeIndex = ANNUAL_DATETIME):
    epw = EPW(epw_file)
    df = pd.DataFrame(index=index)

    df["dbt"] = np.roll(np.array(epw.dry_bulb_temperature.values), -1)
    df["rh"] = np.roll(np.array(epw.relative_humidity.values), -1)
    df["ws"] = np.roll(np.array(epw.wind_speed.values), -1)
    df["hir"] = np.roll(np.array(epw.horizontal_infrared_radiation_intensity.values), -1)

    df["dry_bulb_temperature"] = np.roll(np.array(epw.dry_bulb_temperature.values), -1)
    df["dew_point_temperature"] = np.roll(np.array(epw.dew_point_temperature.values), -1)
    df["relative_humidity"] = np.roll(np.array(epw.relative_humidity.values), -1)
    df["atmospheric_station_pressure"] = np.roll(np.array(epw.atmospheric_station_pressure.values), -1)
    df["extraterrestrial_horizontal_radiation"] = np.roll(np.array(epw.extraterrestrial_horizontal_radiation.values), -1)
    df["extraterrestrial_direct_normal_radiation"] = np.roll(np.array(epw.extraterrestrial_direct_normal_radiation.values), -1)
    df["horizontal_infrared_radiation_intensity"] = np.roll(np.array(epw.horizontal_infrared_radiation_intensity.values), -1)
    df["global_horizontal_radiation"] = np.roll(np.array(epw.global_horizontal_radiation.values), -1)
    df["direct_normal_radiation"] = np.roll(np.array(epw.direct_normal_radiation.values), -1)
    df["diffuse_horizontal_radiation"] = np.roll(np.array(epw.diffuse_horizontal_radiation.values), -1)
    df["global_horizontal_illuminance"] = np.roll(np.array(epw.global_horizontal_illuminance.values), -1)
    df["direct_normal_illuminance"] = np.roll(np.array(epw.direct_normal_illuminance.values), -1)
    df["diffuse_horizontal_illuminance"] = np.roll(np.array(epw.diffuse_horizontal_illuminance.values), -1)
    df["zenith_luminance"] = np.roll(np.array(epw.zenith_luminance.values), -1)
    df["wind_direction"] = np.roll(np.array(epw.wind_direction.values), -1)
    df["wind_speed"] = np.roll(np.array(epw.wind_speed.values), -1)
    df["total_sky_cover"] = np.roll(np.array(epw.total_sky_cover.values), -1)
    df["opaque_sky_cover"] = np.roll(np.array(epw.opaque_sky_cover.values), -1)
    df["visibility"] = np.roll(np.array(epw.visibility.values), -1)
    df["ceiling_height"] = np.roll(np.array(epw.ceiling_height.values), -1)
    df["present_weather_observation"] = np.roll(np.array(epw.present_weather_observation.values), -1)
    df["present_weather_codes"] = np.roll(np.array(epw.present_weather_codes.values), -1)
    df["precipitable_water"] = np.roll(np.array(epw.precipitable_water.values), -1)
    df["aerosol_optical_depth"] = np.roll(np.array(epw.aerosol_optical_depth.values), -1)
    df["snow_depth"] = np.roll(np.array(epw.snow_depth.values), -1)
    df["days_since_last_snowfall"] = np.roll(np.array(epw.days_since_last_snowfall.values), -1)
    df["albedo"] = np.roll(np.array(epw.albedo.values), -1)
    df["liquid_precipitation_depth"] = np.roll(np.array(epw.liquid_precipitation_depth.values), -1)
    df["liquid_precipitation_quantity"] = np.roll(np.array(epw.liquid_precipitation_quantity.values), -1)

    df["wet_bulb_temperature"] = np.vectorize(wet_bulb_from_db_rh)(df.dry_bulb_temperature.values, df.relative_humidity.values, df.atmospheric_station_pressure.values)
    df["humidity_ratio"] = np.vectorize(humid_ratio_from_db_rh)(df.dry_bulb_temperature.values, df.relative_humidity.values, df.atmospheric_station_pressure.values)
    df["enthalpy"] = np.vectorize(enthalpy_from_db_hr)(df.dry_bulb_temperature.values, df.humidity_ratio.values)

    sun_path = Sunpath.from_location(epw.location)
    df["sun_altitude"] = np.array([sun_path.calculate_sun_from_hoy(i).altitude for i in range(8760)])

    df["city"] = epw.location.city
    df["country"] = epw.location.country
    df["station_id"] = epw.location.station_id
    return df


def random_id():
    """ Create a random ID number

    Returns
    -------
    random_id : str
    """
    return str(uuid.uuid4()).replace("-", "")[:10]


def flatten(nested_list):
    if nested_list == []:
        return nested_list
    if isinstance(nested_list[0], list):
        return flatten(nested_list[0]) + flatten(nested_list[1:])
    return nested_list[:1] + flatten(nested_list[1:])


def load_json(json_file: str):
    """ Load a JSON file into a dictionary

    Parameters
    ----------
    json_file : str
        Path to JSON

    Returns
    -------
    data : dict
        JSON contents as dictionary
    """
    with open(json_file) as f:
        data = json.load(f)
    return data


def load_radiance_results(directory: str):
    """ Load the ILL files from a Radiance simulation output (simulated using the Honeybee workflow and Radiance 5.2)

    Parameters
    ----------
    directory : str
        Directory in which the simulation results are located

    Returns
    -------
        [radiation_direct, radiation_diffuse] : float
            Array of direct and diffuse radiation incident at each point simulated
    """

    def coerce_float(val):
        try:
            return np.float64(val)
        except Exception as e:
            return np.nan

    directory = pathlib.Path(directory)

    radiation_total = pd.read_csv(directory / "total..scene..default.ill", sep="\t", converters={0: coerce_float},
                                  names=range(8760)).dropna().reset_index(drop=True).T / 179
    radiation_scene = pd.read_csv(directory / "direct..scene..default.ill", sep="\t", converters={0: coerce_float},
                                  names=range(8760)).dropna().reset_index(drop=True).T / 179
    radiation_direct = (pd.read_csv(directory / "sun..scene..default.ill", sep="\t", converters={0: coerce_float},
                                    names=range(8760)).dropna().reset_index(drop=True).T / 179).values

    radiation_diffuse = (radiation_total - radiation_scene).values

    print("Radiance results loaded")
    return radiation_direct, radiation_diffuse


def load_energyplus_results(file_path: str):
    """ Load a CSV file from an EnergyPlus simulation

    Parameters
    ----------
    file_path : str
        Path to EnergyPlus simulation output file
    Returns
    -------
    surface_temperatures : float
        Surface temperature array (each element is a single surface)
    """
    data = pd.read_csv(file_path, index_col=0, parse_dates=True).dropna(axis=1).values.T
    print("EnergyPlus results loaded")
    return data


def load_points_xyz(file_path: str):
    """ Load a PTS file containing simulation analysis points (usually from a Radiance case)

    Parameters
    ----------
    file_path : str
        Path to EnergyPlus simulation output file
    Returns
    -------
    points : float
        Surface temperature array (each element is a single surface)
    """
    return pd.DataFrame(file_path, columns=["x", "y", "z"]).values


def chunk(enumerable, n=None):
    enumerable = np.array(enumerable)
    return [enumerable[i:i + n] for i in range(0, enumerable.shape[0], n)]


def chunks(enumerable, n=None):
    enumerable = np.array(enumerable)
    return [i for i in np.array_split(enumerable, n)]


def create_directory(directory: str):
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return pathlib.Path(directory)


def find_files(directory: str, endswith: str):
    return list(pathlib.Path(directory).glob('**/*{}'.format(endswith)))


def convert_recipe_to_composite(recipe_json: str, clean: bool = False):
    # load file
    json_file = pathlib.Path(recipe_json)
    d = load_json(json_file)

    # Get output directory
    outdir = json_file.parents[0]

    # Write sample_pts to file
    points = pd.DataFrame(d["points"])
    points.to_csv(outdir / "_sample_points.xyz", index=False, header=False)

    # Write boundary_pts to file
    boundary = pd.DataFrame(np.array(d["boundary"])[:, :-1])
    boundary.to_csv(outdir / "_boundary_points.xyz", index=False, header=False)

    # Write gnd reflectivities to file
    ground_reflectivities = pd.Series(d["ground_reflectivities"])
    ground_reflectivities.to_csv(outdir / "_ground_reflectivities.rfl", index=False, header=False)

    # Write sky view factor to file
    sky_view_factors = pd.Series(d["sky_view_factors"]) / 100
    sky_view_factors.to_csv(outdir / "_sky_view_factors.vf", index=False, header=False)

    # Write surface view factors to file
    surface_view_factors = pd.DataFrame(d["surface_view_factors"])
    surface_view_factors.to_csv(outdir / "_surface_view_factors.vf", index=False, header=False)

    # Remove the recipe json
    if clean:
        json_file.rename(json_file.parent / json_file.name.replace(".", "_OLD."))

    return None


def nukedir(directory: str):
    shutil.rmtree(directory, ignore_errors=True, onerror=None)
    return None
