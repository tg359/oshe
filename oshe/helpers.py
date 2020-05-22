import json
import pathlib
import uuid

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath


ANNUAL_DATETIME = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")


def load_weather(epw_file: str):
    epw = EPW(epw_file)
    df = pd.DataFrame(index=ANNUAL_DATETIME)
    df["dbt"] = np.roll(np.array(epw.dry_bulb_temperature.values), -1)
    df["rh"] = np.roll(np.array(epw.relative_humidity.values), -1)
    df["ws"] = np.roll(np.array(epw.wind_speed.values), -1)
    df["hir"] = np.roll(np.array(epw.horizontal_infrared_radiation_intensity.values), -1)

    sun_path = Sunpath.from_location(epw.location)
    df["sun_altitude"] = np.array([sun_path.calculate_sun_from_hoy(i).altitude for i in range(8760)])
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
