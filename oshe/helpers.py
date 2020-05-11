import json
import pathlib
import uuid

import numpy as np
import pandas as pd
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath


def sun_altitude(epw_file):
    epw = EPW(epw_file)
    sun_path = Sunpath.from_location(epw.location)
    return np.array([sun_path.calculate_sun_from_hoy(i).altitude for i in range(8760)])


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
    """ Load the ILL files from a Radiance simulation output (simulated using the Honeybee workflow)

    Parameters
    ----------
    directory : str
        Directory in which the simulation results are located

    Returns
    -------
        [radiation_direct, radiaction_diffuse] : float
            Array of direct and diffuse radiation incident at each point simulated
    """
    directory = pathlib.Path(directory)
    radiation_total = pd.read_csv(directory / "total..scene..default.ill", skiprows=6, sep="\t",
                                  header=None, ).T.dropna() / 179
    radiation_scene = pd.read_csv(directory / "direct..scene..default.ill", skiprows=6, sep="\t",
                                  header=None, ).T.dropna() / 179
    radiation_diffuse = (radiation_total - radiation_scene).values
    radiation_direct = (pd.read_csv(directory / "sun..scene..default.ill", skiprows=6, sep="\t",
                                    header=None).T.dropna() / 179).values

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
