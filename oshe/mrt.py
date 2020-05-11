from datetime import datetime
from multiprocessing import Pool

import numpy as np

from oshe.oshe import oshe
from .helpers import chunks


def mean_radiant_temperature(surrounding_surfaces_temperature: float, horizontal_infrared_radiation_intensity: float,
                             diffuse_horizontal_solar: float,
                             direct_normal_solar: float, sun_altitude: float, ground_reflectivity: float = 0.25,
                             sky_exposure: float = 1, radiance: bool = False):
    """ Perform a full outdoor sky and context surface radiant heat exchange.

    This method is a vectorised version of the ladybug-comfort outdoor_sky_heat_exch method https://www.ladybug.tools/ladybug-comfort/docs/_modules/ladybug_comfort/solarcal.html#outdoor_sky_heat_exch

    Parameters
    ----------
    surrounding_surfaces_temperature : float
        The temperature of surfaces around the person in degrees Celcius. This includes the ground and any other surfaces blocking the view to the sky. When the temperature of these individual surfaces are known, the input here should be the average temperature of the surfaces weighted by view-factor to the human. When such individual surface temperatures are unknown, the outdoor dry bulb temperature is typically used as a proxy.
    horizontal_infrared_radiation_intensity : float
        The horizontal infrared radiation intensity from the sky in W/m2.
    diffuse_horizontal_solar : float
        Diffuse horizontal solar irradiance in W/m2.
    direct_normal_solar : float
        Direct normal solar irradiance in W/m2.
    sun_altitude : float
        The altitude of the sun in degrees [0-90].
    ground_reflectivity : float
        A number between 0 and 1 the represents the reflectance of the floor. Default is for 0.25 which is characteristic of outdoor grass or dry bare soil.
    sky_exposure : float
        A number between 0 and 1 representing the fraction of the sky vault in occupant’s view. Default is 1 for outdoors in an open field.
    radiance : bool
        True if diffuse and direct radiation from a Radiance simulation

    Returns
    -------
    mean_radiant_temperature : float
        The final MRT experienced as a result of sky heat exchange in C.
    """

    def get_mrt(z):
        return z["mrt"]

    start_time = datetime.now()

    mrt = oshe(
        visible_surfaces_temperature=np.array([surrounding_surfaces_temperature]),
        horizontal_infrared_radiation=np.array([horizontal_infrared_radiation_intensity]),
        diffuse_horizontal_solar_radiation=np.array([diffuse_horizontal_solar]),
        direct_normal_solar_radiation=np.array([direct_normal_solar]),
        solar_altitude=np.array([sun_altitude]),
        sky_exposure=np.array([sky_exposure]),
        floor_reflectance=np.array([ground_reflectivity]),
        radiance=radiance
    )

    td = (datetime.now() - start_time).total_seconds()
    print("Mean radiant temperature calculated [{}]".format('{0:0.02f} seconds'.format(td)))
    return mrt


def mrt_parallel_int(val_dict):
    """ Dummy process to enable passing of dictionary to mean radiant temperature method for parallel processing

    Parameters
    ----------
    val_dict

    Returns
    -------

    """
    mean_radiant_temperature_part = mean_radiant_temperature(
        surrounding_surfaces_temperature=val_dict["srftmp"],
        horizontal_infrared_radiation_intensity=val_dict["hrzifr"],
        diffuse_horizontal_solar=val_dict["difrad"],
        direct_normal_solar=val_dict["dirrad"],
        sun_altitude=val_dict["solalt"],
        ground_reflectivity=val_dict["gndref"],
        sky_exposure=val_dict["skyexp"],
        radiance=val_dict["radrun"]
    )[0]
    print("Thread #{0:} complete".format(val_dict["thread"]))
    return mean_radiant_temperature_part


def mrt_parallel(threads: int, surrounding_surfaces_temperature: float, horizontal_infrared_radiation_intensity: float,
                 diffuse_horizontal_solar: float, direct_normal_solar: float, sun_altitude: float,
                 ground_reflectivity: float = 0.25, sky_exposure: float = 1, radiance: bool = False):
    """ Run the MRT calculation across multiple processors concurrently (useful for REALLY BIG CASES)

    Perform a full outdoor sky and context surface radiant heat exchange.

    This method is a vectorised version of the ladybug-comfort outdoor_sky_heat_exch method https://www.ladybug.tools/ladybug-comfort/docs/_modules/ladybug_comfort/solarcal.html#outdoor_sky_heat_exch

    Parameters
    ----------
    threads : int
        Number of threads to split calculation across
    surrounding_surfaces_temperature : float
        The temperature of surfaces around the person in degrees Celcius. This includes the ground and any other surfaces blocking the view to the sky. When the temperature of these individual surfaces are known, the input here should be the average temperature of the surfaces weighted by view-factor to the human. When such individual surface temperatures are unknown, the outdoor dry bulb temperature is typically used as a proxy.
    horizontal_infrared_radiation_intensity : float
        The horizontal infrared radiation intensity from the sky in W/m2.
    diffuse_horizontal_solar : float
        Diffuse horizontal solar irradiance in W/m2.
    direct_normal_solar : float
        Direct normal solar irradiance in W/m2.
    sun_altitude : float
        The altitude of the sun in degrees [0-90].
    ground_reflectivity : float
        A number between 0 and 1 the represents the reflectance of the floor. Default is for 0.25 which is characteristic of outdoor grass or dry bare soil.
    sky_exposure : float
        A number between 0 and 1 representing the fraction of the sky vault in occupant’s view. Default is 1 for outdoors in an open field.
    radiance : bool
        True if diffuse and direct radiation from a Radiance simulation

    Returns
    -------
    mean_radiant_temperature : float
        The final MRT experienced as a result of sky heat exchange in C.

    """

    dicts = []
    for i in range(threads):
        dicts.append({
            "srftmp": chunks(surrounding_surfaces_temperature, threads)[i],
            "hrzifr": horizontal_infrared_radiation_intensity,
            "difrad": chunks(diffuse_horizontal_solar.T, threads)[i],
            "dirrad": chunks(direct_normal_solar.T, threads)[i],
            "solalt": sun_altitude,
            "gndref": chunks(ground_reflectivity, threads)[i] * 0.5,
            "skyexp": chunks(sky_exposure, threads)[i],
            "radrun": radiance,
            "thread": i
        })

    # if __name__ ==  '__main__':
    p = Pool(processes=threads)
    output = p.map(mrt_parallel_int, dicts)
    p.close()

    # Restack results
    mean_radiant_temperature_result = np.vstack(output)

    return mean_radiant_temperature_result
