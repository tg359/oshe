import numpy as np
from datetime import datetime
from oshe.oshe import oshe


def mean_radiant_temperature(longwave_mean_radiant_temperature: float, horizontal_infrared_radiation_intensity: float, diffuse_horizontal_solar: float, direct_normal_solar: float, sun_altitude: float, ground_reflectivity: float = 0.25, sky_exposure: float = 1):
    """ Perform a full outdoor sky and context surface radiant heat exchange.

    This method is a vectorised version of the ladybug-comfort outdoor_sky_heat_exch method https://www.ladybug.tools/ladybug-comfort/docs/_modules/ladybug_comfort/solarcal.html#outdoor_sky_heat_exch

    Parameters
    ----------
    longwave_mean_radiant_temperature : float
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

    Returns
    -------
    mean_radiant_temperature : float
        The final MRT experienced as a result of sky heat exchange in C.
    """

    def get_mrt(z):
        return z["mrt"]

    start_time = datetime.now()

    mrt = oshe(
        srfs_temp=np.array([longwave_mean_radiant_temperature]),
        horiz_ir=np.array([horizontal_infrared_radiation_intensity]),
        diff_horiz_solar=np.array([diffuse_horizontal_solar]),
        dir_normal_solar=np.array([direct_normal_solar]),
        alt=np.array([sun_altitude]),
        sky_exposure=np.array([sky_exposure]),
        floor_reflectance=np.array([ground_reflectivity])
    )

    td = (datetime.now() - start_time).total_seconds()
    print("Mean radiant temperature calculated [{}]".format('{0:0.02f} seconds'.format(td)))
    return mrt
