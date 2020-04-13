# coding=utf-8

"""Functions for adjusting MRT for radiative sky exchange (including shortwave solar).

The solarcal formulas of this module are taken from the following publications:
[1] Arens, E., T. Hoyt, X. Zhou, L. Huang, H. Zhang and S. Schiavon. 2015.
Modeling the comfort effects of short-wave solar radiation indoors.
Building and Environment, 88, 3-9. http://dx.doi.org/10.1016/j.buildenv.2014.09.004
https://escholarship.org/uc/item/89m1h2dg
[2] ASHRAE Standard 55 (2017). "Thermal Environmental Conditions for Human Occupancy".
Properties:
    * SOLARCAL_SPLINES:
        A dictionary with two keys: 'standing' and 'seated'.
        Each value for these keys is a 2D matrix of projection factors
        for human geometry.  Each row refers to an degree of azimuth and each
        column refers to a degree of altitude.
"""


import numpy as np


def calc_sky_temperature(horizontal_infrared_radiation, source_emissivity=1):
    """Calculate sky temperature in Celsius.

    See EnergyPlus Engineering Reference for more information:
    https://bigladdersoftware.com/epx/docs/8-9/engineering-reference/
    climate-calculations.html#energyplus-sky-temperature-calculation

    Args:
        horizontal_infrared_radiation: A float value that represents horizontal infrared radiation
            intensity in W/m2.
        source_emissivity: A float value between 0 and 1 indicating the emissivity
             of the heat source that is radiating to the sky. Default is 1 for
             most outdoor surfaces.

    Returns:
        sky_temp: A sky temperature value in C.
    """
    sigma = 5.6697e-8  # stefan-boltzmann constant
    return ((horizontal_infrared_radiation / (source_emissivity * sigma)) ** 0.25) - 273.15


def outdoor_sky_heat_exchange(visible_surfaces_temperature, horizontal_infrared_radiation, diffuse_horizontal_solar_radiation, direct_normal_solar_radiation, solar_altitude,
                              sky_exposure=1, fraction_body_exposed=1, floor_reflectance=0.25,
                              posture='standing', sharp=135,
                              body_absorptivity=0.7, body_emissivity=0.95, radiance=False):
    """ Perform a full outdoor sky radiant heat exchange.
    Args:
        visible_surfaces_temperature: The temperature of surfaces around the person in degrees
            Celsius. This includes the ground and any other surfaces
            blocking the view to the sky. When the temperature of these
            individual surfaces are known, the input here should be the
            average temperature of the surfaces weighted by view-factor to the human.
            When such individual surface temperatures are unknown, the outdoor
            dry bulb temperature is typically used as a proxy.
        horizontal_infrared_radiation: The horizontal infrared radiation intensity from the sky in W/m2.
        diffuse_horizontal_solar_radiation: Diffuse horizontal solar irradiance in W/m2.
        direct_normal_solar_radiation: Direct normal solar irradiance in W/m2.
        solar_altitude: The altitude of the sun in degrees [0-90].
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
        fraction_body_exposed: A number between 0 and 1 representing the fraction of
            the body exposed to direct sunlight. Note that this does not include the
            body’s self-shading; only the shading from surroundings.
            Default is 1 for a person standing in an open area.
        floor_reflectance: A number between 0 and 1 the represents the
            reflectance of the floor. Default is for 0.25 which is characteristic
            of outdoor grass or dry bare soil.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). 0 signifies sun that is
            shining directly into the person's face and 180 signifies sun that
            is shining at the person's back. Default is 135, assuming that a person
            typically faces their side or back to the sun to avoid glare.
        body_absorptivity: A number between 0 and 1 representing the average
            shortwave absorptivity of the body (including clothing and skin color).
            Typical clothing values - white: 0.2, khaki: 0.57, black: 0.88
            Typical skin values - white: 0.57, brown: 0.65, black: 0.84
            Default is 0.7 for average (brown) skin and medium clothing.
        body_emissivity: A number between 0 and 1 representing the average
            long-wave emissivity of the body.  Default is 0.95, which is almost
            always the case except in rare situations of wearing metallic clothing.
    Returns:
        mrt: The final MRT experienced as a result of sky heat exchange in C.
    """
    # set defaults using the input parameters
    fractional_efficiency = 0.696 if posture == 'seated' else 0.725

    # calculate the influence of shortwave irradiance
    if solar_altitude >= 0:
        s_flux = body_solar_flux_from_parts(diffuse_horizontal_solar_radiation, direct_normal_solar_radiation,
                                            solar_altitude, sharp, 1 if radiance else sky_exposure,
                                            fraction_body_exposed, floor_reflectance, posture)
        short_effective_radiant_field = effective_radiant_field_from_body_solar_flux(s_flux, body_absorptivity, body_emissivity)
        short_mrt_delta = mrt_delta_from_effective_radiant_field(short_effective_radiant_field, fractional_efficiency)
    else:
        short_effective_radiant_field = 0
        short_mrt_delta = 0

    # calculate the influence of long-wave heat exchange with the sky
    long_mrt_delta = longwave_mrt_delta_from_horiz_ir(horizontal_infrared_radiation, visible_surfaces_temperature,
                                                      sky_exposure, body_emissivity)
    long_effective_radiant_field = effective_radiant_field_from_mrt_delta(long_mrt_delta, fractional_efficiency)

    # calculate final MRT as a result of both long-wave and shortwave heat exchange
    sky_adjusted_mrt = visible_surfaces_temperature + short_mrt_delta + long_mrt_delta
    return sky_adjusted_mrt


oshe = np.vectorize(outdoor_sky_heat_exchange)


def mrt_delta_from_effective_radiant_field(effective_radiant_field, fraction_body_exposed=0.725, radiant_heat_transfer_coefficient=6.012):
    """Calculate the mean radiant temperature (MRT) delta as a result of an ERF.
    Args:
        effective_radiant_field: A number representing the effective radiant field (ERF) on the
            person in W/m2.
        fraction_body_exposed: A number representing the fraction of the body
            surface exposed to radiation from the environment. This is typically
            either 0.725 for a standing or supine person or 0.696 for a seated
            person. Default is 0.725 for a standing person.
        radiant_heat_transfer_coefficient: A number representing the radiant heat transfer coefficient
            in (W/m2-K).  Default is 6.012, which is almost always the case.
    """
    return effective_radiant_field / (fraction_body_exposed * radiant_heat_transfer_coefficient)


def effective_radiant_field_from_mrt_delta(mrt_delta, fraction_body_exposed=0.725, radiant_heat_transfer_coefficient=6.012):
    """Calculate the effective radiant field (ERF) from a MRT delta.
    Args:
        mrt_delta: A mean radiant temperature (MRT) delta in Kelvin or degrees Celcius.
        fraction_body_exposed: A number representing the fraction of the body
            surface exposed to radiation from the environment. This is typically
            either 0.725 for a standing or supine person or 0.696 for a seated
            person. Default is 0.725 for a standing person.
        radiant_heat_transfer_coefficient: A number representing the radiant heat transfer coefficient
            in (W/m2-K).  Default is 6.012, which is almost always the case.
    """
    return mrt_delta * fraction_body_exposed * radiant_heat_transfer_coefficient


def longwave_mrt_delta_from_horiz_ir(horiz_ir, srfs_temp, sky_exposure=1,
                                     body_emissivity=0.95):
    """Calculate the MRT delta as a result of longwave radiant exchange with the sky.
    Note that this value is typically negative since the earth (and humans)
    tend to radiate heat out to space in the longwave portion of the spectrum.
    Args:
        horiz_ir: A float value that represents the downwelling horizontal
            infrared radiation intensity in W/m2.
        srfs_temp: The temperature of surfaces around the person in degrees
            Celsius. This includes the ground and any other surfaces
            blocking the view to the sky. Typically, the dry bulb temperature
            is used when such surface temperatures are unknown.
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
    """
    sky_temp = calc_sky_temperature(horiz_ir, body_emissivity)
    return longwave_mrt_delta_from_sky_temp(sky_temp, srfs_temp, sky_exposure)


def longwave_mrt_delta_from_sky_temp(sky_temp, srfs_temp, sky_exposure=1):
    """Calculate the MRT delta as a result of longwave radiant exchange with the sky.
    Note that this value is typically negative since the earth (and humans)
    tend to radiate heat out to space in the longwave portion of the spectrum.
    Args:
        sky_temp: The sky temperature in degrees Celcius.
        srfs_temp: The temperature of surfaces around the person in degrees
            Celcius. This includes the ground and any other surfaces
            blocking the view to the sky. Typically, the dry bulb temperature
            is used when such surface temperatures are unknown.
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
    """
    return 0.5 * sky_exposure * (sky_temp - srfs_temp)


def effective_radiant_field_from_body_solar_flux(solar_flux, body_absorptivity=0.7, body_emissivity=0.95):
    """Calculate effective radiant field (ERF) from incident solar flux on body in W/m2.
    Args:
        solar_flux: A number for the average solar flux over the human body in W/m2.
        body_absorptivity: A number between 0 and 1 representing the average
            shortwave absorptivity of the body (including clothing and skin color).
            Typical clothing values - white: 0.2, khaki: 0.57, black: 0.88
            Typical skin values - white: 0.57, brown: 0.65, black: 0.84
            Default is 0.7 for average (brown) skin and medium clothing.
        body_emissivity: A number between 0 and 1 representing the average
            long-wave emissivity of the body.  Default is 0.95, which is almost
            always the case except in rare situations of wearing metalic clothing.
    """
    return solar_flux * (body_absorptivity / body_emissivity)


def body_solar_flux_from_parts(diffuse_horizontal_solar_radiation, direct_normal_solar_radiation, altitude,
                               sharp=135, sky_exposure=1, fract_exposed=1,
                               floor_reflectance=0.25, posture='standing'):
    """Estimate the total solar flux on human geometry from solar components.
    Args:
        diffuse_horizontal_solar_radiation: Diffuse horizontal solar irradiance in W/m2.
        direct_normal_solar_radiation: Direct normal solar irradiance in W/m2.
        altitude: The altitude of the sun in degrees [0-90].
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). 0 signifies sun that is
            shining directly into the person's face and 180 signifies sun that
            is shining at the person's back. Default is 135, asuming that a person
            typically faces their side or back to the sun to avoid glare.
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
        fract_exposed: A number between 0 and 1 representing the fraction of
            the body exposed to direct sunlight. Note that this does not include the
            body’s self-shading; only the shading from surroundings.
            Default is 1 for a person standing in an open area.
        floor_reflectance: A number between 0 and 1 the represents the
            reflectance of the floor. Default is for 0.25 which is characteristic
            of outdoor grass or dry bare soil.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
    """
    fract_eff = 0.696 if posture == 'seated' else 0.725
    glob_horiz = diffuse_horizontal_solar_radiation + (direct_normal_solar_radiation * np.sin(np.radians(altitude)))

    dir_solar = body_dir_from_dir_normal(direct_normal_solar_radiation, altitude, sharp,
                                         posture, fract_exposed)
    diff_solar = body_diff_from_diff_horiz(diffuse_horizontal_solar_radiation, sky_exposure, fract_eff)
    ref_solar = body_ref_from_glob_horiz(glob_horiz, floor_reflectance,
                                         sky_exposure, fract_eff)
    return dir_solar + diff_solar + ref_solar


def body_solar_flux_from_horiz_parts(diffuse_horizontal_solar_radiation, dir_horiz_solar, altitude,
                                     sharp=135, fract_exposed=1,
                                     floor_reflectance=0.25, posture='standing'):
    """Estimate total solar flux on human geometry from horizontal solar components.
    This method is useful for cases when one wants to take the hourly results
    of a spatial radiation study with Radiance and use them to build a map
    of ERF or MRT delta on a person.
    Args:
        diffuse_horizontal_solar_radiation: Diffuse horizontal solar irradiance in W/m2.
        dir_horiz_solar: Direct horizontal solar irradiance in W/m2.
        altitude: The altitude of the sun in degrees [0-90].
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). 0 signifies sun that is
            shining directly into the person's face and 180 signifies sun that
            is shining at the person's back. Default is 135, asuming that a person
            typically faces their side or back to the sun to avoid glare.
        fract_exposed: A number between 0 and 1 representing the fraction of
            the body exposed to direct sunlight. Note that this does not include the
            body’s self-shading; only the shading from surroundings.
            Default is 1 for a person standing in an open area.
        floor_reflectance: A number between 0 and 1 the represents the
            reflectance of the floor. Default is for 0.25 which is characteristic
            of outdoor grass or dry bare soil.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
    """
    fract_eff = 0.696 if posture == 'seated' else 0.725
    glob_horiz = diffuse_horizontal_solar_radiation + dir_horiz_solar

    dir_solar = body_dir_from_dir_horiz(dir_horiz_solar, altitude, sharp,
                                        posture, fract_exposed)
    diff_solar = body_diff_from_diff_horiz(diffuse_horizontal_solar_radiation, 1, fract_eff)
    ref_solar = body_ref_from_glob_horiz(glob_horiz, floor_reflectance, 1, fract_eff)
    return dir_solar + diff_solar + ref_solar


def body_diff_from_diff_horiz(diffuse_horizontal_solar_radiation, sky_exposure=1, fraction_body_exposed=0.725):
    """Estimate the diffuse solar flux on human geometry from diffuse horizontal solar.
    Args:
        diffuse_horizontal_solar_radiation: Diffuse horizontal solar irradiance in W/m2.
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
        fraction_body_exposed: A number representing the fraction of the body
            surface exposed to radiation from the environment. This is typically
            either 0.725 for a standing or supine person or 0.696 for a seated
            person. Default is 0.725 for a standing person.
    """
    return 0.5 * sky_exposure * fraction_body_exposed * diffuse_horizontal_solar_radiation


def body_ref_from_glob_horiz(glob_horiz_solar, floor_reflectance=0.25,
                             sky_exposure=1, fraction_body_exposed=0.725):
    """Estimate floor-reflected solar flux on human geometry from global horizontal solar.
    Args:
        glob_horiz_solar: Global horizontal solar irradiance in W/m2.
        floor_reflectance: A number between 0 and 1 the represents the
            reflectance of the floor. Default is for 0.25 which is characteristic
            of outdoor grass or dry bare soil.
        sky_exposure: A number between 0 and 1 representing the fraction of the
            sky vault in occupant’s view. Default is 1 for outdoors in an
            open field.
        fraction_body_exposed: A number representing the fraction of the body
            surface exposed to radiation from the environment. This is typically
            either 0.725 for a standing or supine person or 0.696 for a seated
            person. Default is 0.725 for a standing person.
    """
    return 0.5 * sky_exposure * fraction_body_exposed * glob_horiz_solar * floor_reflectance


def body_dir_from_dir_horiz(dir_horiz_solar, altitude, sharp=135,
                            posture='standing', fract_exposed=1):
    """Estimate the direct solar flux on human geometry from direct horizontal solar.
    Args:
        dir_horiz_solar: Direct horizontal solar irradiance in W/m2.
        altitude: A number between 0 and 90 representing the altitude of the
            sun in degrees.
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). 0 signifies sun that is
            shining directly into the person's face and 180 signifies sun that
            is shining at the person's back. Default is 135, asuming that a person
            typically faces their side or back to the sun to avoid glare.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
        fract_exposed: A number between 0 and 1 representing the fraction of
            the body exposed to direct sunlight. Note that this does not include
            the body’s self-shading; only the shading from surroundings.
            Default is 1 for a person in an open area.
    """
    proj_fac = get_projection_factor_simple(altitude, sharp, posture)
    direct_normal_solar_radiation = dir_horiz_solar / np.sin(np.radians(altitude))
    return proj_fac * fract_exposed * direct_normal_solar_radiation


def body_dir_from_dir_normal(direct_normal_solar_radiation, altitude, sharp=135,
                             posture='standing', fract_exposed=1):
    """Estimate the direct solar flux on human geometry from direct horizontal solar.
    Args:
        direct_normal_solar_radiation: Direct normal solar irradiance in W/m2.
        altitude: A number between 0 and 90 representing the altitude of the
            sun in degrees.
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). 0 signifies sun that is
            shining directly into the person's face and 180 signifies sun that
            is shining at the person's back. Default is 135, asuming that a person
            typically faces their side or back to the sun to avoid glare.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
        fract_exposed: A number between 0 and 1 representing the fraction of
            the body exposed to direct sunlight. Note that this does not include
            the body’s self-shading; only the shading from surroundings.
            Default is 1 for a person in an open area.
    """
    proj_fac = get_projection_factor_simple(altitude, sharp, posture)
    return proj_fac * fract_exposed * direct_normal_solar_radiation


def sharp_from_solar_and_body_azimuth(solar_azimuth, body_azimuth=0):
    """Calculate solar horizontal angle relative to front of person (SHARP).
    Args:
        solar_azimuth: A number between 0 and 360 representing the solar azimuth
            in degrees (0=North, 90=East, 180=South, 270=West).
        body_azimuth: A number between 0 and 360 representing the direction that
            the human is facing in degrees (0=North, 90=East, 180=South, 270=West).
    """
    angle_diff = abs(solar_azimuth - body_azimuth)
    if angle_diff <= 180:
        return angle_diff
    else:
        return 360 - angle_diff


def get_projection_factor_simple(altitude, sharp=135, posture='standing'):
    """Get the fraction of body surface area exposed to direct sun using a simpler method.
    This is effectively Ap / Ad in the original Solarcal equations.
    This is a more portable version of the get_projection_area() function
    since it does not rely on the large matrix of projection factors
    stored externally in csv files. However, it is less precise since it
    effectively interpolates over the missing parts of the matrix. So this is
    only recommended for cases where such csv files are missing.
    Args:
        altitude: A number between 0 and 90 representing the altitude of the
            sun in degrees.
        sharp: A number between 0 and 180 representing the solar horizontal
            angle relative to front of person (SHARP). Default is 135, asuming
            a person typically faces their side or back to the sun to avoid glare.
        posture: A text string indicating the posture of the body. Letters must
            be lowercase.  Choose from the following: "standing", "seated", "supine".
            Default is "standing".
    """
    if posture == 'supine':
        altitude, sharp = transpose_altitude_azimuth(altitude, sharp)
        posture = 'standing'

    if posture == 'standing':
        ap_table = ((0.254, 0.254, 0.228, 0.187, 0.149, 0.104, 0.059),
                    (0.248, 0.248, 0.225, 0.183, 0.145, 0.102, 0.059),
                    (0.239, 0.239, 0.218, 0.177, 0.138, 0.096, 0.059),
                    (0.225, 0.225, 0.199, 0.165, 0.127, 0.09, 0.059),
                    (0.205, 0.205, 0.182, 0.151, 0.116, 0.083, 0.059),
                    (0.183, 0.183, 0.165, 0.136, 0.109, 0.078, 0.059),
                    (0.167, 0.167, 0.155, 0.131, 0.107, 0.078, 0.059),
                    (0.175, 0.175, 0.161, 0.131, 0.111, 0.081, 0.059),
                    (0.199, 0.199, 0.178, 0.147, 0.12, 0.084, 0.059),
                    (0.22, 0.22, 0.196, 0.16, 0.126, 0.088, 0.059),
                    (0.238, 0.238, 0.21, 0.17, 0.133, 0.091, 0.059),
                    (0.249, 0.249, 0.22, 0.177, 0.138, 0.093, 0.059),
                    (0.252, 0.252, 0.223, 0.178, 0.138, 0.093, 0.059))
    elif posture == 'seated':
        ap_table = ((0.202, 0.226, 0.212, 0.211, 0.182, 0.156, 0.123),
                    (0.203, 0.228, 0.205, 0.2, 0.187, 0.158, 0.123),
                    (0.2, 0.231, 0.207, 0.202, 0.184, 0.155, 0.123),
                    (0.191, 0.227, 0.205, 0.201, 0.175, 0.149, 0.123),
                    (0.177, 0.214, 0.195, 0.192, 0.168, 0.141, 0.123),
                    (0.16, 0.196, 0.182, 0.181, 0.162, 0.134, 0.123),
                    (0.15, 0.181, 0.173, 0.17, 0.153, 0.129, 0.123),
                    (0.163, 0.18, 0.164, 0.158, 0.145, 0.125, 0.123),
                    (0.182, 0.181, 0.156, 0.145, 0.136, 0.122, 0.123),
                    (0.195, 0.181, 0.146, 0.134, 0.128, 0.118, 0.123),
                    (0.207, 0.178, 0.135, 0.121, 0.117, 0.117, 0.123),
                    (0.213, 0.174, 0.125, 0.109, 0.109, 0.116, 0.123),
                    (0.209, 0.167, 0.117, 0.106, 0.106, 0.114, 0.123))
    else:
        raise TypeError('Posture type {} is not recognized.'.format(posture))

    def _find_span(arr, x):
        # For ordered array arr, find the left index of the closest interval x falls in.
        for i in range(len(arr) - 1):
            if x <= arr[i+1] and x >= arr[i]:
                return i
        raise ValueError('altitude/azimuth {} is outside of acceptable ranges'.format(x))

    alt_range = (0, 15, 30, 45, 60, 75, 90)
    az_range = (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180)
    alt_i = _find_span(alt_range, altitude)
    az_i = _find_span(az_range, sharp)

    ap11 = ap_table[az_i][alt_i]
    ap12 = ap_table[az_i][alt_i + 1]
    ap21 = ap_table[az_i + 1][alt_i]
    ap22 = ap_table[az_i + 1][alt_i + 1]

    az1 = az_range[az_i]
    az2 = az_range[az_i+1]
    alt1 = alt_range[alt_i]
    alt2 = alt_range[alt_i+1]

    # Bi-linear interpolation
    ap = ap11 * (az2 - sharp) * (alt2 - altitude)
    ap += ap21 * (sharp - az1) * (alt2 - altitude)
    ap += ap12 * (az2 - sharp) * (altitude - alt1)
    ap += ap22 * (sharp - az1) * (altitude - alt1)
    ap /= (az2 - az1) * (alt2 - alt1)

    return ap


def transpose_altitude_azimuth(altitude, azimuth):
    """Transpose altitude and azimuth.
    This is necessary for getting correct projection factors for a supine posture
    from the standing posture matrix.
    """
    alt_temp = altitude
    altitude = abs(90 - azimuth)
    azimuth = alt_temp
    return altitude, azimuth