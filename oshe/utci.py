import numpy as np
from datetime import datetime
from .helpers import chunks
from multiprocessing import Pool


def saturated_vapour_pressure(air_temperature: float) -> float:
    """ Calculate saturated vapor pressure (hPa) at temperature (C).

    This equation of saturation vapor pressure is specific to the UTCI model.

    References
    ----------
    Source: Peuhkuri, Ruut, and Carsten Rode. 2016. "Heat and Mass Transfer in Buildings."”"

    Parameters
    ----------
    air_temperature : float

    Returns
    -------
    saturated_vapour_pressure : float

    """

    return 288.68 * (1.098 + air_temperature / 100) ** 8.02


def universal_thermal_climate_index(air_temperature: np.ndarray, mean_radiant_temperature: np.ndarray, wind_speed: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """ Calculate Universal Thermal Climate Index (UTCI) using a polynomial approximation.

    UTCI is an international standard for outdoor air_temperature sensation (aka. "feels-like" temperature) that attempts to fill the following requirements:
    1) Thermo-physiological significance in the whole range of heat exchange conditions of existing thermal environments
    2) Valid in all climates, seasons, and scales
    3) Useful for key applications in human biometeorology.

    This function here is a Python version of the original UTCI_approx application written in Fortran. Version a 0.002, October 2009

    The original Fortran code can be found at www.utci.org.

    References
    ----------
    Peter Bröde, Dusan Fiala, Krzysztof Blazejczyk, Yoram Epstein, Ingvar Holmér, Gerd Jendritzky, Bernhard Kampmann, Mark Richards, Hannu Rintamäki, Avraham Shitzer, George Havenith. 2009. Calculating UTCI Equivalent Temperature. In: JW Castellani & TL Endrusick, eds. Proceedings of the 13th International Conference on Environmental Ergonomics, USARIEM, Natick, MA.

    Parameters
    ----------
    air_temperature : float
    Air air_temperature [C]
    mean_radiant_temperature : float
    Mean radiant air_temperature [C]
    wind_speed : float
    Wind speed 10 m above ground level [m/s]. Note that this meteorolical speed at 10 m is simply 1.5 times the speed felt at ground in the original Fiala model used to build UTCI.
    relative_humidity : float
    Relative humidity [%]

    Returns
    -------
    universal_thermal_climate_index : float
    The Universal Thermal Climate Index (UTCI) for the input conditions as approximated by a 4-D polynomial.
    """

    start_time = datetime.now()
    # Convert to correct units
    delta_temperature = mean_radiant_temperature - air_temperature
    vapour_pressure = saturated_vapour_pressure(air_temperature) * (relative_humidity / 100.0) / 1000.0 # convert vapour pressure to kPa

    # Check for wind speed limits exceedence and replace with limit value
    wind_speed[wind_speed < 0.5] = 0.5
    wind_speed[wind_speed > 17] = 17

    utci_approx = air_temperature + 0.607562052 - 0.0227712343 * air_temperature + (8.06470249 * (10 ** (-4))) * np.power(air_temperature, 2) + (-1.54271372 * (10 ** (-4))) * np.power(air_temperature, 3) + (-3.24651735 * (10 ** (-6))) * np.power(air_temperature, 4) + (7.32602852 * (10 ** (-8))) * np.power(air_temperature, 5) + (1.35959073 * (10 ** (-9))) * np.power(air_temperature, 6) + (-2.25836520) * wind_speed + 0.0880326035 * air_temperature * wind_speed + 0.00216844454 * np.power(air_temperature, 2) * wind_speed + (-1.53347087 * (10 ** (-5))) * np.power(air_temperature, 3) * wind_speed + (-5.72983704 * (10 ** (-7))) * np.power(air_temperature, 4) * wind_speed + (-2.55090145 * (10 ** (-9))) * np.power(air_temperature, 5) * wind_speed + (-0.751269505) * np.power(wind_speed, 2) + (-0.00408350271) * air_temperature * np.power(wind_speed, 2) + (-5.21670675 * (10 ** (-5))) * np.power(air_temperature, 2) * np.power(wind_speed, 2) + (1.94544667 * (10 ** (-6))) * np.power(air_temperature, 3) * np.power(wind_speed, 2) + (1.14099531 * (10 ** (-8))) * np.power(air_temperature, 4) * np.power(wind_speed, 2) + 0.158137256 * np.power(wind_speed, 3) + (-6.57263143 * (10 ** (-5))) * air_temperature * np.power(wind_speed, 3) + (2.22697524 * (10 ** (-7))) * np.power(air_temperature, 2) * np.power(wind_speed, 3) + (-4.16117031 * (10 ** (-8))) * np.power(air_temperature, 3) * np.power(wind_speed, 3) + (-0.0127762753) * np.power(wind_speed, 4) + (9.66891875 * (10 ** (-6))) * air_temperature * np.power(wind_speed, 4) + (2.52785852 * (10 ** (-9))) * np.power(air_temperature, 2) * np.power(wind_speed, 4) + (4.56306672 * (10 ** (-4))) * np.power(wind_speed, 5) + (-1.74202546 * (10 ** (-7))) * air_temperature * np.power(wind_speed, 5) + (-5.91491269 * (10 ** (-6))) * np.power(wind_speed, 6) + (0.398374029) * delta_temperature + (1.83945314 * (10 ** (-4))) * air_temperature * delta_temperature + (-1.73754510 * (10 ** (-4))) * np.power(air_temperature, 2) * delta_temperature + (-7.60781159 * (10 ** (-7))) * np.power(air_temperature, 3) * delta_temperature + (3.77830287 * (10 ** (-8))) * np.power(air_temperature, 4) * delta_temperature + (5.43079673 * (10 ** (-10))) * np.power(air_temperature, 5) * delta_temperature + (-0.0200518269) * wind_speed * delta_temperature + (8.92859837 * (10 ** (-4))) * air_temperature * wind_speed * delta_temperature + (3.45433048 * (10 ** (-6))) * np.power(air_temperature, 2) * wind_speed * delta_temperature + (-3.77925774 * (10 ** (-7))) * np.power(air_temperature, 3) * wind_speed * delta_temperature + (-1.69699377 * (10 ** (-9))) * np.power(air_temperature, 4) * wind_speed * delta_temperature + (1.69992415 * (10 ** (-4))) * np.power(wind_speed, 2) * delta_temperature + (-4.99204314 * (10 ** (-5))) * air_temperature * np.power(wind_speed, 2) * delta_temperature + (2.47417178 * (10 ** (-7))) * np.power(air_temperature, 2) * np.power(wind_speed,  2) * delta_temperature + (1.07596466 * (10 ** (-8))) * np.power(air_temperature, 3) * np.power(wind_speed,  2) * delta_temperature + (8.49242932 * (10 ** (-5))) * np.power(wind_speed, 3) * delta_temperature + (1.35191328 * (10 ** (-6))) * air_temperature * np.power(wind_speed, 3) * delta_temperature + (-6.21531254 * (10 ** (-9))) * np.power(air_temperature, 2) * np.power(wind_speed,  3) * delta_temperature + (-4.99410301 * (10 ** (-6))) * np.power(wind_speed, 4) * delta_temperature + (-1.89489258 * (10 ** (-8))) * air_temperature * np.power(wind_speed, 4) * delta_temperature + (8.15300114 * (10 ** (-8))) * np.power(wind_speed, 5) * delta_temperature + (7.55043090 * (10 ** (-4))) * np.power(delta_temperature, 2) + (-5.65095215 * (10 ** (-5))) * air_temperature * np.power(delta_temperature, 2) + (-4.52166564 * (10 ** (-7))) * np.power(air_temperature, 2) * np.power(delta_temperature, 2) + (2.46688878 * (10 ** (-8))) * np.power(air_temperature, 3) * np.power(delta_temperature, 2) + (2.42674348 * (10 ** (-10))) * np.power(air_temperature, 4) * np.power(delta_temperature, 2) + (1.54547250 * (10 ** (-4))) * wind_speed * np.power(delta_temperature, 2) + (5.24110970 * (10 ** (-6))) * air_temperature * wind_speed * np.power(delta_temperature, 2) + (-8.75874982 * (10 ** (-8))) * np.power(air_temperature, 2) * wind_speed * np.power(delta_temperature,  2) + (-1.50743064 * (10 ** (-9))) * np.power(air_temperature, 3) * wind_speed * np.power(delta_temperature,  2) + (-1.56236307 * (10 ** (-5))) * np.power(wind_speed, 2) * np.power(delta_temperature, 2) + (-1.33895614 * (10 ** (-7))) * air_temperature * np.power(wind_speed, 2) * np.power(delta_temperature,  2) + (2.49709824 * (10 ** (-9))) * np.power(air_temperature, 2) * np.power(wind_speed, 2) * np.power(  delta_temperature, 2) + (6.51711721 * (10 ** (-7))) * np.power(wind_speed, 3) * np.power(delta_temperature, 2) + (1.94960053 * (10 ** (-9))) * air_temperature * np.power(wind_speed, 3) * np.power(delta_temperature,  2) + (-1.00361113 * (10 ** (-8))) * np.power(wind_speed, 4) * np.power(delta_temperature, 2) + (-1.21206673 * (10 ** (-5))) * np.power(delta_temperature, 3) + (-2.18203660 * (10 ** (-7))) * air_temperature * np.power(delta_temperature, 3) + (7.51269482 * (10 ** (-9))) * np.power(air_temperature, 2) * np.power(delta_temperature, 3) + (9.79063848 * (10 ** (-11))) * np.power(air_temperature, 3) * np.power(delta_temperature, 3) + (1.25006734 * (10 ** (-6))) * wind_speed * np.power(delta_temperature, 3) + (-1.81584736 * (10 ** (-9))) * air_temperature * wind_speed * np.power(delta_temperature, 3) + (-3.52197671 * (10 ** (-10))) * np.power(air_temperature, 2) * wind_speed * np.power(  delta_temperature, 3) + (-3.36514630 * (10 ** (-8))) * np.power(wind_speed, 2) * np.power(delta_temperature, 3) + (1.35908359 * (10 ** (-10))) * air_temperature * np.power(wind_speed, 2) * np.power(delta_temperature,  3) + (4.17032620 * (10 ** (-10))) * np.power(wind_speed, 3) * np.power(delta_temperature, 3) + (-1.30369025 * (10 ** (-9))) * np.power(delta_temperature, 4) + (4.13908461 * (10 ** (-10))) * air_temperature * np.power(delta_temperature, 4) + (9.22652254 * (10 ** (-12))) * np.power(air_temperature, 2) * np.power(delta_temperature, 4) + (-5.08220384 * (10 ** (-9))) * wind_speed * np.power(delta_temperature, 4) + (-2.24730961 * (10 ** (-11))) * air_temperature * wind_speed * np.power(delta_temperature, 4) + (1.17139133 * (10 ** (-10))) * np.power(wind_speed, 2) * np.power(delta_temperature, 4) + (6.62154879 * (10 ** (-10))) * np.power(delta_temperature, 5) + (4.03863260 * (10 ** (-13))) * air_temperature * np.power(delta_temperature, 5) + (1.95087203 * (10 ** (-12))) * wind_speed * np.power(delta_temperature, 5) + (-4.73602469 * (10 ** (-12))) * np.power(delta_temperature, 6) + (5.12733497) * vapour_pressure + (-0.312788561) * air_temperature * vapour_pressure + (-0.0196701861) * np.power(air_temperature, 2) * vapour_pressure + (9.99690870 * (10 ** (-4))) * np.power(air_temperature, 3) * vapour_pressure + (9.51738512 * (10 ** (-6))) * np.power(air_temperature, 4) * vapour_pressure + (-4.66426341 * (10 ** (-7))) * np.power(air_temperature, 5) * vapour_pressure + (0.548050612) * wind_speed * vapour_pressure + (-0.00330552823) * air_temperature * wind_speed * vapour_pressure + (-0.00164119440) * np.power(air_temperature, 2) * wind_speed * vapour_pressure + (-5.16670694 * (10 ** (-6))) * np.power(air_temperature, 3) * wind_speed * vapour_pressure + (9.52692432 * (10 ** (-7))) * np.power(air_temperature, 4) * wind_speed * vapour_pressure + (-0.0429223622) * np.power(wind_speed, 2) * vapour_pressure + (0.00500845667) * air_temperature * np.power(wind_speed, 2) * vapour_pressure + (1.00601257 * (10 ** (-6))) * np.power(air_temperature, 2) * np.power(wind_speed,  2) * vapour_pressure + (-1.81748644 * (10 ** (-6))) * np.power(air_temperature, 3) * np.power(wind_speed,  2) * vapour_pressure + (-1.25813502 * (10 ** (-3))) * np.power(wind_speed, 3) * vapour_pressure + (-1.79330391 * (10 ** (-4))) * air_temperature * np.power(wind_speed, 3) * vapour_pressure + (2.34994441 * (10 ** (-6))) * np.power(air_temperature, 2) * np.power(wind_speed,  3) * vapour_pressure + (1.29735808 * (10 ** (-4))) * np.power(wind_speed, 4) * vapour_pressure + (1.29064870 * (10 ** (-6))) * air_temperature * np.power(wind_speed, 4) * vapour_pressure + (-2.28558686 * (10 ** (-6))) * np.power(wind_speed, 5) * vapour_pressure + (-0.0369476348) * delta_temperature * vapour_pressure + (0.00162325322) * air_temperature * delta_temperature * vapour_pressure + (-3.14279680 * (10 ** (-5))) * np.power(air_temperature, 2) * delta_temperature * vapour_pressure + (2.59835559 * (10 ** (-6))) * np.power(air_temperature, 3) * delta_temperature * vapour_pressure + (-4.77136523 * (10 ** (-8))) * np.power(air_temperature, 4) * delta_temperature * vapour_pressure + (8.64203390 * (10 ** (-3))) * wind_speed * delta_temperature * vapour_pressure + (-6.87405181 * (10 ** (-4))) * air_temperature * wind_speed * delta_temperature * vapour_pressure + (-9.13863872 * (10 ** (-6))) * np.power(air_temperature,  2) * wind_speed * delta_temperature * vapour_pressure + (5.15916806 * (10 ** (-7))) * np.power(air_temperature,  3) * wind_speed * delta_temperature * vapour_pressure + (-3.59217476 * (10 ** (-5))) * np.power(wind_speed, 2) * delta_temperature * vapour_pressure + (3.28696511 * (10 ** (-5))) * air_temperature * np.power(wind_speed,  2) * delta_temperature * vapour_pressure + (-7.10542454 * (10 ** (-7))) * np.power(air_temperature, 2) * np.power(wind_speed,  2) * delta_temperature * vapour_pressure + (-1.24382300 * (10 ** (-5))) * np.power(wind_speed, 3) * delta_temperature * vapour_pressure + (-7.38584400 * (10 ** (-9))) * air_temperature * np.power(wind_speed,  3) * delta_temperature * vapour_pressure + (2.20609296 * (10 ** (-7))) * np.power(wind_speed, 4) * delta_temperature * vapour_pressure + (-7.32469180 * (10 ** (-4))) * np.power(delta_temperature, 2) * vapour_pressure + (-1.87381964 * (10 ** (-5))) * air_temperature * np.power(delta_temperature, 2) * vapour_pressure + (4.80925239 * (10 ** (-6))) * np.power(air_temperature, 2) * np.power(delta_temperature,  2) * vapour_pressure + (-8.75492040 * (10 ** (-8))) * np.power(air_temperature, 3) * np.power(delta_temperature,  2) * vapour_pressure + (2.77862930 * (10 ** (-5))) * wind_speed * np.power(delta_temperature, 2) * vapour_pressure + (-5.06004592 * (10 ** (-6))) * air_temperature * wind_speed * np.power(delta_temperature,  2) * vapour_pressure + (1.14325367 * (10 ** (-7))) * np.power(air_temperature, 2) * wind_speed * np.power(delta_temperature,  2) * vapour_pressure + (2.53016723 * (10 ** (-6))) * np.power(wind_speed, 2) * np.power(delta_temperature,  2) * vapour_pressure + (-1.72857035 * (10 ** (-8))) * air_temperature * np.power(wind_speed, 2) * np.power(delta_temperature,  2) * vapour_pressure + (-3.95079398 * (10 ** (-8))) * np.power(wind_speed, 3) * np.power(delta_temperature,  2) * vapour_pressure + (-3.59413173 * (10 ** (-7))) * np.power(delta_temperature, 3) * vapour_pressure + (7.04388046 * (10 ** (-7))) * air_temperature * np.power(delta_temperature, 3) * vapour_pressure + (-1.89309167 * (10 ** (-8))) * np.power(air_temperature, 2) * np.power(delta_temperature,  3) * vapour_pressure + (-4.79768731 * (10 ** (-7))) * wind_speed * np.power(delta_temperature, 3) * vapour_pressure + (7.96079978 * (10 ** (-9))) * air_temperature * wind_speed * np.power(delta_temperature,  3) * vapour_pressure + (1.62897058 * (10 ** (-9))) * np.power(wind_speed, 2) * np.power(delta_temperature,  3) * vapour_pressure + (3.94367674 * (10 ** (-8))) * np.power(delta_temperature, 4) * vapour_pressure + (-1.18566247 * (10 ** (-9))) * air_temperature * np.power(delta_temperature, 4) * vapour_pressure + (3.34678041 * (10 ** (-10))) * wind_speed * np.power(delta_temperature, 4) * vapour_pressure + (-1.15606447 * (10 ** (-10))) * np.power(delta_temperature, 5) * vapour_pressure + (-2.80626406) * np.power(vapour_pressure, 2) + (0.548712484) * air_temperature * np.power(vapour_pressure, 2) + (-0.00399428410) * np.power(air_temperature, 2) * np.power(vapour_pressure, 2) + (-9.54009191 * (10 ** (-4))) * np.power(air_temperature, 3) * np.power(vapour_pressure, 2) + (1.93090978 * (10 ** (-5))) * np.power(air_temperature, 4) * np.power(vapour_pressure, 2) + (-0.308806365) * wind_speed * np.power(vapour_pressure, 2) + (0.0116952364) * air_temperature * wind_speed * np.power(vapour_pressure, 2) + (4.95271903 * (10 ** (-4))) * np.power(air_temperature, 2) * wind_speed * np.power(vapour_pressure,  2) + (-1.90710882 * (10 ** (-5))) * np.power(air_temperature, 3) * wind_speed * np.power(vapour_pressure,  2) + (0.00210787756) * np.power(wind_speed, 2) * np.power(vapour_pressure, 2) + (-6.98445738 * (10 ** (-4))) * air_temperature * np.power(wind_speed, 2) * np.power(vapour_pressure,  2) + (2.30109073 * (10 ** (-5))) * np.power(air_temperature, 2) * np.power(wind_speed, 2) * np.power(  vapour_pressure, 2) + (4.17856590 * (10 ** (-4))) * np.power(wind_speed, 3) * np.power(vapour_pressure, 2) + (-1.27043871 * (10 ** (-5))) * air_temperature * np.power(wind_speed, 3) * np.power(vapour_pressure,  2) + (-3.04620472 * (10 ** (-6))) * np.power(wind_speed, 4) * np.power(vapour_pressure, 2) + (0.0514507424) * delta_temperature * np.power(vapour_pressure, 2) + (-0.00432510997) * air_temperature * delta_temperature * np.power(vapour_pressure, 2) + (8.99281156 * (10 ** (-5))) * np.power(air_temperature, 2) * delta_temperature * np.power(  vapour_pressure, 2) + (-7.14663943 * (10 ** (-7))) * np.power(air_temperature, 3) * delta_temperature * np.power(  vapour_pressure, 2) + (-2.66016305 * (10 ** (-4))) * wind_speed * delta_temperature * np.power(vapour_pressure, 2) + (2.63789586 * (10 ** (-4))) * air_temperature * wind_speed * delta_temperature * np.power(  vapour_pressure, 2) + (-7.01199003 * (10 ** (-6))) * np.power(air_temperature,  2) * wind_speed * delta_temperature * np.power(vapour_pressure, 2) + (-1.06823306 * (10 ** (-4))) * np.power(wind_speed, 2) * delta_temperature * np.power(vapour_pressure, 2) + (3.61341136 * (10 ** (-6))) * air_temperature * np.power(wind_speed, 2) * delta_temperature * np.power(vapour_pressure, 2) + (2.29748967 * (10 ** (-7))) * np.power(wind_speed, 3) * delta_temperature * np.power(vapour_pressure, 2) + (3.04788893 * (10 ** (-4))) * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (-6.42070836 * (10 ** (-5))) * air_temperature * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (1.16257971 * (10 ** (-6))) * np.power(air_temperature, 2) * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (7.68023384 * (10 ** (-6))) * wind_speed * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (-5.47446896 * (10 ** (-7))) * air_temperature * wind_speed * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (-3.59937910 * (10 ** (-8))) * np.power(wind_speed, 2) * np.power(delta_temperature, 2) * np.power(vapour_pressure, 2) + (-4.36497725 * (10 ** (-6))) * np.power(delta_temperature, 3) * np.power(vapour_pressure, 2) + (1.68737969 * (10 ** (-7))) * air_temperature * np.power(delta_temperature, 3) * np.power(vapour_pressure, 2) + (2.67489271 * (10 ** (-8))) * wind_speed * np.power(delta_temperature, 3) * np.power(vapour_pressure, 2) + (3.23926897 * (10 ** (-9))) * np.power(delta_temperature, 4) * np.power(vapour_pressure, 2) + (-0.0353874123) * np.power(vapour_pressure, 3) + (-0.221201190) * air_temperature * np.power(vapour_pressure, 3) + (0.0155126038) * np.power(air_temperature, 2) * np.power(vapour_pressure, 3) + (-2.63917279 * (10 ** (-4))) * np.power(air_temperature, 3) * np.power(vapour_pressure, 3) + (0.0453433455) * wind_speed * np.power(vapour_pressure, 3) + (-0.00432943862) * air_temperature * wind_speed * np.power(vapour_pressure, 3) + (1.45389826 * (10 ** (-4))) * np.power(air_temperature, 2) * wind_speed * np.power(vapour_pressure, 3) + (2.17508610 * (10 ** (-4))) * np.power(wind_speed, 2) * np.power(vapour_pressure, 3) + (-6.66724702 * (10 ** (-5))) * air_temperature * np.power(wind_speed, 2) * np.power(vapour_pressure, 3) + (3.33217140 * (10 ** (-5))) * np.power(wind_speed, 3) * np.power(vapour_pressure, 3) + (-0.00226921615) * delta_temperature * np.power(vapour_pressure, 3) + (3.80261982 * (10 ** (-4))) * air_temperature * delta_temperature * np.power(vapour_pressure, 3) + (-5.45314314 * (10 ** (-9))) * np.power(air_temperature, 2) * delta_temperature * np.power(vapour_pressure, 3) + (-7.96355448 * (10 ** (-4))) * wind_speed * delta_temperature * np.power(vapour_pressure, 3) + (2.53458034 * (10 ** (-5))) * air_temperature * wind_speed * delta_temperature * np.power(vapour_pressure, 3) + (-6.31223658 * (10 ** (-6))) * np.power(wind_speed, 2) * delta_temperature * np.power(vapour_pressure, 3) + (3.02122035 * (10 ** (-4))) * np.power(delta_temperature, 2) * np.power(vapour_pressure, 3) + (-4.77403547 * (10 ** (-6))) * air_temperature * np.power(delta_temperature, 2) * np.power(vapour_pressure, 3) + (1.73825715 * (10 ** (-6))) * wind_speed * np.power(delta_temperature, 2) * np.power(vapour_pressure, 3) + (-4.09087898 * (10 ** (-7))) * np.power(delta_temperature, 3) * np.power(vapour_pressure, 3) + (0.614155345) * np.power(vapour_pressure, 4) + (-0.0616755931) * air_temperature * np.power(vapour_pressure, 4) + (0.00133374846) * np.power(air_temperature, 2) * np.power(vapour_pressure, 4) + (0.00355375387) * wind_speed * np.power(vapour_pressure, 4) + (-5.13027851 * (10 ** (-4))) * air_temperature * wind_speed * np.power(vapour_pressure, 4) + (1.02449757 * (10 ** (-4))) * np.power(wind_speed, 2) * np.power(vapour_pressure, 4) + (-0.00148526421) * delta_temperature * np.power(vapour_pressure, 4) + (-4.11469183 * (10 ** (-5))) * air_temperature * delta_temperature * np.power(vapour_pressure, 4) + (-6.80434415 * (10 ** (-6))) * wind_speed * delta_temperature * np.power(vapour_pressure, 4) + (-9.77675906 * (10 ** (-6))) * np.power(delta_temperature, 2) * np.power(vapour_pressure, 4) + (0.0882773108) * np.power(vapour_pressure, 5) + (-0.00301859306) * air_temperature * np.power(vapour_pressure, 5) + (0.00104452989) * wind_speed * np.power(vapour_pressure, 5) + (2.47090539 * (10 ** (-4))) * delta_temperature * np.power(vapour_pressure, 5) + (0.00148348065) * np.power(vapour_pressure, 6)
    td = (datetime.now() - start_time).total_seconds()

    print("Universal thermal climate index calculated [{}]".format('{0:0.02f} seconds'.format(td)))
    return utci_approx


def utci_parallel_int(val_dict):
    universal_thermal_climate_index_part = universal_thermal_climate_index(
        val_dict["dbt"],
        val_dict["mrt"],
        val_dict["ws"],
        val_dict["rh"]
    )
    print("Thread #{0:} complete".format(val_dict["thread"]))
    return universal_thermal_climate_index_part


def utci_parallel(threads: int, air_temperature: np.ndarray, mean_radiant_temperature: np.ndarray, wind_speed: np.ndarray, relative_humidity: np.ndarray) -> np.ndarray:
    """ Run the UTCI calculation across multiple processors concurrently (useful for REALLY BIG CASES)
    Returns
    -------
    universal_thermal_climate_index : float
        The final UTCI experienced as a result of sky and surface heat exchange, and environmental conditions in C.
    """

    dicts = []
    for i in range(threads):
        dicts.append({
            "dbt": air_temperature,
            "mrt": chunks(mean_radiant_temperature.T, threads)[i],
            "ws": wind_speed,
            "rh": relative_humidity,
            "thread": i
        })

    p=Pool(processes=threads)
    output = p.map(utci_parallel_int, dicts)
    p.close()

    # Restack results
    universal_thermal_climate_index_result = np.vstack(output)

    return universal_thermal_climate_index_result
