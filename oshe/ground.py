import warnings, pathlib, tempfile

import pandas as pd
import numpy as np
#
from ladybug.epw import EPW
from ladybug.sunpath import Sunpath

from .surface_temperature import open_field_ground_surface_temperature
from .radiation import open_field_radiation
from .mrt import mean_radiant_temperature
from .utci import universal_thermal_climate_index
from .oshe import calc_sky_temperature


class Ground(object):
    def __init__(self, ground_type: str = None, roughness: str = "MediumRough", thickness: float = 0.2, conductivity: float = 1.73,density: float = 2243,specific_heat: float = 837,thermal_absorptance: float = 0.9,solar_absorptance: float = 0.65,visible_absorptance: float = 0.7):

        self.index = pd.date_range(start="2018-01-01 00:00:00", end="2019-01-01 00:00:00", freq="60T", closed="left")

        # Error handling for specified properties
        if roughness not in ["VeryRough", "Rough", "MediumRough", "MediumSmooth", "Smooth", "VerySmooth"]:
            raise ValueError('Roughness must be one of "VeryRough", "Rough", "MediumRough", "MediumSmooth", "Smooth" or "VerySmooth"')
        if thickness < 0:
            raise ValueError('Thickness must be greater than 0')
        if conductivity < 0:
            raise ValueError('Conductivity must be greater than 0')
        if density < 0:
            raise ValueError('Density must be greater than 0')
        if 100 >= specific_heat:
            raise ValueError('Specific heat must be greater than 100')
        if (thermal_absorptance < 0) | (thermal_absorptance > 0.99999):
            raise ValueError('Thermal absorptance must be between 0 and 0.99999')
        if (solar_absorptance < 0) | (solar_absorptance > 1):
            raise ValueError('Solar absorptance must be between 0 and 1 (inclusive)')
        if (visible_absorptance < 0) | (visible_absorptance > 1):
            raise ValueError('Visible absorptance must be between 0 and 1 (inclusive)')

        # A set of generic ground materials with pre-set thermal properties
        ground_materials = {
            "DrySand": {
                "roughness": "Rough",
                "thickness": 0.2,
                "conductivity": 0.33,
                "density": 1555,
                "specific_heat": 800,
                "thermal_absorptance": 0.85,
                "solar_absorptance": 0.65,
                "visible_absorptance": 0.7
            },
            "DryDust": {
                "roughness": "Rough",
                "thickness": 0.2,
                "conductivity": 0.5,
                "density": 1600,
                "specific_heat": 1026,
                "thermal_absorptance": 0.9,
                "solar_absorptance": 0.7,
                "visible_absorptance": 0.7
            },
            "MoistSoil": {
                "roughness": "Rough",
                "thickness": 0.2,
                "conductivity": 1,
                "density": 1250,
                "specific_heat": 1252,
                "thermal_absorptance": 0.92,
                "solar_absorptance": 0.75,
                "visible_absorptance": 0.7
            },
            "Mud": {
                "roughness": "MediumRough",
                "thickness": 0.2,
                "conductivity": 1.4,
                "density": 1840,
                "specific_heat": 1480,
                "thermal_absorptance": 0.95,
                "solar_absorptance": 0.8,
                "visible_absorptance": 0.7
            },
            "Concrete": {
                "roughness": "MediumRough",
                "thickness": 0.2,
                "conductivity": 1.73,
                "density": 2243,
                "specific_heat": 837,
                "thermal_absorptance": 0.9,
                "solar_absorptance": 0.65,
                "visible_absorptance": 0.7
            },
            "Asphalt": {
                "roughness": "MediumRough",
                "thickness": 0.2,
                "conductivity": 0.75,
                "density": 2360,
                "specific_heat": 920,
                "thermal_absorptance": 0.93,
                "solar_absorptance": 0.87,
                "visible_absorptance": 0.7
            },
            "Rock": {
                "roughness": "MediumRough",
                "thickness": 0.2,
                "conductivity": 3,
                "density": 2700,
                "specific_heat": 790,
                "thermal_absorptance": 0.96,
                "solar_absorptance": 0.55,
                "visible_absorptance": 0.7
            },
            "Default": {
                "roughness": roughness,
                "thickness": thickness,
                "conductivity": conductivity,
                "density": density,
                "specific_heat": specific_heat,
                "thermal_absorptance": thermal_absorptance,
                "solar_absorptance": solar_absorptance,
                "visible_absorptance": visible_absorptance
            }
        }

        if ground_type is None:
            ground_type = "Default"
        elif ground_type not in ground_materials:
            warnings.warn("ground_type given is not known. Where ground material properties are not given manually, these will default to those of Concrete.", UserWarning)
            ground_type = "Default"

        # Material properties
        self.name = ground_type.lower()
        self.ground_type = ground_type if ground_type is not None else "custom"
        self.roughness = ground_materials[ground_type]["roughness"]
        self.thickness = ground_materials[ground_type]["thickness"]
        self.conductivity = ground_materials[ground_type]["conductivity"]
        self.density = ground_materials[ground_type]["density"]
        self.specific_heat = ground_materials[ground_type]["specific_heat"]
        self.thermal_absorptance = ground_materials[ground_type]["thermal_absorptance"]
        self.solar_absorptance = ground_materials[ground_type]["solar_absorptance"]
        self.visible_absorptance = ground_materials[ground_type]["visible_absorptance"]
        self.reflectivity = 1 - ((ground_materials[ground_type]["solar_absorptance"] + ground_materials[ground_type]["visible_absorptance"]) / 2)

        # Calculated/assigned properties
        self.shaded = None
        self.sun_altitude = None
        self.radiation_direct = None
        self.radiation_diffuse = None
        self.ground_surface_temperature = None
        self.dry_bulb_temperature = None
        self.relative_humidity = None
        self.wind_speed = None
        self.horizontal_infrared_radiation_intensity = None
        self.mean_radiant_temperature = None
        self.universal_thermal_climate_index = None

        self.df_surface_temperature = None
        self.df_utci = None
        self.df_mrt = None

    def __repr__(self):
        return_string = "Ground: \n"
        for k, v in self.__dict__.items():
            if v is not None:
                return_string += "- {}: {}\n".format(k, v)
        return return_string

    def srf_temp(self, epw_file: str, idd_file: str, case_name: str = None, output_directory: str = None, shaded: bool = False):
        # Calculate annual hourly ground surface temperatures
        self.ground_surface_temperature = open_field_ground_surface_temperature(
            epw_file=epw_file,
            idd_file=idd_file,
            case_name=case_name,
            output_directory=output_directory,
            shaded=shaded,
            roughness=self.roughness,
            thickness=self.thickness,
            conductivity=self.conductivity,
            density=self.density,
            specific_heat=self.specific_heat,
            thermal_absorptance=self.thermal_absorptance,
            solar_absorptance=self.solar_absorptance,
            visible_absorptance=self.visible_absorptance
        )
        self.df_surface_temperature = pd.Series(self.ground_surface_temperature[0], name="ground_surface_temperature", index=self.index).to_frame()
        print("Surface temperature calculated for {}".format(self.name))
        return self.ground_surface_temperature

    def mrt(self, epw_file: str, idd_file: str, case_name: str = None, output_directory: str = None, shaded: bool = False, write: bool = True):

        case_name = "open_field" if case_name is None else case_name
        output_directory = pathlib.Path(tempfile.gettempdir()) if output_directory is None else pathlib.Path(output_directory)

        # Load weatherfile variables
        epw = EPW(epw_file)
        self.dry_bulb_temperature = np.array(epw.dry_bulb_temperature.values)
        self.horizontal_infrared_radiation_intensity = np.array(epw.horizontal_infrared_radiation_intensity.values)
        sun_path = Sunpath.from_location(epw.location)
        self.sun_altitude = np.array([sun_path.calculate_sun_from_hoy(i).altitude for i in range(8760)])

        # Load annual hourly ground surface temperatures
        self.ground_surface_temperature = self.srf_temp(epw_file, idd_file, case_name, output_directory, shaded)

        # Calculate annual hourly sky temperature
        sky_temp = calc_sky_temperature(self.horizontal_infrared_radiation_intensity)

        # Join surface tempertaures
        all_srf_temps = np.vstack([self.ground_surface_temperature, sky_temp])

        # Create sky and ground view factors
        view_factors = np.array([0.9, 0.1]) # Make ground a larger impactor of comfort than sky

        # Calculate surrounding surface temperatures including sky
        surrounding_surface_temperatures = np.power(np.matmul(view_factors.T, np.power(all_srf_temps.T + 273.15, 4).T), 0.25) - 273.15

        # Calculate annual hourly incident radiation
        self.radiation_direct, self.radiation_diffuse = open_field_radiation(
            epw_file=epw_file,
            ground_reflectance=self.reflectivity,
            case_name=case_name,
            output_directory=output_directory,
            shaded=shaded
        )

        # Reshape radiation arrays
        self.radiation_direct = self.radiation_direct.T[0]
        self.radiation_diffuse = self.radiation_diffuse.T[0]

        # Calculate annual hourly MRT
        self.mean_radiant_temperature = mean_radiant_temperature(
            surrounding_surfaces_temperature=surrounding_surface_temperatures,
            horizontal_infrared_radiation_intensity=self.horizontal_infrared_radiation_intensity,
            diffuse_horizontal_solar=self.radiation_diffuse,
            direct_normal_solar=self.radiation_direct,
            sun_altitude=self.sun_altitude,
            ground_reflectivity=self.reflectivity,#0,
            sky_exposure=0 if shaded else 1,
            radiance=True
        )[0]

        if write:
            mrt_file = str(output_directory / case_name) + "/" + str(case_name) + ".mrt"
            self.df_mrt = pd.Series(self.mean_radiant_temperature, name="mean_radiant_temperature", index=self.index).to_frame()
            self.df_mrt.round(3).to_csv(mrt_file, index=False)
            print("MRT file written to {}".format(mrt_file))

        return self.mean_radiant_temperature


    def utci(self, epw_file: str, idd_file: str, case_name: str = None, output_directory: str = None, shaded: bool = False, wind=1, write: bool = True):

        case_name = "open_field" if case_name is None else case_name
        output_directory = pathlib.Path(tempfile.gettempdir()) if output_directory is None else pathlib.Path(output_directory)

        # Load weather file variables
        epw = EPW(epw_file)
        self.dry_bulb_temperature = np.array(epw.dry_bulb_temperature.values)
        self.wind_speed = np.array(epw.wind_speed.values)
        self.relative_humidity = np.array(epw.relative_humidity.values)
        wind_speed = self.wind_speed if wind == 1 else 0 if wind == 0 else wind

        # Calculate mean radiant temperature
        self.mean_radiant_temperature = self.mrt(
            epw_file=epw_file,
            idd_file=idd_file,
            case_name=case_name,
            output_directory=output_directory,
            shaded=shaded,
            write=write
        )

        # Calculate UTCI
        self.universal_thermal_climate_index = universal_thermal_climate_index(
            air_temperature=self.dry_bulb_temperature,
            mean_radiant_temperature=self.mean_radiant_temperature,
            wind_speed=wind_speed,
            relative_humidity=self.relative_humidity
        )

        if write:
            utci_file = str(output_directory / case_name) + "/" + str(case_name) + ".utci"
            self.df_utci = pd.Series(self.universal_thermal_climate_index, name="universal_thermal_climate_index", index=self.index).to_frame()
            self.df_utci.round(3).to_csv(utci_file, index=False)
            print("UTCI file written to {}".format(utci_file))

        return self.universal_thermal_climate_index
