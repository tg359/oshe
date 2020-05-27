import io
import pathlib
import platform
import subprocess
import tempfile
import typing
import warnings

import pandas as pd
import numpy as np
from eppy.modeleditor import IDF, IDDAlreadySetError
from ladybug.epw import EPW

from .geometry import Ground, Shade
from .helpers import flatten, chunk


class EsoFile(object):
    def __init__(self, eso_file: str):
        self.eso_file = eso_file
        self.objects = None
        self.variables = None
        self.values = None
        self.reporting_frequency = None
        self.version = None
        self.creation_date = None

        self.read_eso()
        self.df = self.to_frame()

    def read_eso(self):

        self.objects = []
        self.variables = []
        self.reporting_frequency = []
        self.values = []

        with open(self.eso_file, "r") as f:
            data = [i.strip() for i in f.readlines()]

            # Get metadata
            self.version = data[0].split(", ")[1].replace("Version ", "")
            self.creation_date = pd.to_datetime(data[0].split(", ")[2].replace("YMD=", ""))

            # Get reporting frequency and variables
            for i in data:
                if "!" in i and "! " not in i:
                    # Reporting frequency
                    self.reporting_frequency.append(i.split(" !")[-1])
                    # Object
                    self.objects.append(i.split(" !")[0].split(",")[2])
                    # Variable
                    self.variables.append(i.split(" !")[0].split(",")[3])

            # Get the values
            n_vals = len(self.reporting_frequency)
            split_idx = [n for n, i in enumerate(data) if i == "End of Data Dictionary"][0] + 2
            self.values = np.array(chunk([i.split(",")[1] for i in data[split_idx:-2] if len(i.split(",")) <= 2],  n=n_vals)).astype(np.float64).T

    def to_frame(self):
        idx = pd.date_range("2018-01-01 00:30:00", freq="60T", periods=8760)
        df = pd.DataFrame(data=self.values.T, index=idx,
                          columns=pd.MultiIndex.from_arrays([self.objects, self.variables]))
        return df


def run_energyplus(epw_file: str, idd_file: str, ground: Ground, shades: Shade = None,
                   output_directory: str = pathlib.Path(tempfile.gettempdir()), case_name: str = "openfield",
                   run: bool = False) -> EsoFile:
    """ Calculate the surface temperature of a bit of ground using EnergyPlus

    Parameters
    ----------
    epw_file : str
        Path to EPW
    idd_file : str
        Path to EnergyPlus IDD file
    ground : Ground
        Ground object containing thermal properties and geometry for ground in open-field
    shades : Shade
        List of shading objects
    output_directory : str
        Directory where simulation results will be stored
    case_name : str
        Name of case being simulated

    Returns
    -------
        EsoFile : Energyplus results object
    """

    # Construct case output directory and process variables
    output_directory = pathlib.Path(output_directory)
    eplus_output_path = output_directory / case_name / "ground_surface_temperature"
    eplus_output_path.mkdir(parents=True, exist_ok=True)
    idd_file = pathlib.Path(idd_file)
    eplus = idd_file.parent / "energyplus"
    idf_file = eplus_output_path / "in.idf"
    # csv_file = eplus_output_path / "eplusout.csv"
    eso_file = eplus_output_path / "eplusout.eso"

    # Construct EPlus file to run
    try:
        IDF.setiddname(str(idd_file))
    except IDDAlreadySetError as e:
        pass

    # Clear directory of results prior to running
    for x in eplus_output_path.iterdir():
        pathlib.Path.unlink(x)

    # Construct case for simulation
    idf = IDF(io.StringIO(""))

    # Build EnergyPlus case #
    building = idf.newidfobject("BUILDING")
    building.Name = "Openfield"
    building.North_Axis = 0
    building.Terrain = "Suburbs"
    building.Solar_Distribution = "FullExteriorWithReflections"
    building.Maximum_Number_of_Warmup_Days = 25
    building.Minimum_Number_of_Warmup_Days = 6

    timestep = idf.newidfobject("TIMESTEP")
    timestep.Number_of_Timesteps_per_Hour = 12  # 12 used as minimum for accurate vegetated surface simulation

    global_geometry_rules = idf.newidfobject("GLOBALGEOMETRYRULES")
    global_geometry_rules.Starting_Vertex_Position = "UpperLeftCorner"
    global_geometry_rules.Vertex_Entry_Direction = "Counterclockwise"
    global_geometry_rules.Coordinate_System = "Relative"

    shadow_calculation = idf.newidfobject("SHADOWCALCULATION")
    shadow_calculation.Calculation_Method = "TimestepFrequency"
    shadow_calculation.Calculation_Frequency = 1
    shadow_calculation.Maximum_Figures_in_Shadow_Overlap_Calculations = 3000

    # Set-up ground object based on Ground material passed
    try:
        # Load monthly ground temperatures from weather-file and assign to shallow objects
        ground_temperature_shallow = idf.newidfobject("SITE:GROUNDTEMPERATURE:SHALLOW")
        ground_temperature_building_surface = idf.newidfobject("SITE:GROUNDTEMPERATURE:BUILDINGSURFACE")
        for i, j in list(zip(*[pd.date_range("2018", "2019", freq="1M").strftime("%B"),
                               EPW(epw_file).monthly_ground_temperature[0.5].values])):
            setattr(ground_temperature_shallow, "{}_Surface_Ground_Temperature".format(i), j)
            setattr(ground_temperature_building_surface, "{}_Ground_Temperature".format(i), j)
    except Exception as e:
        warnings.warn(
            "Something went wrong - shallow ground temperatures (0.5m depth) from weatherfile not included in simulation\n{0:}".format(
                e))

    try:
        # Load monthly ground temperatures from weather-file and assign to deep objects
        ground_temperature_shallow = idf.newidfobject("SITE:GROUNDTEMPERATURE:DEEP")
        for i, j in list(zip(*[pd.date_range("2018", "2019", freq="1M").strftime("%B"),
                               EPW(epw_file).monthly_ground_temperature[4].values])):
            setattr(ground_temperature_shallow, "{}_Deep_Ground_Temperature".format(i), j)
    except Exception as e:
        warnings.warn(
            "Something went wrong - deep ground temperatures (4m depth) from weatherfile not included in simulation\n{0:}".format(
                e))

    # Load objects into IDF from inputs
    for eppy_object in flatten(ground.to_eppy(idd_file)):
        idf.idfobjects[eppy_object.obj[0]].append(eppy_object)
    if shades is not None:
        for shd in flatten([i.to_eppy(idd_file) for i in shades]):
            idf.idfobjects[shd.obj[0]].append(shd)

    diagnostics = idf.newidfobject("OUTPUT:DIAGNOSTICS")
    diagnostics.Key_1 = "DisplayExtraWarnings"

    output_vardict = idf.newidfobject("OUTPUT:VARIABLEDICTIONARY")
    output_vardict.Key_Field = "IDF"

    idf.saveas(idf_file)

    if run:
        # Run simulation
        cmd = '"{0:}" -a -w "{1:}" -d "{2:}" "{3:}"'.format(eplus.absolute(), epw_file, eplus_output_path, idf_file)
        subprocess.call(cmd, shell=True)
        print("Simulation completed")

        # Read surface temperature results
        # ground_surface_temperature = load_energyplus_results(csv_file)
        # results = EsoFile(eso_file=eso_file)
        # ground_surface_temperature = results.values

        return EsoFile(eso_file=eso_file)
    else:
        return idf_file


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


def view_factor_calculation():
    # TODO - Add view factor calculation method from sensor point location to sky, ground and shade objects
    return None



