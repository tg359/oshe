import io
import pathlib
import subprocess
import tempfile

import pandas as pd
from eppy.modeleditor import IDF
from ladybug.epw import EPW

from .helpers import load_energyplus_results


def open_field_ground_surface_temperature(epw_file: str, idd_file: str, case_name: str = None, output_directory: str = None, shaded: bool = False, roughness: str = "MediumRough", thickness: float = 0.2, conductivity: float = 1.73, density: float = 2243, specific_heat: float = 837, thermal_absorptance: float = 0.9, solar_absorptance: float = 0.65, visible_absorptance: float = 0.7, ) -> float:
    """ Calculate the surface temperature of a bit of ground using EnergyPlus

    Parameters
    ----------
    epw_file : str
        Path to EPW
    idd_file : str
        Path to EnergyPlus IDD file
    case_name : str
        Name of case being simulated
    output_directory : str
        Directory where simulation results will be stored
    shaded : bool
        True if shaded. Shade properties are 10% transmissive to allow for reflections from nearby surfaaces as shade object is an enclosure
    roughness : str
        Roughness of ground material
    thickness : float
        Thickness of ground material
    conductivity : float
        Conductivity of ground material
    density : float
        Density of ground material
    specific_heat : float
        Specific heat capacity of ground material
    thermal_absorptance : float
        Thermal absorptance (emissivity) of ground material
    solar_absorptance
        Solar absorptance (inverted radiation reflectivity) of ground material
    visible_absorptance
        Visible absorptance (inverted visible radiation reflectivity) of ground material

    Returns
    -------
        ground_surface_temperature : float
    """

    # Construct case output directory and process variables
    case_name = "openfield" if case_name is None else case_name
    output_directory = pathlib.Path(tempfile.gettempdir()) if output_directory is None else pathlib.Path(output_directory)
    eplus_output_path = pathlib.Path(output_directory) / case_name / "ground_surface_temperature"
    eplus_output_path.mkdir(parents=True, exist_ok=True)
    idd_file = pathlib.Path(idd_file)
    eplus = idd_file.parent / "energyplus.exe"
    idf_file = eplus_output_path / "in.idf"
    csv_file = eplus_output_path / "eplusout.csv"

    # Load monthly ground air_temperature from weather-file
    monthly_ground_temperatures = EPW(epw_file).monthly_ground_temperature[0.5].values

    # Construct EPlus file to run
    IDF.setiddname(str(idd_file))

    # Clear directory of results prior to running
    for x in eplus_output_path.iterdir():
        pathlib.Path.unlink(x)

    # Construct case for simulation
    idf = IDF(io.StringIO(""))

    # Build EnergyPlus case
    building = idf.newidfobject("BUILDING")
    building.Name = "Building 1"
    building.North_Axis = 0
    building.Terrain = "City"
    building.Solar_Distribution = "FullExterior"
    building.Maximum_Number_of_Warmup_Days = 25
    building.Minimum_Number_of_Warmup_Days = 6

    timestep = idf.newidfobject("TIMESTEP")
    timestep.Number_of_Timesteps_per_Hour = 6

    globalgeometryrules = idf.newidfobject("GLOBALGEOMETRYRULES")
    globalgeometryrules.Starting_Vertex_Position = "UpperLeftCorner"
    globalgeometryrules.Vertex_Entry_Direction = "Counterclockwise"
    globalgeometryrules.Coordinate_System = "Relative"

    shadowcalculation = idf.newidfobject("SHADOWCALCULATION")
    shadowcalculation.Calculation_Method = "TimestepFrequency"
    shadowcalculation.Calculation_Frequency = 1
    shadowcalculation.Maximum_Figures_in_Shadow_Overlap_Calculations = 3000

    groundtemperature = idf.newidfobject("SITE:GROUNDTEMPERATURE:BUILDINGSURFACE")
    for i, j in list(zip(*[pd.date_range("2018", "2019", freq="1M").strftime("%B"), monthly_ground_temperatures])):
        setattr(groundtemperature, "{}_Ground_Temperature".format(i), j)

    zone = idf.newidfobject("ZONE")
    zone.Name = "ground_zone"

    # Add ground surface and edge geometry
    if True:
        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_bottom"
        gnd.Surface_Type = "Floor"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = -2
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = -2

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_top"
        gnd.Surface_Type = "Roof"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Outdoors"
        gnd.Sun_Exposure = "Sunexposed"
        gnd.Wind_Exposure = "Windexposed"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = 0
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = 0
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_0"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = 10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = -10
        gnd.Vertex_4_Ycoordinate = -10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_1"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = -10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_2"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = 10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_3"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = 10
        gnd.Vertex_2_Ycoordinate = 10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = -10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

    # Generic materials for ground types
    material = idf.newidfobject("MATERIAL")
    material.Name = "ground_material"
    material.Roughness = roughness
    material.Thickness = thickness
    material.Conductivity = conductivity
    material.Density = density
    material.Specific_Heat = specific_heat
    material.Thermal_Absorptance = thermal_absorptance
    material.Solar_Absorptance = solar_absorptance
    material.Visible_Absorptance = visible_absorptance

    construction = idf.newidfobject("CONSTRUCTION")
    construction.Name = "ground_construction"
    construction.Outside_Layer = "ground_material"

    shd_schedule_limits = idf.newidfobject("SCHEDULETYPELIMITS")
    shd_schedule_limits.Name = "shade_schedule_type_limit"
    shd_schedule_limits.Lower_Limit_Value = 0
    shd_schedule_limits.Upper_Limit_Value = 1
    shd_schedule_limits.Numeric_Type = "Continuous"

    shd_schedule = idf.newidfobject("SCHEDULE:CONSTANT")
    shd_schedule.Name = "shade_schedule"
    shd_schedule.Schedule_Type_Limits_Name = "shade_schedule_type_limit"
    shd_schedule.Hourly_Value = 0 if shaded else 1

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_top"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 3
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 3
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_0"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = -50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = -50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_1"
    shd.Vertex_1_Xcoordinate = 50
    shd.Vertex_1_Ycoordinate = -50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = 50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = 50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_2"
    shd.Vertex_1_Xcoordinate = 50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = 50
    shd.Vertex_2_Ycoordinate = 50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = -50
    shd.Vertex_3_Ycoordinate = 50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = -50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_3"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = 50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = -50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = -50
    shd.Vertex_4_Ycoordinate = -50
    shd.Vertex_4_Zcoordinate = 3

    diagnostics = idf.newidfobject("OUTPUT:DIAGNOSTICS")
    diagnostics.Key_1 = "DisplayExtraWarnings"

    outputvariable = idf.newidfobject("OUTPUT:VARIABLE")
    outputvariable.Key_Value = "ground_top"
    outputvariable.Variable_Name = "Surface Outside Face Temperature"
    outputvariable.Reporting_Frequency = "hourly"

    output_vardict = idf.newidfobject("OUTPUT:VARIABLEDICTIONARY")
    output_vardict.Key_Field = "IDF"

    idf.saveas(idf_file)

    # Run simulation
    cmd = '"{0:}" -a -r -w "{1:}" -d "{2:}" "{3:}"'.format(eplus.absolute(), epw_file, eplus_output_path, idf_file)
    subprocess.call(cmd, shell=True)
    print("Ground surface temperature simulation completed")

    # Read surface temperature results
    ground_surface_temperature = load_energyplus_results(csv_file)

    return ground_surface_temperature

def open_field_ground_surface_temperature2(epw_file: str, idd_file: str, case_name: str = None, output_directory: str = None, shaded: bool = False, roughness: str = "MediumRough", thickness: float = 0.2, conductivity: float = 1.73, density: float = 2243, specific_heat: float = 837, thermal_absorptance: float = 0.9, solar_absorptance: float = 0.65, visible_absorptance: float = 0.7, ) -> float:
    """ Calculate the surface temperature of a bit of ground using EnergyPlus

    Parameters
    ----------
    epw_file : str
        Path to EPW
    idd_file : str
        Path to EnergyPlus IDD file
    case_name : str
        Name of case being simulated
    output_directory : str
        Directory where simulation results will be stored
    shaded : bool
        True if shaded. Shade properties are 10% transmissive to allow for reflections from nearby surfaaces as shade object is an enclosure
    roughness : str
        Roughness of ground material
    thickness : float
        Thickness of ground material
    conductivity : float
        Conductivity of ground material
    density : float
        Density of ground material
    specific_heat : float
        Specific heat capacity of ground material
    thermal_absorptance : float
        Thermal absorptance (emissivity) of ground material
    solar_absorptance
        Solar absorptance (inverted radiation reflectivity) of ground material
    visible_absorptance
        Visible absorptance (inverted visible radiation reflectivity) of ground material

    Returns
    -------
        ground_surface_temperature : float
    """

    # Construct case output directory and process variables
    case_name = "openfield" if case_name is None else case_name
    output_directory = pathlib.Path(tempfile.gettempdir()) if output_directory is None else pathlib.Path(output_directory)
    eplus_output_path = pathlib.Path(output_directory) / case_name / "ground_surface_temperature"
    eplus_output_path.mkdir(parents=True, exist_ok=True)
    idd_file = pathlib.Path(idd_file)
    eplus = idd_file.parent / "energyplus.exe"
    idf_file = eplus_output_path / "in.idf"
    csv_file = eplus_output_path / "eplusout.csv"

    # Load monthly ground air_temperature from weather-file
    monthly_ground_temperatures = EPW(epw_file).monthly_ground_temperature[0.5].values

    # Construct EPlus file to run
    IDF.setiddname(str(idd_file))

    # Clear directory of results prior to running
    for x in eplus_output_path.iterdir():
        pathlib.Path.unlink(x)

    # Construct case for simulation
    idf = IDF(io.StringIO(""))

    # Build EnergyPlus case
    building = idf.newidfobject("BUILDING")
    building.Name = "Building 1"
    building.North_Axis = 0
    building.Terrain = "City"
    building.Solar_Distribution = "FullExterior"
    building.Maximum_Number_of_Warmup_Days = 25
    building.Minimum_Number_of_Warmup_Days = 6

    timestep = idf.newidfobject("TIMESTEP")
    timestep.Number_of_Timesteps_per_Hour = 6

    globalgeometryrules = idf.newidfobject("GLOBALGEOMETRYRULES")
    globalgeometryrules.Starting_Vertex_Position = "UpperLeftCorner"
    globalgeometryrules.Vertex_Entry_Direction = "Counterclockwise"
    globalgeometryrules.Coordinate_System = "Relative"

    shadowcalculation = idf.newidfobject("SHADOWCALCULATION")
    shadowcalculation.Calculation_Method = "TimestepFrequency"
    shadowcalculation.Calculation_Frequency = 1
    shadowcalculation.Maximum_Figures_in_Shadow_Overlap_Calculations = 3000

    groundtemperature = idf.newidfobject("SITE:GROUNDTEMPERATURE:BUILDINGSURFACE")
    for i, j in list(zip(*[pd.date_range("2018", "2019", freq="1M").strftime("%B"), monthly_ground_temperatures])):
        setattr(groundtemperature, "{}_Ground_Temperature".format(i), j)

    zone = idf.newidfobject("ZONE")
    zone.Name = "ground_zone"

    # Add ground surface and edge geometry
    if True:
        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_bottom"
        gnd.Surface_Type = "Floor"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = -2
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = -2

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_top"
        gnd.Surface_Type = "Roof"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Outdoors"
        gnd.Sun_Exposure = "Sunexposed"
        gnd.Wind_Exposure = "Windexposed"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = 0
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = 0
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_0"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = 10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = -10
        gnd.Vertex_4_Ycoordinate = -10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_1"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = -10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = -10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = -10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = -10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_2"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = -10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = 10
        gnd.Vertex_2_Ycoordinate = -10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = 10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = 10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

        gnd = idf.newidfobject("BUILDINGSURFACE:DETAILED")
        gnd.Name = "ground_edge_3"
        gnd.Surface_Type = "Wall"
        gnd.Construction_Name = "ground_construction"
        gnd.Zone_Name = "ground_zone"
        gnd.Outside_Boundary_Condition = "Ground"
        gnd.Sun_Exposure = "Nosun"
        gnd.Wind_Exposure = "Nowind"
        gnd.Vertex_1_Xcoordinate = 10
        gnd.Vertex_1_Ycoordinate = 10
        gnd.Vertex_1_Zcoordinate = 0
        gnd.Vertex_2_Xcoordinate = 10
        gnd.Vertex_2_Ycoordinate = 10
        gnd.Vertex_2_Zcoordinate = -2
        gnd.Vertex_3_Xcoordinate = -10
        gnd.Vertex_3_Ycoordinate = 10
        gnd.Vertex_3_Zcoordinate = -2
        gnd.Vertex_4_Xcoordinate = -10
        gnd.Vertex_4_Ycoordinate = 10
        gnd.Vertex_4_Zcoordinate = 0

    # Generic materials for ground types
    material = idf.newidfobject("MATERIAL")
    material.Name = "ground_material"
    material.Roughness = roughness
    material.Thickness = thickness
    material.Conductivity = conductivity
    material.Density = density
    material.Specific_Heat = specific_heat
    material.Thermal_Absorptance = thermal_absorptance
    material.Solar_Absorptance = solar_absorptance
    material.Visible_Absorptance = visible_absorptance

    construction = idf.newidfobject("CONSTRUCTION")
    construction.Name = "ground_construction"
    construction.Outside_Layer = "ground_material"

    shd_schedule_limits = idf.newidfobject("SCHEDULETYPELIMITS")
    shd_schedule_limits.Name = "shade_schedule_type_limit"
    shd_schedule_limits.Lower_Limit_Value = 0
    shd_schedule_limits.Upper_Limit_Value = 1
    shd_schedule_limits.Numeric_Type = "Continuous"

    shd_schedule = idf.newidfobject("SCHEDULE:CONSTANT")
    shd_schedule.Name = "shade_schedule"
    shd_schedule.Schedule_Type_Limits_Name = "shade_schedule_type_limit"
    shd_schedule.Hourly_Value = 0 if shaded else 1

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_top"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 3
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 3
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_0"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = -50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = -50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_1"
    shd.Vertex_1_Xcoordinate = 50
    shd.Vertex_1_Ycoordinate = -50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = 50
    shd.Vertex_2_Ycoordinate = -50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = 50
    shd.Vertex_3_Ycoordinate = 50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = 50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_2"
    shd.Vertex_1_Xcoordinate = 50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = 50
    shd.Vertex_2_Ycoordinate = 50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = -50
    shd.Vertex_3_Ycoordinate = 50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = -50
    shd.Vertex_4_Ycoordinate = 50
    shd.Vertex_4_Zcoordinate = 3

    shd = idf.newidfobject("SHADING:BUILDING:DETAILED")
    shd.Transmittance_Schedule_Name = "shade_schedule"
    shd.Name = "shade_edge_3"
    shd.Vertex_1_Xcoordinate = -50
    shd.Vertex_1_Ycoordinate = 50
    shd.Vertex_1_Zcoordinate = 3
    shd.Vertex_2_Xcoordinate = -50
    shd.Vertex_2_Ycoordinate = 50
    shd.Vertex_2_Zcoordinate = 0
    shd.Vertex_3_Xcoordinate = -50
    shd.Vertex_3_Ycoordinate = -50
    shd.Vertex_3_Zcoordinate = 0
    shd.Vertex_4_Xcoordinate = -50
    shd.Vertex_4_Ycoordinate = -50
    shd.Vertex_4_Zcoordinate = 3

    diagnostics = idf.newidfobject("OUTPUT:DIAGNOSTICS")
    diagnostics.Key_1 = "DisplayExtraWarnings"

    outputvariable = idf.newidfobject("OUTPUT:VARIABLE")
    outputvariable.Key_Value = "ground_top"
    outputvariable.Variable_Name = "Surface Outside Face Temperature"
    outputvariable.Reporting_Frequency = "hourly"

    output_vardict = idf.newidfobject("OUTPUT:VARIABLEDICTIONARY")
    output_vardict.Key_Field = "IDF"

    idf.saveas(idf_file)

    # Run simulation
    cmd = '"{0:}" -a -r -w "{1:}" -d "{2:}" "{3:}"'.format(eplus.absolute(), epw_file, eplus_output_path, idf_file)
    subprocess.call(cmd, shell=True)
    print("Ground surface temperature simulation completed")

    # Read surface temperature results
    ground_surface_temperature = load_energyplus_results(csv_file)

    return ground_surface_temperature

