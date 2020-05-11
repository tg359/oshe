import io
from typing import List

from eppy.bunch_subclass import EpBunch
from eppy.modeleditor import IDF, IDDAlreadySetError
from honeybee.radiance.material.plastic import Plastic
from honeybee.radiance.material.trans import Trans
from honeybee.radiance.properties import RadianceProperties

from .helpers import random_id


class MaterialBase(object):
    """ Base class for Material object """

    def __init__(self, name: str = None):
        self.name = "{0:}_{1:}".format("material" if name is None else name, random_id())

    def __repr__(self):
        return "{klass}\n{attrs}".format(
            klass=self.__class__.__name__,
            attrs="\n".join("- {}: {!r}".format(k, v) for k, v in self.__dict__.items()),
        )


class Softscape(MaterialBase):
    """ Vegetated ground material object
    Parameters
    ----------
    roughness : str
        Material roughness class (should be one of VeryRough, Rough, MediumRough, MediumSmooth, Smooth, VerySmooth)
    thickness : float
        Thickness of material in m
    conductivity : float
        Conductivity of material in W/m·K
    density : float
        Density of material in kg/m3
    specific_heat : float
        Specific heat capcity of material in J/kg·K
    thermal_absorptance : float
        Emissivity of material (value between 0 and 1)
    solar_absorptance : float
        Proportion of incident radiation absorbed by material (value between 0 and 1)
    visible_absorptance : float
        Proportion of light absorbed by material (value between 0 and 1)
    plant_height : float
        Average plant height in m
    leaf_area_index : float
        This is the projected leaf area per unit area of soil surface. This field is dimensionless and is limited to values in the range of 0.001 < LAI < 5.0. Default is 1.0. At the present time the fraction vegetation cover is calculated directly from LAI (Leaf Area Index) using an empirical relation. The user may find it necessary to increase the specified value of LAI in order to represent high fractional coverage of the surface by vegetation.
    leaf_reflectivity : float
        This field represents the fraction of incident solar radiation that is reflected by the individual leaf surfaces (albedo). Solar radiation includes the visible spectrum as well as infrared and ultraviolet wavelengths. Values for this field must be between 0.05 and 0.5. Default is .22. Typical values are .18 to .25.
    leaf_emissivity : float
        This field is the ratio of thermal radiation emitted from leaf surfaces to that emitted by an ideal black body at the same temperature. This parameter is used when calculating the long wavelength radiant exchange at the leaf surfaces. Values for this field must be between 0.8 and 1.0 (with 1.0 representing “black body” conditions). Default is .95.
    minimum_stomatal_resistance : float
        This field represents the resistance of the plants to moisture transport. It has units of s/m. Plants with low values of stomatal resistance will result in higher evapotranspiration rates than plants with high resistance. Values for this field must be in the range of 50.0 to 300.0. Default is 180.
    soil_layer_name : str
        This field is a unique reference name that the user assigns to the soil layer for a particular material. This name can then be referred to by other input data
    saturation_volumetric_moisture_content_of_the_soil_layer : float
        The field allows for user input of the saturation moisture content of the soil layer. Maximum moisture content is typically less than .5. Range is [.1,.5] with the default being .3.
    residual_volumetric_moisture_content_of_the_soil_layer : float
        The field allows for user input of the residual moisture content of the soil layer. Default is .01, range is [.01,.1].
    initial_volumetric_moisture_content_of_the_soil_layer : float
        The field allows for user input of the initial moisture content of the soil layer. Range is (.05, .5] with the default being .1.
    moisture_diffusion_calculation_method : str
        The field allows for two models to be selected: Simple or Advanced. Simple is the original Ecoroof model - based on a constant diffusion of moisture through the soil. This model starts with the soil in two layers. Every time the soil properties update is called, it will look at the two soils moisture layers and asses which layer has more moisture in it. It then takes moisture from the higher moisture layer and redistributes it to the lower moisture layer at a constant rate. Advanced is the later Ecoroof model. If you use it, you will need to increase your number of timesteps in hour for the simulation with a recommended value of 20. This moisture transport model is based on a project which looked at the way moisture transports through soil. It uses a finite difference method to divide the soil into layers (nodes). It redistributes the soil moisture according the model described in:
    """

    def __init__(self, roughness: str = "MediumRough", thickness: float = 0.1, conductivity: float = 0.35,
                 density: float = 1100, specific_heat: float = 1200, thermal_absorptance: float = 0.95,
                 solar_absorptance: float = 0.7, visible_absorptance: float = 0.75, plant_height: float = 0.2,
                 leaf_area_index: float = 1, leaf_reflectivity: float = 0.22, leaf_emissivity: float = 0.95,
                 minimum_stomatal_resistance: float = 180, soil_layer_name: str = "Soil",
                 saturation_volumetric_moisture_content_of_the_soil_layer: float = 0.3,
                 residual_volumetric_moisture_content_of_the_soil_layer: float = 0.01,
                 initial_volumetric_moisture_content_of_the_soil_layer: float = 0.1,
                 moisture_diffusion_calculation_method: str = "Advanced"):
        super().__init__()
        self.material_type = self.__class__.__name__
        self.roughness = roughness
        self.thickness = thickness
        self.conductivity = conductivity
        self.density = density
        self.specific_heat = specific_heat
        self.thermal_absorptance = thermal_absorptance
        self.solar_absorptance = solar_absorptance
        self.visible_absorptance = visible_absorptance
        self.plant_height: float = plant_height
        self.leaf_area_index: float = leaf_area_index
        self.leaf_reflectivity: float = leaf_reflectivity
        self.leaf_emissivity: float = leaf_emissivity
        self.minimum_stomatal_resistance: float = minimum_stomatal_resistance
        self.soil_layer_name: str = soil_layer_name
        self.saturation_volumetric_moisture_content_of_the_soil_layer: float = saturation_volumetric_moisture_content_of_the_soil_layer
        self.residual_volumetric_moisture_content_of_the_soil_layer: float = residual_volumetric_moisture_content_of_the_soil_layer
        self.initial_volumetric_moisture_content_of_the_soil_layer: float = initial_volumetric_moisture_content_of_the_soil_layer
        self.moisture_diffusion_calculation_method: str = moisture_diffusion_calculation_method
        self.reflectivity = self._average_reflectivity()

    def _average_reflectivity(self) -> float:
        average_ground_reflectivity = 1 - (self.solar_absorptance + self.visible_absorptance) / 2
        ground_fraction = 1 - min([1, self.leaf_area_index])
        leaf_fraction = 1 - ground_fraction
        return ground_fraction * average_ground_reflectivity + leaf_fraction * self.leaf_reflectivity

    def to_hb(self) -> RadianceProperties:
        """ Convert Material into Honeybee Radiance material property object """
        return RadianceProperties(
            material=Plastic(self.name, r_reflectance=self.reflectivity, g_reflectance=self.reflectivity,
                             b_reflectance=self.reflectivity))

    def to_eppy(self, idd_file: str) -> List[EpBunch]:
        """ Convert Material into Eppy material and construction objects """
        try:
            IDF.setiddname(str(idd_file))
        except IDDAlreadySetError as e:
            pass
        idf = IDF(io.StringIO(""))

        # Create list of eppy objects
        eppy_objects = []

        material = idf.newidfobject("MATERIAL:ROOFVEGETATION")
        material.Name = self.name
        material.Height_of_Plants = self.plant_height
        material.Leaf_Area_Index = self.leaf_area_index
        material.Leaf_Reflectivity = self.leaf_reflectivity
        material.Leaf_Emissivity = self.leaf_emissivity
        material.Minimum_Stomatal_Resistance = self.minimum_stomatal_resistance
        material.Soil_Layer_Name = self.soil_layer_name
        material.Roughness = self.roughness
        material.Thickness = self.thickness
        material.Conductivity_of_Dry_Soil = self.conductivity
        material.Density_of_Dry_Soil = self.density
        material.Specific_Heat_of_Dry_Soil = self.specific_heat
        material.Thermal_Absorptance = self.thermal_absorptance
        material.Solar_Absorptance = self.solar_absorptance
        material.Visible_Absorptance = self.visible_absorptance
        material.Saturation_Volumetric_Moisture_Content_of_the_Soil_Layer = self.saturation_volumetric_moisture_content_of_the_soil_layer
        material.Residual_Volumetric_Moisture_Content_of_the_Soil_Layer = self.residual_volumetric_moisture_content_of_the_soil_layer
        material.Initial_Volumetric_Moisture_Content_of_the_Soil_Layer = self.initial_volumetric_moisture_content_of_the_soil_layer
        material.Moisture_Diffusion_Calculation_Method = self.moisture_diffusion_calculation_method
        eppy_objects.append(material)

        construction = idf.newidfobject("CONSTRUCTION")
        construction.Name = self.name
        construction.Outside_Layer = self.name
        eppy_objects.append(construction)

        return eppy_objects

    def __repr__(self) -> str:
        return super().__repr__()


class Hardscape(MaterialBase):
    """ Solid ground material object
    Parameters
    ----------
    roughness : str
        Material roughness class (should be one of VeryRough, Rough, MediumRough, MediumSmooth, Smooth, VerySmooth)
    thickness : float
        Thickness of material in m
    conductivity : float
        Conductivity of material in W/m·K
    density : float
        Density of material in kg/m3
    specific_heat : float
        Specific heat capcity of material in J/kg·K
    thermal_absorptance : float
        Emissivity of material (value between 0 and 1)
    solar_absorptance : float
        Proportion of incident radiation absorbed by material (value between 0 and 1)
    visible_absorptance : float
        Proportion of light absorbed by material (value between 0 and 1)
    """

    def __init__(self, roughness: str = "MediumRough", thickness: float = 0.1, conductivity: float = 0.35,
                 density: float = 1100, specific_heat: float = 1200, thermal_absorptance: float = 0.95,
                 solar_absorptance: float = 0.7, visible_absorptance: float = 0.75):
        super().__init__()
        self.material_type = self.__class__.__name__
        self.roughness = roughness
        self.thickness = thickness
        self.conductivity = conductivity
        self.density = density
        self.specific_heat = specific_heat
        self.thermal_absorptance = thermal_absorptance
        self.solar_absorptance = solar_absorptance
        self.visible_absorptance = visible_absorptance
        self.reflectivity = self._average_reflectivity()

    def _average_reflectivity(self) -> float:
        return 1 - (self.solar_absorptance + self.visible_absorptance) / 2

    def to_hb(self) -> RadianceProperties:
        """ Convert Material into Honeybee Radiance material property object """
        return RadianceProperties(
            material=Plastic(self.name, r_reflectance=self.reflectivity, g_reflectance=self.reflectivity,
                             b_reflectance=self.reflectivity))

    def to_eppy(self, idd_file: str) -> List[EpBunch]:
        """ Convert Material into Eppy material and construction objects """
        try:
            IDF.setiddname(str(idd_file))
        except IDDAlreadySetError as e:
            pass
        idf = IDF(io.StringIO(""))

        # Create list of eppy objects to populate
        eppy_objects = []

        material = idf.newidfobject("MATERIAL")
        material.Name = self.name
        material.Roughness = self.roughness
        material.Thickness = self.thickness
        material.Conductivity = self.conductivity
        material.Density = self.density
        material.Specific_Heat = self.specific_heat
        material.Thermal_Absorptance = self.thermal_absorptance
        material.Solar_Absorptance = self.solar_absorptance
        material.Visible_Absorptance = self.visible_absorptance
        eppy_objects.append(material)

        construction = idf.newidfobject("CONSTRUCTION")
        construction.Name = self.name
        construction.Outside_Layer = self.name
        eppy_objects.append(construction)

        return eppy_objects

    def __repr__(self) -> str:
        return super().__repr__()


class Water(MaterialBase):
    """ Water ground material object

    Parameters
    ----------
    thickness : float
        Thickness of material in m
    conductivity : float
        Conductivity of material in W/m·K
    solar_transmittance : float
        Density of material in kg/m3
    solar_reflectance : float
        Specific heat capcity of material in J/kg·K
    visible_transmittance : float
        Emissivity of material (value between 0 and 1)
    visible_reflectance : float
        Proportion of incident radiation absorbed by material (value between 0 and 1)
    emissivity : float
        Proportion of light absorbed by material (value between 0 and 1)
    """

    def __init__(self, thickness: float = 0.5, solar_transmittance: float = 0.837, solar_reflectance: float = 0.075,
                 visible_transmittance: float = 0.898, visible_reflectance: float = 0.081, conductivity: float = 0.9,
                 emissivity: float = 0.96):
        super().__init__()
        self.material_type = self.__class__.__name__
        self.thickness = thickness
        self.solar_transmittance = solar_transmittance
        self.solar_reflectance = solar_reflectance
        self.visible_transmittance = visible_transmittance
        self.visible_reflectance = visible_reflectance
        self.conductivity = conductivity
        self.emissivity = emissivity
        self.reflectivity = self._average_reflectivity()
        self.transmittance = self._average_transmittance()

    def _average_reflectivity(self) -> float:
        return 1 - (self.solar_reflectance + self.visible_reflectance) / 2

    def _average_transmittance(self) -> float:
        return (self.solar_transmittance + self.visible_transmittance) / 2

    def to_hb(self) -> RadianceProperties:
        return RadianceProperties(
            material=Trans(name=self.name, r_reflectance=self.reflectance, g_reflectance=self.reflectance,
                           b_reflectance=self.reflectance, transmitted_diff=self.transmittance,
                           transmitted_spec=self.transmittance))

    def to_eppy(self, idd_file: str) -> List[EpBunch]:
        """ Convert Material into Eppy material and construction objects """
        try:
            IDF.setiddname(str(idd_file))
        except IDDAlreadySetError as e:
            pass
        idf = IDF(io.StringIO(""))

        # Create list of eppy objects to populate
        eppy_objects = []

        material = idf.newidfobject("WINDOWMATERIAL:GLAZING")
        material.Name = self.name
        material.Optical_Data_Type = "SpectralAverage"
        material.Thickness = self.thickness
        material.Solar_Transmittance_at_Normal_Incidence = self.solar_transmittance
        material.Front_Side_Solar_Reflectance_at_Normal_Incidence = self.solar_reflectance
        material.Back_Side_Solar_Reflectance_at_Normal_Incidence = self.solar_reflectance
        material.Visible_Transmittance_at_Normal_Incidence = self.visible_transmittance
        material.Front_Side_Visible_Reflectance_at_Normal_Incidence = self.visible_reflectance
        material.Back_Side_Visible_Reflectance_at_Normal_Incidence = self.visible_reflectance
        material.Infrared_Transmittance_at_Normal_Incidence = self.transmittance
        material.Front_Side_Infrared_Hemispherical_Emissivity = self.emissivity
        material.Back_Side_Infrared_Hemispherical_Emissivity = self.emissivity
        material.Conductivity = self.conductivity
        material.Solar_Diffusing = False
        eppy_objects.append(material)

        construction = idf.newidfobject("CONSTRUCTION")
        construction.Name = self.name
        construction.Outside_Layer = self.name
        eppy_objects.append(construction)

        return eppy_objects

    def __repr__(self):
        return super().__repr__()


# DEFAULTS
material_dict = {
    "FABRIC": Hardscape(roughness="Smooth", thickness=0.001, conductivity=45.28, density=7824, specific_heat=500,
                        thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7),
    "WATER": Water(thickness=0.5, solar_transmittance=0.837, solar_reflectance=0.075, visible_transmittance=0.898,
                   visible_reflectance=0.081, conductivity=0.9, emissivity=0.96),
    "GRASS": Softscape(roughness="Rough", thickness=0.2, conductivity=1.0, density=1250, specific_heat=1252,
                       thermal_absorptance=0.92, solar_absorptance=0.75, visible_absorptance=0.75, plant_height=0.05,
                       leaf_area_index=1.71, leaf_reflectivity=0.19, leaf_emissivity=0.95,
                       minimum_stomatal_resistance=180, soil_layer_name="Soil",
                       saturation_volumetric_moisture_content_of_the_soil_layer=0.3,
                       residual_volumetric_moisture_content_of_the_soil_layer=0.01,
                       initial_volumetric_moisture_content_of_the_soil_layer=0.1),
    "SHRUBS": Softscape(roughness="Rough", thickness=0.2, conductivity=0.5, density=1600, specific_heat=1026,
                        thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7, plant_height=0.25,
                        leaf_area_index=2.08, leaf_reflectivity=0.21, leaf_emissivity=0.95,
                        minimum_stomatal_resistance=180, soil_layer_name="Soil",
                        saturation_volumetric_moisture_content_of_the_soil_layer=0.3,
                        residual_volumetric_moisture_content_of_the_soil_layer=0.01,
                        initial_volumetric_moisture_content_of_the_soil_layer=0.1),
    "METAL": Hardscape(roughness="Smooth", thickness=0.001, conductivity=45.28, density=7824, specific_heat=500,
                       thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7),
    "SAND": Hardscape(roughness="Rough", thickness=0.2, conductivity=0.33, density=1555, specific_heat=800,
                      thermal_absorptance=0.85, solar_absorptance=0.65, visible_absorptance=0.65),
    "SANDSTONE": Hardscape(roughness="MediumRough", thickness=0.2, conductivity=6.2, density=2560, specific_heat=790,
                           thermal_absorptance=0.6, solar_absorptance=0.55, visible_absorptance=0.55),
    "LIMESTONE": Hardscape(roughness="MediumRough", thickness=0.2, conductivity=3.2, density=2560, specific_heat=790,
                           thermal_absorptance=0.96, solar_absorptance=0.55, visible_absorptance=0.55),
    "HARDWOOD": Hardscape(roughness="MediumSmooth", thickness=0.1, conductivity=0.167, density=680, specific_heat=1630,
                          thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7),
    "SOFTWOOD": Hardscape(roughness="MediumSmooth", thickness=0.1, conductivity=0.129, density=496, specific_heat=1630,
                          thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7),
    "CONCRETE": Hardscape(roughness="MediumRough", thickness=0.2, conductivity=1.73, density=2243, specific_heat=837,
                          thermal_absorptance=0.9, solar_absorptance=0.65, visible_absorptance=0.65),
    "ASPHALT": Hardscape(roughness="MediumRough", thickness=0.2, conductivity=0.75, density=2360, specific_heat=920,
                         thermal_absorptance=0.93, solar_absorptance=0.87, visible_absorptance=0.87),
    "INTERFACE": Hardscape(roughness="MediumRough", thickness=0.2, conductivity=5.0, density=1000, specific_heat=1000,
                           thermal_absorptance=0.9, solar_absorptance=0.7, visible_absorptance=0.7)
}
