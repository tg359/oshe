import oshe as tc
idd_file = r"C:\openstudio-2.7.0\EnergyPlus\Energy+.idd"

def gen_stuff(key):
    temp = tc.ground.Ground(ground_type=key)
    temp.utci(
        epw_file=r"C:\Users\tgerrish\Desktop\MEX_JAL_Puerto.Vallarta-Ordaz.Intl.AP.766013_TMYx.epw",
        idd_file=idd_file,
        case_name=key
    )
    return temp

def mrt_parallel(val_dict):
    mean_radiant_temperature_part = tc.mrt.mean_radiant_temperature(
        surrounding_surfaces_temperature=val_dict["srftmp"],
        horizontal_infrared_radiation_intensity=val_dict["hrzifr"], 
        diffuse_horizontal_solar=val_dict["difrad"], 
        direct_normal_solar=val_dict["dirrad"], 
        sun_altitude=val_dict["solalt"], 
        ground_reflectivity=val_dict["gndref"],
        sky_exposure=val_dict["skyexp"], 
        radiance=True
    )[0]
    return mean_radiant_temperature_part

def utci_parallel(val_dict):
    universal_thermal_climate_index_part = tc.utci.universal_thermal_climate_index(
        val_dict["dbt"],
        val_dict["mrt"],
        val_dict["ws"],
        val_dict["rh"]
    )
    return universal_thermal_climate_index_part

def pl_plt(val_dict):
    val_dict["obj"].plot_plan(val_dict["rad_files"], _type=val_dict["t"], day_period=val_dict["d_period"], season_period=val_dict["s_period"], save_path=val_dict["sp"], pts=False, label_pts=None, legend=False, highlight_pt=None, clip=val_dict["boundary"], tone_color="k", close=True)