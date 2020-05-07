
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib import dates
from matplotlib.patches import Patch, Polygon, Path, PathPatch

from .helpers import chunk

# General housework
pd.plotting.register_matplotlib_converters()


class UTCI(object):
    def __init__(self, utci_openfield, utci, points):

        self.index = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")
        self.utci_openfield = pd.Series(utci_openfield, index=self.index)  # List of hourly annual UTCI values
        self.utci = pd.DataFrame(utci.T, index=self.index)  # array of point/hour values for year
        self.points = pd.DataFrame(points, columns=["x", "y", "z"])  # array of point locations (x, y, z)
        self.utci_difference = pd.DataFrame((utci_openfield - utci).T, index=self.index)  # openfield/points UTCI diff

        # Time filter masks
        self.mask_daily = ((self.index.hour >= 0) & (self.index.hour <= 24))
        self.mask_morning = ((self.index.hour >= 5) & (self.index.hour <= 10))
        self.mask_midday = ((self.index.hour >= 11) & (self.index.hour <= 13))
        self.mask_afternoon = ((self.index.hour >= 14) & (self.index.hour <= 18))
        self.mask_evening = ((self.index.hour >= 19) & (self.index.hour <= 22))
        self.mask_night = ((self.index.hour >= 23) | (self.index.hour <= 4))

        self.mask_annual = (self.index.month <= 12)
        self.mask_spring = ((self.index.month >= 3) & (self.index.month <= 5))
        self.mask_summer = ((self.index.month >= 6) & (self.index.month <= 8))
        self.mask_autumn = ((self.index.month >= 9) & (self.index.month <= 11))
        self.mask_winter = ((self.index.month <= 2) | (self.index.month >= 12))

        self.mask_may = (self.index.month == 5)
        self.mask_october = (self.index.month == 10)
        self.mask_may_october = ((self.index.month == 5) | (self.index.month == 10))

        self.mask_shoulder_morning = ((self.index.hour >= 7) & (self.index.hour <= 10))
        self.mask_shoulder_afternoon = ((self.index.hour >= 16) & (self.index.hour <= 20))
        self.mask_shoulder_day = (((self.index.hour >= 7) & (self.index.hour <= 10)) | ((self.index.hour >= 16) & (self.index.hour <= 20)))

        self.mask_may_afternoon = self.mask_may & self.mask_shoulder_afternoon
        self.mask_may_morning = self.mask_may & self.mask_shoulder_morning
        self.mask_may_shoulder = self.mask_may & self.mask_shoulder_day
        self.mask_october_afternoon = self.mask_october & self.mask_shoulder_afternoon
        self.mask_october_morning = self.mask_october & self.mask_shoulder_morning
        self.mask_october_shoulder = self.mask_october & self.mask_shoulder_day
        self.mask_shoulder_all = self.mask_shoulder_day & self.mask_may_october

    def get_x(self):
        return self.points.T[0]

    def get_y(self):
        return self.points.T[1]

    def comfort_in_period(self, mask, lower=9, upper=28):
        """ Within a masked time period, calculate the proportion of hours within a value range

        Parameters
        ----------
        mask
        lower
        upper

        Returns
        -------

        """

        count = mask.sum()  # Get number of possible values in mask
        masked_data = self.utci[mask]  # Filter the dataset by time

        # Filter the remaining data by comfort
        comfortable_mask = (masked_data >= lower) & (masked_data <= upper)
        filtered_data = masked_data[comfortable_mask]

        count_remaining = filtered_data.count()  # Count number of potential values
        resulting_frequency = count_remaining / count  # Get proportion of comfortable hours in period

        return resulting_frequency

    def reduction_in_period(self, mask, threshold=4):
        """ Within a masked time period, calculate the proportion of hours where a threshold reduction is met

        Parameters
        ----------
        mask
        threshold

        Returns
        -------

        """

        count = mask.sum()  # Get number of possible values in mask
        masked_data = self.utci_difference[mask]  # Filter the dataset by time

        # Filter the remaining data by threshold value
        comfortable_mask = (masked_data >= threshold)
        filtered_data = masked_data[comfortable_mask]

        count_remaining = filtered_data.count()  # Count number of potential values
        resulting_frequency = count_remaining / count  # Get proportion of comfortable hours in period

        return resulting_frequency

    def reduction_summary(self, threshold=4, percentile=0.95):
        """ Calculate the frequency where the threshold reduction is achieved for an nth-percentile hourly values

        Parameters
        ----------
        threshold
        percentile

        Returns
        -------

        """
        morn_str = "07:00 - 10:00"
        aft_str = "16:00 - 19:00"
        varz = [
            [self.mask_shoulder_morning, ["Annual", morn_str]],
            [self.mask_shoulder_afternoon, ["Annual", aft_str]],
            [self.mask_may_morning, ["May", morn_str]],
            [self.mask_may_afternoon, ["May", aft_str]],
            [self.mask_october_morning, ["October", morn_str]],
            [self.mask_october_afternoon, ["October", aft_str]],
        ]

        # Create the "nth percentile" point
        temp = self.utci_difference.quantile(percentile, axis=1)

        # Get the number of hours within specified ranges where threshold reduction is met
        d = {
            "Annual": {morn_str: None, aft_str: None},
            "May": {morn_str: None, aft_str: None},
            "October": {morn_str: None, aft_str: None}
        }
        for i, [j, k] in varz:
            temp_filtered = temp[i]
            d[j][k] = ((temp_filtered >= threshold).sum() / temp_filtered.count())
        return pd.DataFrame.from_dict(d)


def load_radiance_geometries(rad_files, exclude=["GND_SURROUNDING", "generic_wall"], underlay=True):
    """ Load the radiance geometry into a dictionary, containing vertex groups and materials names as key

    Parameters
    ----------
    rad_files
    exclude

    Returns
    -------
    Dictionary containing faces

    """
    # TODO - This method is very messy ... but it works. Needs rethinking almost entirely
    materials = []
    vertices = []
    for rad_file in rad_files:

        # Open RAD file containing geometry
        with open(rad_file, "r") as f:
            opq = [i.strip() for i in f.readlines()]

        # For each RAD geometry object, get the vertices describing the polygon and material name
        for geo in chunk(opq[3:], n=4):
            materials.append(geo[0].split(" ")[0])
            vertices.append(np.array(chunk([float(i) for i in geo[-1].split(" ")[1:]], n=3))[:, :2].tolist())

    # Create list of unique materials
    geo_materials = list(np.unique(materials))

    # Sort the vertices and materials to populate the dataset
    geo_data = {}
    for ii in geo_materials:
        if ii not in exclude:
            geo_data[ii] = {"vertices": []}
    for v, m in list(zip(*[vertices, materials])):
        if m not in exclude:
            geo_data[m]["vertices"].append(list(v))

    # Default material properties - modify these to format the context geometry
    material_properties = {
        "BDG_CONTEXT": {
            "alpha": 1.0, 
            "fc": "#BEBEBE", 
            "zorder": 2, 
            "lw": 1.0, 
            "ec": "#BEBEBE", 
            "ls": "-", 
            "name": "Building"},
        "GND_CONCRETE": {
            "alpha": 1.0, 
            "fc": "#BEBEBE", 
            "zorder": 1, 
            "lw": 0.25, 
            "ec": "#BEBEBE", 
            "ls": ":", 
            "name": "Concrete"},
        "GND_GRASS": {
            "alpha": 1.0, 
            "fc": "#005800", 
            "zorder": 1, 
            "lw": 0.25, 
            "ec": "#005800", 
            "ls": ":", 
            "name": "Grass"},
        "GND_STONE": {
            "alpha": 1.0, 
            "fc": "#FFDEAD", 
            "zorder": 4, 
            "lw": 0.75, 
            "ec": "#FFDEAD", 
            "ls": ":", 
            "name": "Stone"},
        "GND_SAND": {
            "alpha": 1.0,
            "fc": "#FFF2DE",
            "zorder": 4,
            "lw": 0.75,
            "ec": "#FFF2DE",
            "ls": ":",
            "name": "Sand"},
        "GND_WATER": {
            "alpha": 1.0, 
            "fc": "#3FBFBF", 
            "zorder": 1, 
            "lw": 0.25, 
            "ec": "#3FBFBF", 
            "ls": ":", 
            "name": "Water"},
        "SHD_SOLID": {
            "alpha": 0.8, 
            "fc": "#696969", 
            "zorder": 3, 
            "lw": 1.0, 
            "ec": "#696969", 
            "ls": "-", 
            "name": "Solid shade"},
        "VEG_GHAF": {
            "alpha": 0.3, 
            "fc": "#007F00", 
            "zorder": 4, 
            "lw": 1.0, 
            "ec": "#007F00", 
            "ls": "-", 
            "name": "Ghaf (style) tree"
        },
    }

    # Create custom legend for geometry
    for k, v in material_properties.items():
        try:
            geo_data[k]["lgd"] = Patch(facecolor=v["fc"], edgecolor=None, label=v["name"])
            geo_data[k]["fc"] = v["fc"]
            geo_data[k]["name"] = v["name"]
            geo_data[k]["alpha"] = v["alpha"]
            geo_data[k]["ec"] = v["ec"]
            geo_data[k]["lw"] = v["lw"]
            geo_data[k]["ls"] = v["ls"]
            geo_data[k]["zorder"] = v["zorder"]
        except Exception as e:
            print("{} not found in radiance geometry - skipping".format(k))

    # Create patch collection for plotting
    for k, v in geo_data.items():
        patches = []
        for verts in v["vertices"]:
            if underlay:
                patches.append(Polygon(verts, closed=True, fill=False, fc=None, ec="#D1D1D1", lw=1, ls="-", zorder=8, alpha=0.25))
            else:
                patches.append(Polygon(verts, closed=True, fill=True, fc=v["fc"], ec=v["ec"], lw=v["lw"], ls=v["ls"], zorder=v["zorder"], alpha=v["alpha"]))
        geo_data[k]["patches"] = patches

    return geo_data


def utci_comfort_heatmap(hourly_utci_values, tone_color="k", invert_y=False, title=None, save_path=None, close=True):
    utci_cmap = ListedColormap(
        ['#0D104B', '#262972', '#3452A4', '#3C65AF', '#37BCED', '#2EB349', '#F38322', '#C31F25', '#7F1416', '#580002'])
    utci_cmap_bounds = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
    utci_cmap_norm = BoundaryNorm(utci_cmap_bounds, utci_cmap.N)

    # Create pivotable dataframe
    idx = pd.date_range(start="2018-01-01 00:00:00", periods=8760, freq="60T", closed="left")
    df = pd.Series(hourly_utci_values, name="Universal Thermal Climate Index", index=idx).to_frame()

    # Data plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    heatmap = ax.imshow(
        pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
        cmap=utci_cmap, norm=utci_cmap_norm, aspect='auto', interpolation='none',
        extent=[dates.date2num(df.index.min()), dates.date2num(df.index.max()), 726449, 726450])

    # Axes formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    if invert_y:
        ax.invert_yaxis()

    # Ticks, labels and spines formatting
    ax.tick_params(labelleft=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)
    [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]
    ax.grid(b=True, which='major', color='white', linestyle='--', alpha=0.9)
    [tick.set_color(tone_color) for tick in ax.get_yticklines()]
    [tick.set_color(tone_color) for tick in ax.get_xticklines()]

    # Colorbar
    heatmap.set_clim(-100, 100)
    cb_ticks = [-70, -33.5, -20, -6.5, 4.5, 17.5, 29, 35, 42, 73]
    cb_tick_labels = [
                         "Extreme\nheat stress", "Very strong\nheat stress", "Strong\nheat stress",
                         "Moderate\nheat stress", "No\nthermal stress", "Slight\ncold stress", "Moderate\ncold stress",
                         "Strong\ncold stress", "Very strong\ncold stress", "Extreme\ncold stress"
                     ][::-1]

    cb = fig.colorbar(heatmap, cmap=utci_cmap, orientation='horizontal', drawedges=False,
                      ticks=cb_ticks, fraction=0.05, aspect=100, pad=0.125)
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
    [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
    cb.ax.axes.set_xticklabels(cb_tick_labels, fontsize="small")
    cb.ax.set_xlabel("UTCI comfort category [°C]", fontsize="medium", color=tone_color)
    cb.ax.xaxis.set_label_position('top')
    cb.outline.set_visible(False)

    # Title
    title = "Universal Thermal Climate Index" if title is None else title
    ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

    # Tidy up
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        print("Plot saved to {}".format(save_path))
    if close:
        plt.close()


def utci_reduction_heatmap(hourly_utci_values, vrange=[-10, 10], title=None, tone_color="k", invert_y=False, save_path=None, close=False):

    utci_reduction_cmap = LinearSegmentedColormap.from_list("reduction", ["#70339e", "#887f8f", "#ffffff", "#0e7cab", "#00304a"], 100)

    # Create pivotable dataframe
    idx = pd.date_range(start="2018-01-01 00:00:00", freq="60T", periods=8760, closed="left")
    df = pd.Series(hourly_utci_values, name="Universal Thermal Climate Index", index=idx).to_frame()

    # Data plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    heatmap = ax.imshow(
        pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
        cmap=utci_reduction_cmap,
        aspect='auto',
        interpolation='none',
        vmin=vrange[0],
        vmax=vrange[1],
        extent=[dates.date2num(df.index.min()), dates.date2num(df.index.max()), 726449, 726450]
    )

    # Axes formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    if invert_y:
        ax.invert_yaxis()

    # Ticks, labels and spines formatting
    ax.tick_params(labelleft=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)
    [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]
    ax.grid(b=True, which='major', color='white', linestyle='--', alpha=0.9)
    [tick.set_color(tone_color) for tick in ax.get_yticklines()]
    [tick.set_color(tone_color) for tick in ax.get_xticklines()]

    # Colorbar
    cb = fig.colorbar(heatmap, cmap=utci_reduction_cmap, orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.125)  #
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color, fontsize="small")
    [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
    cb.ax.set_xlabel("UTCI improvement [°C] (higher is better)", fontsize="medium", color=tone_color)
    cb.ax.xaxis.set_label_position('top')
    cb.outline.set_visible(False)

    # Title
    title = "Universal Thermal Climate Index Difference" if title is None else title
    ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

    # Tidy up
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        print("Plot saved to {}".format(save_path))
    if close:
        plt.close()


def utci_day_comparison(hourly_utci_values_a, hourly_utci_values_b, months=np.arange(1, 13, 1), names=["A", "B"], title=None, tone_color="k", save_path=None, close=False):

    day_idx = pd.date_range("2018-05-15 00:00:00", freq="60T", periods=24, closed="left")
    annual_idx = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")
    annual_mask = annual_idx.month.isin(months)
    masked_idx = annual_idx[annual_mask]

    # Create series to plot
    a = pd.Series(index=day_idx, data=pd.Series(hourly_utci_values_a[annual_mask]).groupby([masked_idx.hour]).quantile(0.5).values)
    b = pd.Series(index=day_idx, data=pd.Series(hourly_utci_values_b[annual_mask]).groupby([masked_idx.hour]).quantile(0.5).values)

    # Plotting
    locator = dates.HourLocator(byhour=np.arange(0, 25, 3))
    minor_locator = dates.HourLocator(byhour=np.arange(0, 25, 1))
    formatter = dates.DateFormatter('%H:%M')

    # Instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Plot input values
    ax.plot(a, label=names[0], c="#00A4E2", lw=2, zorder=2)
    ax.plot(b, label=names[1], c="#FF3E3E", lw=2, zorder=2)

    # Set plot limits
    ylims = [min([a.min(), b.min()]) - 3, max([a.max(), b.max()]) + 3]
    ax.set_ylim(ylims)

    # Add vertical difference lines at interesting indices
    sorted_diff = (a - b).sort_values(ascending=True)
    for n, l in enumerate([23]):
        i = sorted_diff.index[l]
        j = -sorted_diff.iloc[l]
        k = np.max([a[i], b[i]], axis=0)
        pref = "Max" if n == 0 else "Median" if n == 1 else "aargh what are you doing here!?!"

        ax.vlines(i, ymin=k, ymax=ax.get_ylim()[1] * 0.93, colors="#555555", lw=2, ls="-", zorder=1, alpha=0.2)
        ax.text(i, ax.get_ylim()[1], "{1:}\n{0:+0.1f}°".format(j, pref), ha="center", va="top", color=tone_color, size="small")

    # Format plot area
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
    [ax.spines[j].set_color(tone_color) for j in ['bottom', 'left']]
    ax.set_xlim([day_idx.min(), day_idx.max()])
    ax.grid(b=True, which='minor', axis='both', c=tone_color, ls=':', lw=0.5, alpha=0.1)
    ax.grid(b=True, which='major', axis='both', c=tone_color, ls=':', lw=1, alpha=0.2)
    ax.tick_params(which="both", length=1, color=tone_color)
    plt.setp(ax.get_xticklabels(), color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)
    ax.set_xlabel("Time of day", color=tone_color)
    ax.set_ylabel("UTCI (°C)", color=tone_color)

    # Legend
    lgd = ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right", ncol=2)
    [plt.setp(text, color=tone_color) for text in lgd.get_texts()]

    # Set title
    title = "Average diurnal UTCI" if title is None else title
    ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

    # Tidy
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        print("Plot saved to {}".format(save_path))
    if close:
        plt.close()

    # return a, b


    #

    #
    # # Tidy
    # plt.tight_layout()
    #
    # if save_path:
    #     fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
    #     print("Plot saved to {}".format(save_path))
    # if close:
    #     plt.close()

