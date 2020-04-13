from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
import matplotlib.pyplot as plt
from matplotlib import dates, ticker
import pandas as pd
import numpy as np
from matplotlib.patches import Patch, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from .helpers import chunk


class UTCI(object):

    def __init__(self, utci_openfield, utci_points, points, comfort_lim=[8.5, 28.5], reduction_lim=4):
        self.index = pd.date_range(start="2018-01-01 01:00:00", end="2019-01-01 01:00:00", freq="60T", closed="left")
        self.utci_openfield = utci_openfield  # List of hourly annual UTCI values
        self.utci_points = utci_points  # array of point/hour values for year
        self.points = points  # array of point locations (x, y, z)

        self.df_utci = pd.DataFrame(utci_points.T, index=self.index)
        self.df_utci_reduction = pd.DataFrame((utci_openfield - utci_points).T, index=self.index)

        self.comfort_low, self.comfort_high = comfort_lim
        self.reduction_lim = reduction_lim

        # Filter masks
        self.morning = (self.index.hour >= 7) & (self.index.hour <= 10)
        self.afternoon = (self.index.hour >= 16) & (self.index.hour <= 20)
        self.may = self.index.month == 5
        self.october = self.index.month == 10
        self.openfield_comfort = (utci_openfield >= 9) & (utci_openfield <= 26)
        self.openfield_comfort_plus = (utci_openfield >= 9) & (utci_openfield <= 28)

        # To add using methods

        # Values
        self.may_morning_comfort = None
        self.may_afternoon_comfort = None
        self.may_shoulder_comfort = None
        self.october_morning_comfort = None
        self.october_afternoon_comfort = None
        self.october_shoulder_comfort = None
        self.shoulder_morning_comfort = None
        self.shoulder_afternoon_comfort = None
        self.shoulder_shoulder_comfort = None
        self.annual_comfort = None
        self.annual_morning_comfort = None
        self.annual_afternoon_comfort = None
        self.annual_shoulder_comfort = None
        self.comfort()

        self.may_morning_reduction = None
        self.may_afternoon_reduction = None
        self.may_shoulder_reduction = None
        self.october_morning_reduction = None
        self.october_afternoon_reduction = None
        self.october_shoulder_reduction = None
        self.shoulder_morning_reduction = None
        self.shoulder_afternoon_reduction = None
        self.shoulder_shoulder_reduction = None
        self.annual_reduction = None
        self.annual_morning_reduction = None
        self.annual_afternoon_reduction = None
        self.annual_shoulder_reduction = None
        self.reduction()

        # Plotting geometries
        self.geo_materials = None
        self.geo_data = {}

        # Legend
        self.legend_elements = None

        # Filter masks
        self.masks = {
            "Daily": ((self.index.hour >= 0) & (self.index.hour <= 24)),
            "Morning": ((self.index.hour >= 5) & (self.index.hour <= 10)),
            "Midday": ((self.index.hour >= 11) & (self.index.hour <= 13)),
            "Afternoon": ((self.index.hour >= 14) & (self.index.hour <= 18)),
            "Evening": ((self.index.hour >= 19) & (self.index.hour <= 22)),
            "Night": ((self.index.hour >= 23) | (self.index.hour <= 4)),

            "Annual": (self.index.month <= 12),
            "Spring": ((self.index.month >= 3) & (self.index.month <= 5)),
            "Summer": ((self.index.month >= 6) & (self.index.month <= 8)),
            "Autumn": ((self.index.month >= 9) & (self.index.month <= 11)),
            "Winter": ((self.index.month <= 2) | (self.index.month >= 12)),

            "Morning Shoulder": ((self.index.hour >= 7) & (self.index.hour <= 10)),
            "Afternoon Shoulder": ((self.index.hour >= 16) & (self.index.hour <= 20)),
            "Morning & Afternoon Shoulder": (((self.index.hour >= 7) & (self.index.hour <= 10)) | ((self.index.hour >= 16) & (self.index.hour <= 20))),

            "May": (self.index.month == 5),
            "October": (self.index.month == 10),
            "May & October": ((self.index.month == 5) | (self.index.month == 10)),
        }

        # Colormaps
        # UTCI comfort colormap - UTCI values will be coloured based on their comfort values
        self.utci_cmap = ListedColormap(['#0D104B', '#262972', '#3452A4', '#3C65AF', '#37BCED', '#2EB349', '#F38322', '#C31F25', '#7F1416', '#580002'])
        self.utci_cmap_bounds = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
        self.utci_cmap_norm = BoundaryNorm(self.utci_cmap_bounds, self.utci_cmap.N)
        self.utci_reduction_cmap = LinearSegmentedColormap.from_list("reduction", ["#70339e", "#887f8f", "#ffffff", "#0e7cab", "#00304a"], 100)
        self.utci_comfortfreq_cmap = LinearSegmentedColormap.from_list("comfortfreq", ["#01073D", "#020B61", "#000E8F", "#11258C", "#003EBA", "#005BE3", "#006EE3", "#007CDB", "#009DD1", "#00ACBF", "#07B5AF", "#12B8A7", "#30BA9A", "#74C498", "#92C48D", "#AABF7E", "#BBC282", "#DBD2A7", "#F0E3AF", "#FFFFFF"][::-1], 100)

    def openfield_summary(self):
        summary_vals = np.array([
            np.array([
                (len(self.utci_openfield[self.morning & self.openfield_comfort]) / len(self.utci_openfield[self.morning])) * 100,
                (len(self.utci_openfield[self.afternoon & self.openfield_comfort]) / len(self.utci_openfield[self.afternoon])) * 100,
                (len(self.utci_openfield[self.morning & self.openfield_comfort_plus]) / len(self.utci_openfield[self.morning])) * 100,
                (len(self.utci_openfield[self.afternoon & self.openfield_comfort_plus]) / len(self.utci_openfield[self.afternoon])) * 100
            ]),

            np.array([
                (len(self.utci_openfield[self.may & self.morning & self.openfield_comfort]) / len(self.utci_openfield[self.may & self.morning])) * 100,
                (len(self.utci_openfield[self.may & self.afternoon & self.openfield_comfort]) / len(self.utci_openfield[self.may & self.afternoon])) * 100,
                (len(self.utci_openfield[self.may & self.morning & self.openfield_comfort_plus]) / len(self.utci_openfield[self.may & self.morning])) * 100,
                (len(self.utci_openfield[self.may & self.afternoon & self.openfield_comfort_plus]) / len(self.utci_openfield[self.may & self.afternoon])) * 100,
            ]),
            np.array([
                (len(self.utci_openfield[self.october & self.morning & self.openfield_comfort]) / len(self.utci_openfield[self.october & self.morning])) * 100,
                (len(self.utci_openfield[self.october & self.afternoon & self.openfield_comfort]) / len(self.utci_openfield[self.october & self.afternoon])) * 100,
                (len(self.utci_openfield[self.october & self.morning & self.openfield_comfort_plus]) / len(self.utci_openfield[self.october & self.morning])) * 100,
                (len(self.utci_openfield[self.october & self.afternoon & self.openfield_comfort_plus]) / len(self.utci_openfield[self.october & self.afternoon])) * 100,
            ])
        ]).T

        return pd.concat([
            pd.Series(name="Open field", data=["Comfort", "Comfort", "Comfort + slight heat stress", "Comfort + slight heat stress"]),
            pd.Series(name="UTCI", data=["9-26", "9-26", "9-28", "9-28"]),
            pd.Series(name="Time", data=["07:00-10:00", "16:00-19:00", "07:00-10:00", "16:00-19:00"]),
            pd.DataFrame(summary_vals, columns=["Annual", "May", "October"])
        ], axis=1)

    def comfort(self):
        mask_comfort = (self.df_utci >= self.comfort_low) & (self.df_utci <= self.comfort_high)
        print("Calculating UTCI comfort levels")
        may_morn = self.df_utci[mask_comfort][self.may & self.morning]
        may_eve = self.df_utci[mask_comfort][self.may & self.afternoon]
        may_shoulder = self.df_utci[mask_comfort][self.may & (self.afternoon | self.morning)]

        oct_morn = self.df_utci[mask_comfort][self.october & self.morning]
        oct_eve = self.df_utci[mask_comfort][self.october & self.afternoon]
        oct_shoulder = self.df_utci[mask_comfort][self.october & (self.afternoon | self.morning)]

        shoulder_morn = self.df_utci[mask_comfort][(self.may | self.october) & self.morning]
        shoulder_eve = self.df_utci[mask_comfort][(self.may | self.october) & self.afternoon]
        shoulder_shoulder = self.df_utci[mask_comfort][(self.may | self.october) & (self.afternoon | self.morning)]

        annual = self.df_utci[mask_comfort]
        annual_morn = self.df_utci[mask_comfort][self.morning]
        annual_eve = self.df_utci[mask_comfort][self.afternoon]
        annual_shoulder = self.df_utci[mask_comfort][self.afternoon | self.morning]

        self.may_morning_comfort = (may_morn.count() / may_morn.shape[0] * 100).values
        self.may_afternoon_comfort = (may_eve.count() / may_eve.shape[0] * 100).values
        self.may_shoulder_comfort = (may_shoulder.count() / may_shoulder.shape[0] * 100).values

        self.october_morning_comfort = (oct_morn.count() / oct_morn.shape[0] * 100).values
        self.october_afternoon_comfort = (oct_eve.count() / oct_eve.shape[0] * 100).values
        self.october_shoulder_comfort = (oct_shoulder.count() / oct_shoulder.shape[0] * 100).values

        self.shoulder_morning_comfort = (shoulder_morn.count() / shoulder_morn.shape[0] * 100).values
        self.shoulder_afternoon_comfort = (shoulder_eve.count() / shoulder_eve.shape[0] * 100).values
        self.shoulder_shoulder_comfort = (shoulder_shoulder.count() / shoulder_shoulder.shape[0] * 100).values

        self.annual_comfort = (annual.count() / annual.shape[0] * 100).values
        self.annual_morning_comfort = (annual_morn.count() / annual_morn.shape[0] * 100).values
        self.annual_afternoon_comfort = (annual_eve.count() / annual_eve.shape[0] * 100).values
        self.annual_shoulder_comfort = (annual_shoulder.count() / annual_shoulder.shape[0] * 100).values
        
    def reduction(self):
        mask_reduction = self.df_utci_reduction <= -self.reduction_lim
        print("Calculating UTCI openfield reduction levels")
        may_morn = self.df_utci[mask_reduction][self.may & self.morning]
        may_eve = self.df_utci[mask_reduction][self.may & self.afternoon]
        may_shoulder = self.df_utci[mask_reduction][self.may & (self.afternoon | self.morning)]

        oct_morn = self.df_utci[mask_reduction][self.october & self.morning]
        oct_eve = self.df_utci[mask_reduction][self.october & self.afternoon]
        oct_shoulder = self.df_utci[mask_reduction][self.october & (self.afternoon | self.morning)]

        shoulder_morn = self.df_utci[mask_reduction][(self.may | self.october) & self.morning]
        shoulder_eve = self.df_utci[mask_reduction][(self.may | self.october) & self.afternoon]
        shoulder_shoulder = self.df_utci[mask_reduction][(self.may | self.october) & (self.afternoon | self.morning)]

        annual = self.df_utci[mask_reduction]
        annual_morn = self.df_utci[mask_reduction][self.morning]
        annual_eve = self.df_utci[mask_reduction][self.afternoon]
        annual_shoulder = self.df_utci[mask_reduction][self.afternoon | self.morning]

        self.may_morning_reduction = (may_morn.count() / may_morn.shape[0] * 100).values
        self.may_afternoon_reduction = (may_eve.count() / may_eve.shape[0] * 100).values
        self.may_shoulder_reduction = (may_shoulder.count() / may_shoulder.shape[0] * 100).values

        self.october_morning_reduction = (oct_morn.count() / oct_morn.shape[0] * 100).values
        self.october_afternoon_reduction = (oct_eve.count() / oct_eve.shape[0] * 100).values
        self.october_shoulder_reduction = (oct_shoulder.count() / oct_shoulder.shape[0] * 100).values

        self.shoulder_morning_reduction = (shoulder_morn.count() / shoulder_morn.shape[0] * 100).values
        self.shoulder_afternoon_reduction = (shoulder_eve.count() / shoulder_eve.shape[0] * 100).values
        self.shoulder_shoulder_reduction = (shoulder_shoulder.count() / shoulder_shoulder.shape[0] * 100).values

        self.annual_reduction = (annual.count() / annual.shape[0] * 100).values
        self.annual_morning_reduction = (annual_morn.count() / annual_morn.shape[0] * 100).values
        self.annual_afternoon_reduction = (annual_eve.count() / annual_eve.shape[0] * 100).values
        self.annual_shoulder_reduction = (annual_shoulder.count() / annual_shoulder.shape[0] * 100).values

    def load_geometries(self, rad_files):
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
        self.geo_materials = list(np.unique(materials))

        # Sort the vertices and materials to populate the dataset
        for ii in self.geo_materials:
            self.geo_data[ii] = []
        for v, m in list(zip(*[vertices, materials])):
            self.geo_data[m].append(list(v))

    def generate_legend(self, material_properties=None):

        if material_properties is None:
            material_properties = {
                "BDG_CONTEXT": {"alpha": 1.0, "color": "#BEBEBE", "zorder": 2, "linewidth": 1.0, "edgecolor": "#BEBEBE",
                                "linestyle": "-", "name": "Building", "hatch": "|"},
                "GND_ACTIVEPATHWAY": {"alpha": 1.0, "color": "#FFBF00", "zorder": 1, "linewidth": 0.25,
                                      "edgecolor": "#FFBF00", "linestyle": ":", "name": "Active pathway",
                                      "hatch": "\\"},
                "GND_ASPHALT": {"alpha": 1.0, "color": "#696969", "zorder": 1, "linewidth": 0.25,
                                "edgecolor": "#696969", "linestyle": ":", "name": "Asphalt", "hatch": "/"},
                "GND_CONCRETE": {"alpha": 1.0, "color": "#BEBEBE", "zorder": 1, "linewidth": 0.25,
                                 "edgecolor": "#BEBEBE", "linestyle": ":", "name": "Concrete", "hatch": "-"},
                "GND_GRASS": {"alpha": 1.0, "color": "#005800", "zorder": 1, "linewidth": 0.25, "edgecolor": "#005800",
                              "linestyle": ":", "name": "Grass", "hatch": "+"},
                "GND_PLAY": {"alpha": 1.0, "color": "#BF3F3F", "zorder": 1, "linewidth": 0.25, "edgecolor": "#BF3F3F",
                             "linestyle": ":", "name": "Playground", "hatch": "x"},
                "GND_SAND": {"alpha": 1.0, "color": "#FFEDD3", "zorder": 1, "linewidth": 0.25, "edgecolor": "#FFEDD3",
                             "linestyle": ":", "name": "Sand", "hatch": "o"},
                "GND_STONE": {"alpha": 1.0, "color": "#FFDEAD", "zorder": 1, "linewidth": 0.25, "edgecolor": "#FFDEAD",
                              "linestyle": ":", "name": "Stone", "hatch": "O"},
                "GND_SURROUNDING": {"alpha": 0.0, "color": "#000000", "zorder": 1, "linewidth": 0.25,
                                    "edgecolor": "#000000", "linestyle": ":", "name": "Underground", "hatch": "."},
                "GND_VEGETATION": {"alpha": 1.0, "color": "#6C9400", "zorder": 1, "linewidth": 0.25,
                                   "edgecolor": "#6C9400", "linestyle": ":", "name": "Planting", "hatch": "+"},
                "GND_WATER": {"alpha": 1.0, "color": "#3FBFBF", "zorder": 1, "linewidth": 0.25, "edgecolor": "#3FBFBF",
                              "linestyle": ":", "name": "Water", "hatch": "|"},
                "SHD_P1_POROUS": {"alpha": 0.4, "color": "#F0F0F0", "zorder": 3, "linewidth": 1.0,
                                  "edgecolor": "#F0F0F0", "linestyle": "-", "name": "Porous shade", "hatch": "\\"},
                "SHD_P2_POROUS": {"alpha": 0.4, "color": "#F0F0F0", "zorder": 3, "linewidth": 1.0,
                                  "edgecolor": "#F0F0F0", "linestyle": "-", "name": "Porous shade", "hatch": "x"},
                "SHD_SOLID": {"alpha": 0.8, "color": "#696969", "zorder": 3, "linewidth": 1.0, "edgecolor": "#696969",
                              "linestyle": "-", "name": "Solid shade", "hatch": "x"},
                "VEG_GHAF": {"alpha": 0.3, "color": "#007F00", "zorder": 4, "linewidth": 1.0, "edgecolor": "#007F00",
                             "linestyle": "-", "name": "Ghaf (style) tree", "hatch": "*"},
                "VEG_PALM": {"alpha": 0.3, "color": "#3FBF7F", "zorder": 4, "linewidth": 1.0, "edgecolor": "#3FBF7F",
                             "linestyle": "-", "name": "Palm (style) tree", "hatch": "*"},
            }
        self.geo_materials = material_properties

        # TODO - this method is only rleevent for Airport Rd and should be changed to be generic, though setting up this detail would be a pain!
        # Create custom legend for geometry
        self.legend_elements = []
        for k, v in material_properties.items():
            if (k == "SHD_P1_POROUS") | (k == "GND_SURROUNDING"):
                continue
            self.legend_elements.append(Patch(facecolor=v["color"], edgecolor=None, label=v["name"]))

    def get_x(self):
        return self.points.T[0]

    def get_y(self):
        return self.points.T[1]

    def utci_day_comparison(self, point_idx, period="Annual", tone_color="k", save_path=None, close=False):

        time_mask = self.masks[period]
        masked_idx = self.index[time_mask]
        locator = dates.HourLocator(byhour=np.arange(0, 25, 3))
        minor_locator = dates.HourLocator(byhour=np.arange(0, 25, 1))
        formatter = dates.DateFormatter('%H:%M')

        idx = pd.date_range("2018-{0:02.0f}-15 00:00:00".format(5), "2018-{0:02.0f}-16 00:00:00".format(5), freq="60T", closed="left")
        pt = pd.Series(index=idx, data=self.df_utci[point_idx][time_mask].groupby([masked_idx.hour]).median().values)
        of = pd.Series(index=idx, data=pd.Series(self.utci_openfield, index=self.index)[time_mask].groupby([masked_idx.hour]).median().values)

        # Instantiate plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

        # Plot point UTCI values
        ax.plot(pt, label="Point {}".format(point_idx), c="#00A4E2", lw=2, zorder=2)

        # Plot openfield UTCI values
        ax.plot(of, label="Open field", c="#FF3E3E", lw=2, zorder=2)

        # Set plot limits
        ylims = [min([pt.min(), of.min()]) - 3, max([pt.max(), of.max()]) + 3]
        ax.set_ylim(ylims)

        # Add vertical difference lines at periods
        for hr in [7, 11, 16, 20]:
            ax.vlines(pt.index[hr], ymin=pt.values[hr], ymax=of.values[hr], colors="#555555", lw=2, ls="-", zorder=1)
            ax.vlines(pt.index[hr], ymin=np.max([pt.values[hr], of.values[hr]]), ymax=ax.get_ylim()[1] * 0.97, colors="#555555", lw=2, ls="-", zorder=1, alpha=0.2)
            ax.text(pt.index[hr], ax.get_ylim()[1], "{0:+0.1f}°".format(pt.values[hr] - of.values[hr]), ha="center", va="top", color=tone_color, size="small")

        # Format plot area
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(minor_locator)
        [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
        [ax.spines[j].set_color(tone_color) for j in ['bottom', 'left']]
        ax.set_xlim([pt.index.min(), pt.index.max()])
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
        title = "Average diurnal UTCI - {0:} - Point {1:}".format(period, point_idx)
        ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

        # Tidy
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    # def plot_materials(self, rad_files, _type=None, tone_color="k", save_path=None, close=False):
    #
    #     # Load geometries into attribute
    #     self.load_geometries(rad_files)
    #     # Load legend and material properties into attributes
    #     self.generate_legend()
    #
    #     x, y = self.get_x(), self.get_y()
    #
    #     # Instantiate figure and axis and format most of it
    #     fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    #     ax.axis('off')
    #     ax.set_aspect("equal")
    #     ax.set_xlim(x.min() - 3, x.max() + 3)
    #     ax.set_ylim(y.min() - 3, y.max() + 3)
    #
    #     # Add linework version of ground and context patches
    #     patches = []
    #     for k, v in self.geo_data.items():
    #         for verts in v:
    #             if k == _type:
    #                 # Fill patch
    #                 patches.append(Polygon(verts, closed=True, fill=True, fc="#000000", zorder=self.geo_materials[k]["zorder"], alpha=self.geo_materials[k]["alpha"], ec="#D1D1D1",
    #                                    lw=self.geo_materials[k]["linewidth"], ls=self.geo_materials[k]["linestyle"]))
    #         else:
    #             patches.append(Polygon(verts, closed=True, fill=True, fc=None, hatch=self.geo_materials[k]["hatch"], ec="#D1D1D1",
    #                                    lw=self.geo_materials[k]["linewidth"], ls="-",
    #                                    zorder=8, alpha=1))
    #
    #     p = PatchCollection(patches, match_original=True)
    #     ax.add_collection(p)
    #
    #     ax.set_title("{}".format(self.geo_materials[_type]["name"]), color=tone_color, x=0, ha="left", va="bottom")
    #
    #     plt.tight_layout()
    #
    #     if save_path:
    #         fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
    #         print("Plot saved to {}".format(save_path))
    #     if close:
    #         plt.close()


    def plot_plan(self, rad_files, _type=None, day_period="Daily", season_period="Annual", pts=False, label_pts=None, highlight_pt=None, legend=False, tone_color="k", save_path=None, close=False):

        # Load geometries into attribute
        self.load_geometries(rad_files)
        # Load legend and material properties into attributes
        self.generate_legend()

        x, y = self.get_x(), self.get_y()

        # Instantiate figure and axis and format most of it
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.axis('off')
        ax.set_aspect("equal")
        ax.set_xlim(x.min() - 3, x.max() + 3)
        ax.set_ylim(y.min() - 3, y.max() + 3)

        # Define filters for year/day time periods
        time_mask = np.array([self.masks[day_period], self.masks[season_period]]).all(axis=0)

        # Add contourplot of type indicated
        if _type is not None:
            if _type == "comfort":
                cmap = self.utci_comfortfreq_cmap
                z = (self.df_utci[((self.df_utci >= self.comfort_low) & (self.df_utci <= self.comfort_high))][time_mask].count() / time_mask.sum())
                tcf = ax.tricontourf(x, y, z, cmap=cmap, levels=np.linspace(0, 0.7, 100), zorder=1, extend="max", alpha=1)
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=11))
                cb.ax.set_yticklabels(['{:.0%}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Frequency', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                ax.set_title("UTCI comfort (9-28°C)\n{} - {}".format(season_period, day_period), x=0, ha="left", va="bottom", color=tone_color)
            elif _type == "reduction":
                cmap = self.utci_reduction_cmap
                z = self.df_utci_reduction[time_mask].quantile(0.8, axis=0).values
                tcf = ax.tricontourf(x, y, z, cmap=cmap, levels=np.linspace(-10, 10, 100), zorder=1, extend="both", alpha=1)
                tc = ax.tricontour(x, y, z, colors=tone_color, linewidths=0.5, linestyles="--", levels=[4], zorder=10, alpha=1)
                tcl = ax.clabel(tc, colors=tone_color, inline=True, fontsize="small", fmt="%0.0f")
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=10))
                cb.ax.set_yticklabels(['{:.1f}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Reduction', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                ax.set_title("Mean UTCI reduction\n{} - {}".format(season_period, day_period), x=0, ha="left", va="bottom", color=tone_color)
            elif _type == "improvement":
                cmap = self.utci_comfortfreq_cmap
                z = self.df_utci_reduction[time_mask]
                z = z[z > 3.5].count() / z.count()
                tcf = ax.tricontourf(x, y, z, cmap=cmap, levels=np.linspace(0, 0.7, 100), zorder=1, extend="both", alpha=1)
                # tc = ax.tricontour(x, y, z, colors=tone_color, linewidths=0.5, linestyles="--", levels=10, zorder=10, alpha=1)
                # tcl = ax.clabel(tc, colors=tone_color, inline=True, fontsize="small", fmt="%0.0f")
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=10))
                cb.ax.set_yticklabels(['{:.0%}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Improvement', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                ax.set_title("UTCI improvement\n{} - {}".format(season_period, day_period), x=0, ha="left", va="bottom", color=tone_color)


        patches = []
        for k, v in self.geo_data.items():
            for verts in v:
                patches.append(Polygon(verts, closed=True, fill=False, fc=None, ec="#D1D1D1", lw=self.geo_materials[k]["linewidth"], ls=self.geo_materials[k]["linestyle"], zorder=8, alpha=0.25))
                if legend:
                    # Add geometry filled patches legend
                    lgd = ax.legend(handles=self.legend_elements, loc='center left', bbox_to_anchor=(1, 0.5, 1., 0), frameon=False)
                    [plt.setp(i, color=tone_color) for i in lgd.get_texts()]
                    patches.append(Polygon(verts, closed=True, fill=True, fc=self.geo_materials[k]["color"], ec=self.geo_materials[k]["edgecolor"], lw=self.geo_materials[k]["linewidth"], ls=self.geo_materials[k]["linestyle"], zorder=self.geo_materials[k]["zorder"], alpha=self.geo_materials[k]["alpha"]))
        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)

        if pts:
            # Add sample point locations
            ax.scatter(x, y, c=tone_color, s=0.75, marker="x", alpha=0.75, zorder=9)

            if label_pts is not None:
                # Add sample point labels
                for n, (xx, yy) in enumerate(list(zip(*[x, y]))):
                    if (n % label_pts == 0):
                        ax.text(xx, yy, n, fontsize=5, ha="left", va="bottom", zorder=10, color=tone_color)

        if highlight_pt is not None:
            for p in highlight_pt:
                # Highlight a sample point
                ax.add_patch(plt.Circle((x[p], y[p]), radius=2, fc="#D50032", ec="w", lw=1, zorder=11, alpha=0.9))
                ax.annotate(p, xy=(x[p], y[p]), fontsize="x-small", ha="center", va="center", zorder=12, color="w", weight="bold")

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    def utci_reduction_heatmap(self, point_idx, vrange=[-10, 10], title_id=None, tone_color="k", invert_y=False, save_path=None, close=False):



        # Create pivotable dataframe
        idx = pd.date_range(start="2018-01-01 00:00:00", end="2019-01-01 00:00:00", freq="60T", closed="left")
        df = pd.Series(self.utci_openfield - self.utci_points[point_idx], name="Universal Thermal Climate Index", index=idx).to_frame()

        # Data plotting
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        heatmap = ax.imshow(
            pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
            cmap=self.utci_reduction_cmap,
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
        cb = fig.colorbar(heatmap, cmap=self.utci_reduction_cmap, orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.125)  #
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color, fontsize="small")
        [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
        cb.ax.set_xlabel("UTCI improvement [°C] (higher is better)", fontsize="medium", color=tone_color)
        cb.ax.xaxis.set_label_position('top')
        cb.outline.set_visible(False)

        # Title
        title = "Universal Thermal Climate Index Improvement - Point {}".format(point_idx)
        ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

        # Tidy up
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    def utci_heatmap(self, point_idx=None, tone_color="k", invert_y=False, save_path=None, close=True):

        if point_idx is None:
            hourly_utci_values = self.utci_openfield
        else:
            hourly_utci_values = self.utci_points[point_idx]

        # Create pivotable dataframe
        idx = pd.date_range(start="2018-01-01 00:00:00", end="2019-01-01 00:00:00", freq="60T", closed="left")
        df = pd.Series(hourly_utci_values, name="Universal Thermal Climate Index", index=idx).to_frame()

        # Data plotting
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        heatmap = ax.imshow(
            pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
            cmap=self.utci_cmap,
            norm=self.utci_cmap_norm,
            aspect='auto',
            interpolation='none',
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
        heatmap.set_clim(-100, 100)
        cb_ticks = [-70, -33.5, -20, -6.5, 4.5, 17.5, 29, 35, 42, 73]
        cb_tick_labels = ["Extreme\nheat stress", "Very strong\nheat stress", "Strong\nheat stress", "Moderate\nheat stress", "No\nthermal stress", "Slight\ncold stress", "Moderate\ncold stress", "Strong\ncold stress", "Very strong\ncold stress", "Extreme\ncold stress"][::-1]

        cb = fig.colorbar(heatmap, cmap=self.utci_cmap, orientation='horizontal', drawedges=False, ticks=cb_ticks, fraction=0.05, aspect=100, pad=0.125)  #
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
        [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
        cb.ax.axes.set_xticklabels(cb_tick_labels, fontsize="small")
        cb.ax.set_xlabel("UTCI comfort category [°C]", fontsize="medium", color=tone_color)
        cb.ax.xaxis.set_label_position('top')
        cb.outline.set_visible(False)

        # Title
        title = "Universal Thermal Climate Index - Point {}".format(point_idx) if point_idx is not None else "Universal Thermal Climate Index"
        ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

        # Tidy up
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()


def utci_comfort_heatmap(hourly_utci_values, highlight_shoulder=False, tone_color="k", invert_y=False, save_path=None, close=True):
    utci_cmap = ListedColormap(
        ['#0D104B', '#262972', '#3452A4', '#3C65AF', '#37BCED', '#2EB349', '#F38322', '#C31F25', '#7F1416', '#580002'])
    utci_cmap_bounds = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
    utci_cmap_norm = BoundaryNorm(utci_cmap_bounds, utci_cmap.N)

    # Create pivotable dataframe
    idx = pd.date_range(start="2018-01-01 00:00:00", end="2019-01-01 00:00:00", freq="60T", closed="left")
    df = pd.Series(hourly_utci_values, name="Universal Thermal Climate Index", index=idx).to_frame()

    # Data plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    heatmap = ax.imshow(
        pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
        cmap=utci_cmap,
        norm=utci_cmap_norm,
        aspect='auto',
        interpolation='none',
        extent=[dates.date2num(df.index.min()), dates.date2num(df.index.max()), 726449, 726450]
    )

    if highlight_shoulder:
        llww = 2
        llss = "--"
        ax.axvline(x=dates.date2num(idx[(idx.month == 5)][0]), color="w", lw=llww, ls=llss)
        ax.axvline(x=dates.date2num(idx[(idx.month == 6)][0]), color="w", lw=llww, ls=llss)
        ax.axvline(x=dates.date2num(idx[(idx.month == 10)][0]), color="w", lw=llww, ls=llss)
        ax.axvline(x=dates.date2num(idx[(idx.month == 11)][0]), color="w", lw=llww, ls=llss)

        ax.axhline(726449 + ((1 / 24) * 7), color="w", lw=llww, ls=llss)
        ax.axhline(726449 + ((1 / 24) * 10), color="w", lw=llww, ls=llss)
        ax.axhline(726449 + ((1 / 24) * 16), color="w", lw=llww, ls=llss)
        ax.axhline(726449 + ((1 / 24) * 19), color="w", lw=llww, ls=llss)

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
    cb_tick_labels = ["Extreme\nheat stress", "Very strong\nheat stress", "Strong\nheat stress", "Moderate\nheat stress", "No\nthermal stress", "Slight\ncold stress", "Moderate\ncold stress", "Strong\ncold stress", "Very strong\ncold stress", "Extreme\ncold stress"][::-1]

    cb = fig.colorbar(heatmap, cmap=utci_cmap, orientation='horizontal', drawedges=False, ticks=cb_ticks, fraction=0.05, aspect=100, pad=0.125)  #
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
    [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
    cb.ax.axes.set_xticklabels(cb_tick_labels, fontsize="small")
    cb.ax.set_xlabel("UTCI comfort category [°C]", fontsize="medium", color=tone_color)
    cb.ax.xaxis.set_label_position('top')
    cb.outline.set_visible(False)

    # Title
    title = "Universal Thermal Climate Index"
    ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

    # Tidy up
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        print("Plot saved to {}".format(save_path))
    if close:
        plt.close()

def generic_heatmap_check(values, cbar=False, tone_color="k", save_path=None):
    # Create pivotable dataframe
    idx = pd.date_range(start="2018-01-01 00:00:00", end="2019-01-01 00:00:00", freq="60T", closed="left")
    df = pd.Series(values, name="test", index=idx).to_frame()

    # Data plotting
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    heatmap = ax.imshow(
        pd.pivot_table(df, index=df.index.time, columns=df.index.date, values=df.columns[0]).values[::-1],
        cmap="Greys",
        aspect='auto',
        interpolation='none',
        extent=[dates.date2num(df.index.min()), dates.date2num(df.index.max()), 726449, 726450]
    )

    # Axes formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(dates.DateFormatter('%b'))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(dates.DateFormatter('%H:%M'))

    # Ticks, labels and spines formatting
    ax.tick_params(labelleft=True, labelbottom=True)
    plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)
    [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]
    ax.grid(b=True, which='major', color='white', linestyle='--', alpha=0.9)
    [tick.set_color(tone_color) for tick in ax.get_yticklines()]
    [tick.set_color(tone_color) for tick in ax.get_xticklines()]

    # Colorbar
    if cbar:
        cb = fig.colorbar(heatmap, cmap="Greys", orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.125)  #
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
        [tick.set_color(tone_color) for tick in cb.ax.axes.get_xticklines()]
        cb.outline.set_visible(False)

    # Tidy up
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
        print("Plot saved to {}".format(save_path))
