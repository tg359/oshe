import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import dates, ticker
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch, Polygon, Path, PathPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .helpers import chunk, flatten, ANNUAL_DATETIME

# General housework
pd.plotting.register_matplotlib_converters()


# TODO - Add method in GH of attributing colors to layers for use in context plots


class UTCI(object):
    """
    An object containing annual hourly UTCI values for an open field, sample point locations, and the corresponding XYZ coordinates for those sample points
    """

    def __init__(self, utci_openfield, utci, points):

        self.index = ANNUAL_DATETIME
        self.utci_openfield = pd.Series(utci_openfield, index=self.index)  # List of hourly annual UTCI values
        self.utci = pd.DataFrame(utci.T, index=self.index)  # array of point/hour values for year
        self.points = pd.DataFrame(points, columns=["x", "y", "z"])  # array of point locations (x, y, z)
        self.utci_difference = pd.DataFrame((utci_openfield - utci).T, index=self.index)  # openfield.py/points UTCI diff

        self.xrange = self.points.x.min() - 5, self.points.x.max() + 5
        self.yrange = self.points.y.min() - 5, self.points.y.max() + 5

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
        self.mask_shoulder_day = (((self.index.hour >= 7) & (self.index.hour <= 10)) | (
                    (self.index.hour >= 16) & (self.index.hour <= 20)))

        self.mask_may_afternoon = self.mask_may & self.mask_shoulder_afternoon
        self.mask_may_morning = self.mask_may & self.mask_shoulder_morning
        self.mask_may_shoulder = self.mask_may & self.mask_shoulder_day
        self.mask_october_afternoon = self.mask_october & self.mask_shoulder_afternoon
        self.mask_october_morning = self.mask_october & self.mask_shoulder_morning
        self.mask_october_shoulder = self.mask_october & self.mask_shoulder_day
        self.mask_may_october_shoulder = self.mask_shoulder_day & self.mask_may_october

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

    def reduction_summary(self, threshold=4, percentile=0.9, focus_pts=None):
        """ Calculate the frequency where the threshold reduction is achieved for an nth-percentile hourly values

        Parameters
        ----------
        threshold
        percentile
        focus_pts

        Returns
        -------

        """
        morn_str = "07:00 - 10:00"
        aft_str = "16:00 - 19:00"

        ## All points
        # Create the "nth percentile" point
        percentile_pt = self.utci.quantile(1 - percentile, axis=1)

        # Get the number of hours within specified ranges where threshold reduction is met
        d = {
            "Annual": {
                morn_str: utci_reduction(self.utci_openfield, percentile_pt, months=np.arange(1, 13, 1),
                                         hours=[7, 8, 9, 10], threshold=threshold),
                aft_str: utci_reduction(self.utci_openfield, percentile_pt, months=np.arange(1, 13, 1),
                                        hours=[16, 17, 18, 19], threshold=threshold)
            },
            "May": {
                morn_str: utci_reduction(self.utci_openfield, percentile_pt, months=[5],
                                         hours=[7, 8, 9, 10], threshold=threshold),
                aft_str: utci_reduction(self.utci_openfield, percentile_pt, months=[5],
                                        hours=[16, 17, 18, 19], threshold=threshold)
            },
            "October": {
                morn_str: utci_reduction(self.utci_openfield, percentile_pt, months=[10],
                                         hours=[7, 8, 9, 10], threshold=threshold),
                aft_str: utci_reduction(self.utci_openfield, percentile_pt, months=[10],
                                        hours=[16, 17, 18, 19], threshold=threshold)
            }
        }
        all_pts = [pd.concat([pd.DataFrame.from_dict(d)], keys=['All sample points ({0:0.0%}ile)'.format(percentile)])] #, names=['Firstlevel']

        ## Focus points

        if focus_pts is not None:
            for fp in focus_pts:
                d = {
                    "Annual": {
                        morn_str: utci_reduction(self.utci_openfield, self.utci[fp], months=np.arange(1, 13, 1),
                                                         hours=[7, 8, 9, 10], threshold=threshold),
                        aft_str: utci_reduction(self.utci_openfield, self.utci[fp], months=np.arange(1, 13, 1),
                                                        hours=[16, 17, 18, 19], threshold=threshold)
                    },
                    "May": {
                        morn_str: utci_reduction(self.utci_openfield, self.utci[fp], months=[5],
                                                         hours=[7, 8, 9, 10], threshold=threshold),
                        aft_str: utci_reduction(self.utci_openfield, self.utci[fp], months=[5],
                                                        hours=[16, 17, 18, 19], threshold=threshold)
                    },
                    "October": {
                        morn_str: utci_reduction(self.utci_openfield, self.utci[fp], months=[10],
                                                         hours=[7, 8, 9, 10], threshold=threshold),
                        aft_str: utci_reduction(self.utci_openfield, self.utci[fp], months=[10],
                                                        hours=[16, 17, 18, 19], threshold=threshold)
                    }
                }
                all_pts.append(pd.concat([pd.DataFrame.from_dict(d)], keys=["Sample point #{0:}".format(fp)]))

        df = (pd.concat(all_pts, axis=0) * 100).round(2)

        return df

    def plot_context(self, rad_files, pts=False, label_pts=None, highlight_pt=None, tone_color="k", save_path=None,
                     close=True):

        geo_data = load_radiance_geometries(rad_files, underlay=False)
        x, y = self.points[["x", "y"]].values.T

        # Instantiate figure and axis and format most of it
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis('off')
        ax.set_aspect("equal")

        patches = []
        for k, v in geo_data.items():
            patches.append(v["patches"])
        p = PatchCollection(flatten(patches), match_original=True)
        ax.add_collection(p)

        # Add geometry filled patches legend
        lgd_elements = []
        for k, v in geo_data.items():
            lgd_elements.append(geo_data[k]["lgd"])
        lgd = ax.legend(handles=lgd_elements, loc='center left', bbox_to_anchor=(1, 0.5, 1., 0), frameon=False)
        [plt.setp(i, color=tone_color) for i in lgd.get_texts()]

        if pts:
            # Add sample point locations
            ax.scatter(x, y, c=tone_color, s=1, marker="x", alpha=0.75, zorder=9)

            if label_pts is not None:
                for n, (xx, yy) in enumerate(list(zip(*[x, y]))):
                    if n % label_pts == 0:
                        ax.text(xx, yy, n, fontsize=2, ha="left", va="bottom", zorder=10, color=tone_color)

        if highlight_pt is not None:
            for p in highlight_pt:
                # Highlight a sample point
                ax.add_patch(plt.Circle((x[p], y[p]), radius=2, fc="#D50032", ec="w", lw=1, zorder=11, alpha=0.9))
                ax.annotate(p, xy=(x[p], y[p]), fontsize="x-small", ha="center", va="center", zorder=12, color="w",
                            weight="bold")

        ax.set_xlim(self.xrange)
        ax.set_ylim(self.yrange)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False, bbox_extra_artists=(lgd,))
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    def plot_contour(self, _type, mask, rad_files=None, clip=None, title=None, tone_color="k", save_path=None,
                     close=True):

        utci_comfortfreq_cmap = LinearSegmentedColormap.from_list("comfortfreq",
                                                                  ["#01073D", "#020B61", "#000E8F", "#11258C",
                                                                   "#003EBA", "#005BE3", "#006EE3", "#007CDB",
                                                                   "#009DD1", "#00ACBF", "#07B5AF", "#12B8A7",
                                                                   "#30BA9A", "#74C498", "#92C48D", "#AABF7E",
                                                                   "#BBC282", "#DBD2A7", "#F0E3AF", "#FFFFFF"][::-1],
                                                                  100)
        utci_reduction_cmap = LinearSegmentedColormap.from_list("reduction",
                                                                ["#70339e", "#887f8f", "#ffffff", "#0e7cab", "#00304a"],
                                                                100)

        x, y = self.points[["x", "y"]].values.T

        # Instantiate figure and axis and format most of it
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.axis('off')
        ax.set_aspect("equal")

        # Define filters for year/day time periods

        mask_count = mask.sum()

        if _type is not None:
            if _type == "comfort":
                temp = self.utci[mask]
                z = temp[((temp >= 8) & (temp <= 28))].count() / mask_count
                tcf = ax.tricontourf(x, y, z, cmap=utci_comfortfreq_cmap, levels=np.linspace(0, 0.7, 100), zorder=1,
                                     extend="max", alpha=1)
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=11))
                cb.ax.set_yticklabels(
                    ['{:.0%}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Frequency', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                title = "UTCI comfort (9-28°C)" if title is None else title
                ax.set_title(title, x=0, ha="left", va="bottom", color=tone_color)
            elif _type == "reduction":
                z = self.utci_difference[mask].quantile(0.5, axis=0)
                tcf = ax.tricontourf(x, y, z, cmap=utci_reduction_cmap, levels=np.linspace(-10, 10, 100), zorder=1,
                                     extend="both", alpha=1)
                tc = ax.tricontour(x, y, z, colors=tone_color, linewidths=0.5, linestyles=":",
                                   levels=[-4, -3, -2, -1, 0, 1, 2, 3, 4], zorder=10, alpha=0.5)
                tcl = ax.clabel(tc, colors=tone_color, inline=True, fontsize="x-small", fmt="%0.0f")
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=10))
                cb.ax.set_yticklabels(
                    ['{:.1f}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Reduction', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                title = "Mean UTCI reduction" if title is None else title
                ax.set_title(title, x=0, ha="left", va="bottom", color=tone_color)
            elif _type == "improvement":
                z = self.utci_difference[mask]
                z = z[z > 4].count() / z.count()
                tcf = ax.tricontourf(x, y, z, cmap=utci_comfortfreq_cmap, levels=np.linspace(0, 0.7, 100), zorder=1,
                                     extend="both", alpha=1)
                # tc = ax.tricontour(x, y, z, colors=tone_color, linewidths=0.5, linestyles="--", levels=10, zorder=10, alpha=1)
                # tcl = ax.clabel(tc, colors=tone_color, inline=True, fontsize="small", fmt="%0.0f")
                # Add colorbar
                cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.25)
                cb = plt.colorbar(tcf, cax=cax, ticks=ticker.MaxNLocator(nbins=10))
                cb.ax.set_yticklabels(
                    ['{:.0%}'.format(float(x.get_text().replace("−", "-"))) for x in cb.ax.get_yticklabels()])
                cb.ax.set_title('Improvement', color=tone_color, fontsize="medium", ha="left", va="bottom", x=0)
                cb.outline.set_visible(False)
                plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color, fontsize="medium")
                title = "UTCI improvement" if title is None else title
                ax.set_title(title, x=0, ha="left", va="bottom", color=tone_color)

        if rad_files is not None:
            geo_data = load_radiance_geometries(rad_files, underlay=True)
            # Add underlay
            patches = []
            for k, v in geo_data.items():
                patches.append(v["patches"])
            p = PatchCollection(flatten(patches), match_original=True)
            ax.add_collection(p)

        if clip is not None:
            clip_path = Path(np.vstack([clip, clip[0]]), closed=True)
            clip_patch = PathPatch(clip_path, facecolor=None, alpha=0)
            ax.add_patch(clip_patch)
            for c in tcf.collections:
                c.set_clip_path(clip_patch)

        ax.set_xlim(self.xrange)
        ax.set_ylim(self.yrange)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    def plot_comfortable_hours(self, lower=9, upper=28, months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], percentile=0.95,
                               title=None, tone_color="k", save_path=None, close=True):
        a = get_aggregate_day(self.utci, lower=lower, upper=upper, months=months, percentile=percentile)
        b = get_aggregate_day(self.utci_openfield, lower=lower, upper=upper, months=months, percentile=percentile)

        # Plotting
        locator = dates.HourLocator(byhour=np.arange(0, 25, 3))
        minor_locator = dates.HourLocator(byhour=np.arange(0, 25, 1))
        formatter = dates.DateFormatter('%H:%M')

        # Instantiate plot
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))

        # Plot input values
        ax.plot(a, label="{0:0.0f}%-ile sampled points".format(percentile * 100), c="#00A4E2", lw=2, zorder=2)
        ax.plot(b, label="Open field", c="#FF3E3E", lw=2, zorder=2)

        # Fill between series
        ax.fill_between(a.index, a, b, where=a > b, facecolor="#AFC1A2", alpha=0.25, interpolate=True)
        ax.fill_between(a.index, a, b, where=a < b, facecolor="#E6484D", alpha=0.25, interpolate=True)

        # Set plot limits
        ax.set_ylim(0, 1)

        # Add vertical difference lines at interesting indices
        sorted_diff = (a - b).sort_values(ascending=True)
        for n, l in enumerate([0, 23]):
            i = sorted_diff.index[l]
            j = -sorted_diff.iloc[l]
            k = np.max([a[i], b[i]], axis=0)
            ax.vlines(i, ymin=k, ymax=ax.get_ylim()[1] * 0.96, colors="#555555", lw=2, ls="-", zorder=1, alpha=0.2)
            ax.text(i, ax.get_ylim()[1], "{0:+0.1%}".format(-j), ha="center", va="top", color=tone_color, size="small")

        # Format plot area
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
        [ax.spines[j].set_color(tone_color) for j in ['bottom', 'left']]
        ax.set_xlim([a.index.min(), a.index.max()])
        ax.grid(b=True, which='minor', axis='both', c=tone_color, ls=':', lw=0.5, alpha=0.1)
        ax.grid(b=True, which='major', axis='both', c=tone_color, ls=':', lw=1, alpha=0.2)
        ax.tick_params(which="both", length=1, color=tone_color)
        plt.setp(ax.get_xticklabels(), color=tone_color)
        plt.setp(ax.get_yticklabels(), color=tone_color)
        ax.set_xlabel("Time of day", color=tone_color)
        ax.set_ylabel("% hours within {0:}-{1:} UTCI (°C)".format(lower, upper), color=tone_color)

        # Legend
        lgd = ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc="lower right", ncol=2)
        [plt.setp(text, color=tone_color) for text in lgd.get_texts()]

        # Set title
        title = "Comfortable hours" if title is None else title
        ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0)

        # Tidy
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, transparent=False)
            print("Plot saved to {}".format(save_path))
        if close:
            plt.close()

    def generate_plots(self, rad_files, focus_pts, boundary, plot_directory, tone_color="k"):

        for i in focus_pts:
            if i >= len(self.points):
                print("Point higher than number of points available")
                return None

        # Open field comfort heatmap
        sp = os.path.join(plot_directory, "openfield_comfortheatmap.png")
        utci_comfort_heatmap(self.utci_openfield.values, title="Universal Thermal Climate Index - Open field",
                             save_path=sp, close=True, tone_color=tone_color)

        # Comfortable hours summary
        months = [
            [5],
            [10],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ]
        titles = [
            "May",
            "October",
            "Annual"
        ]
        for i, j in list(zip(*[months, titles])):
            sp = os.path.join(plot_directory, "comfortable_hours_{0:}.png".format(j.lower()))
            self.plot_comfortable_hours(lower=9, upper=28, months=i, percentile=0.9,
                                        title="Comfortable hours - {}".format(j), tone_color="k", save_path=sp,
                                        close=True)

        # Context only
        sp = os.path.join(plot_directory, "context.png")
        self.plot_context(rad_files, pts=False, label_pts=False, highlight_pt=None, save_path=sp, close=True,
                          tone_color=tone_color)

        # Context with point labels
        sp = os.path.join(plot_directory, "context_ptlabel.png")
        self.plot_context(rad_files, pts=True, label_pts=True, highlight_pt=None, save_path=sp, close=True,
                          tone_color=tone_color)

        # Context with focus points
        sp = os.path.join(plot_directory, "context_focuspts.png")
        self.plot_context(rad_files, pts=False, label_pts=False, highlight_pt=focus_pts, save_path=sp, close=True,
                          tone_color=tone_color)

        # Comfort, improvement and reduction plots
        masks = [
            self.mask_shoulder_morning,
            self.mask_shoulder_afternoon,
            self.mask_shoulder_day,

            self.mask_may & self.mask_shoulder_morning,
            self.mask_may & self.mask_shoulder_afternoon,
            self.mask_may & self.mask_shoulder_day,

            self.mask_october & self.mask_shoulder_morning,
            self.mask_october & self.mask_shoulder_afternoon,
            self.mask_october & self.mask_shoulder_day,
        ]
        titles = [
            "Annual - Morning Shoulder",
            "Annual - Afternoon Shoulder",
            "Annual - Morning & Afternoon Shoulder",
            "May - Morning Shoulder",
            "May - Afternoon Shoulder",
            "May - Morning & Afternoon Shoulder",
            "October - Morning Shoulder",
            "October - Afternoon Shoulder",
            "October - Morning & Afternoon Shoulder",
        ]
        for k in ["improvement", "comfort", "reduction"]:
            for i, j in list(zip(*[masks, titles])):
                vv = j.replace(" & ", "").replace(" - ", "_").replace(" ", "").lower()
                sp = os.path.join(plot_directory, "{}_{}.png".format(k, vv))
                self.plot_contour(_type=k, mask=i, rad_files=rad_files, clip=boundary, title=j, save_path=sp,
                                  close=True, tone_color=tone_color)

            # Create combined plot
            ims = [
                Image.open(os.path.join(plot_directory, "{}_annual_morningshoulder.png".format(k))),
                Image.open(os.path.join(plot_directory, "{}_annual_afternoonshoulder.png".format(k)))
            ]
            a = append_images(ims, direction='horizontal', bg_color=(255, 255, 255), aligment='center')
            ims = [
                Image.open(os.path.join(plot_directory, "{}_may_morningshoulder.png".format(k))),
                Image.open(os.path.join(plot_directory, "{}_may_afternoonshoulder.png".format(k)))
            ]
            b = append_images(ims, direction='horizontal', bg_color=(255, 255, 255), aligment='center')
            ims = [
                Image.open(os.path.join(plot_directory, "{}_october_morningshoulder.png".format(k))),
                Image.open(os.path.join(plot_directory, "{}_october_afternoonshoulder.png".format(k)))
            ]
            c = append_images(ims, direction='horizontal', bg_color=(255, 255, 255), aligment='center')

            # Combine verticals
            im = append_images([a, b, c], direction='vertical', bg_color=(255, 255, 255), aligment='center')
            im.save(os.path.join(plot_directory, "_{}_collection.png".format(k)))

        # Reduction and comfort heatmaps, profiles
        for fp in focus_pts:

            sp = os.path.join(plot_directory, "pt{0:04d}_reductionheatmap.png".format(fp))
            utci_reduction_heatmap(self.utci_difference[fp].values, save_path=sp, title="Universal Thermal Climate Index Difference - Point {0:}".format(fp), close=True, tone_color=tone_color)
            b = Image.open(sp)

            sp = os.path.join(plot_directory, "pt{0:04d}_comfortheatmap.png".format(fp))
            utci_comfort_heatmap(self.utci[fp].values, save_path=sp, title="Universal Thermal Climate Index - Point {0:}".format(fp), close=True, tone_color=tone_color)
            a = Image.open(sp)

            # Profiles
            months = [5, 10]
            titles = ["May", "October"]
            im_temps = []
            for i, j in list(zip(*[months, titles])):
                sp = os.path.join(plot_directory, "pt{0:04d}_profile_{1:}.png".format(fp, j.lower()))
                im_temps.append(sp)
                utci_day_comparison(self.utci[fp].values, self.utci_openfield,
                                    title="Average diurnal UTCI - {}".format(j), months=[i],
                                    names=["Open field", "Point {}".format(fp)], save_path=sp, close=True,
                                    tone_color=tone_color)

            c = append_images([Image.open(im_temps[0]), Image.open(im_temps[1])], direction='horizontal',
                              bg_color=(255, 255, 255), aligment='center')

            im = append_images([a, b, c], direction='vertical', bg_color=(255, 255, 255), aligment='center')
            im.save(os.path.join(plot_directory, "pt{0:04d}_collected.png".format(fp)))

        # Create summary table for UTCI reduction frequency in shoulder periods
        sp = os.path.join(plot_directory, "shoulder_reduction_summary.csv")
        self.reduction_summary(threshold=4, percentile=0.95, focus_pts=focus_pts).to_csv(sp)
        print("Summary saved to {}".format(sp))


def load_radiance_geometries(rad_files, exclude=["GND_SURROUNDING"], underlay=True):
    """ Load the radiance geometry into a dictionary, containing vertex groups and materials names as key

    Parameters
    ----------
    rad_files
    exclude
    underlay

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
        "GND_SURROUNDING": {
            "alpha": 0.0,
            "fc": "#BEBEBE",
            "zorder": 1,
            "lw": 0.0,
            "ec": "#BEBEBE",
            "ls": ":",
            "name": "Underground"},
        "GND_CONCRETELIGHT": {
            "alpha": 1.0,
            "fc": "#DBDBDB",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#DBDBDB",
            "ls": ":",
            "name": "Concrete (Light)"},
        "GND_CONCRETEDARK": {
            "alpha": 1.0,
            "fc": "#8F8F8F",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#8F8F8F",
            "ls": ":",
            "name": "Concrete (Dark)"},
        "GND_WOOD": {
            "alpha": 1.0,
            "fc": "#964B00",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#964B00",
            "ls": ":",
            "name": "Wood"},
        "GND_GRASS": {
            "alpha": 1.0,
            "fc": "#005800",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#005800",
            "ls": ":",
            "name": "Grass"},
        "GND_ASPHALT": {
            "alpha": 1.0,
            "fc": "#696969",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#696969",
            "ls": ":",
            "name": "Asphalt"},
        "GND_STONE": {
            "alpha": 1.0,
            "fc": "#FFDEAD",
            "zorder": 4,
            "lw": 0.75,
            "ec": "#FFDEAD",
            "ls": ":",
            "name": "Stone"},
        "GND_STONELIGHT": {
            "alpha": 1.0,
            "fc": "#FFF2DE",
            "zorder": 4,
            "lw": 0.75,
            "ec": "#FFF2DE",
            "ls": ":",
            "name": "Stone (Light)"},
        "GND_STONEDARK": {
            "alpha": 1.0,
            "fc": "#CCB491",
            "zorder": 4,
            "lw": 0.75,
            "ec": "#CCB491",
            "ls": ":",
            "name": "Stone (Dark)"},
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
        "GND_PLAYGROUND": {
            "alpha": 1.0,
            "fc": "#BF3F3F",
            "zorder": 1,
            "lw": 0.25,
            "ec": "#BF3F3F",
            "ls": ":",
            "name": "Playground"},
        "SHD_WATER": {
            "alpha": 0.5,
            "fc": "#3FBFBF",
            "zorder": 4,
            "lw": 0.5,
            "ec": "#3FBFBF",
            "ls": "-",
            "name": "Water feature"},
        "SHD_SOLID": {
            "alpha": 0.8,
            "fc": "#C9B4A5",
            "zorder": 3,
            "lw": 1.0,
            "ec": "#C9B4A5",
            "ls": "-",
            "name": "Solid shade"},
        "SHD_FABRIC": {
            "alpha": 0.65,
            "fc": "#E6CDBC",
            "zorder": 3,
            "lw": 1.0,
            "ec": "#E6CDBC",
            "ls": "-",
            "name": "Fabric shade"},
        "SHD_POROUS": {
            "alpha": 0.65,
            "fc": "#EDE2DA",
            "zorder": 3,
            "lw": 1.0,
            "ec": "#EDE2DA",
            "ls": "-",
            "name": "Porous shade"},
        "VEG_GHAF": {
            "alpha": 0.3,
            "fc": "#007F00",
            "zorder": 4,
            "lw": 1.0,
            "ec": "#007F00",
            "ls": "-",
            "name": "Ghaf (style) tree"
        },
        "VEG_PALM": {
            "alpha": 0.3,
            "fc": "#8ABAAE",
            "zorder": 4,
            "lw": 1.0,
            "ec": "#8ABAAE",
            "ls": "-",
            "name": "Palm (style) tree"
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
            # print("{} not found in radiance geometry - skipping".format(k))
            pass

    # Create patch collection for plotting
    for k, v in geo_data.items():
        patches = []
        try:
            for verts in v["vertices"]:
                if underlay:
                    patches.append(
                        Polygon(verts, closed=True, fill=False, fc=None, ec="#D1D1D1", lw=1, ls="-", zorder=8, alpha=0.25))
                else:
                    patches.append(Polygon(verts, closed=True, fill=True, fc=v["fc"], ec=v["ec"], lw=v["lw"], ls=v["ls"],
                                           zorder=v["zorder"], alpha=v["alpha"]))
        except Exception as e:
            print("Material: {0:} doesn't exist in the formatting dictionary. See the load_radiance_geometries() method.".format(k))
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


def utci_reduction_heatmap(hourly_utci_values, vrange=[-10, 10], title=None, tone_color="k", invert_y=False,
                           save_path=None, close=True):
    utci_reduction_cmap = LinearSegmentedColormap.from_list("reduction",
                                                            ["#70339e", "#887f8f", "#ffffff", "#0e7cab", "#00304a"],
                                                            100)

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
    cb = fig.colorbar(heatmap, cmap=utci_reduction_cmap, orientation='horizontal', drawedges=False, fraction=0.05,
                      aspect=100, pad=0.125)  #
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


def utci_day_comparison(hourly_utci_values_a, hourly_utci_values_b, months=np.arange(1, 13, 1), names=["A", "B"],
                        title=None, tone_color="k", save_path=None, close=True):
    day_idx = pd.date_range("2018-05-15 00:00:00", freq="60T", periods=24, closed="left")
    annual_idx = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")
    annual_mask = annual_idx.month.isin(months)
    masked_idx = annual_idx[annual_mask]

    # Create series to plot
    a = pd.Series(index=day_idx,
                  data=pd.Series(hourly_utci_values_a[annual_mask]).groupby([masked_idx.hour]).quantile(0.5).values)
    b = pd.Series(index=day_idx,
                  data=pd.Series(hourly_utci_values_b[annual_mask]).groupby([masked_idx.hour]).quantile(0.5).values)

    # Plotting
    locator = dates.HourLocator(byhour=np.arange(0, 25, 3))
    minor_locator = dates.HourLocator(byhour=np.arange(0, 25, 1))
    formatter = dates.DateFormatter('%H:%M')

    # Instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.5))

    # Plot input values
    ax.plot(a, label=names[1], c="#00A4E2", lw=2, zorder=2)
    ax.plot(b, label=names[0], c="#FF3E3E", lw=2, zorder=2)

    # Set plot limits
    ylims = [min([a.min(), b.min()]) - 3, max([a.max(), b.max()]) + 3]
    ax.set_ylim(ylims)

    # Add vertical difference lines at interesting indices
    sorted_diff = (a - b).sort_values(ascending=True)
    for n, l in enumerate([0, 23]):
        i = sorted_diff.index[l]
        j = -sorted_diff.iloc[l]
        k = np.max([a[i], b[i]], axis=0)
        ax.vlines(i, ymin=k, ymax=ax.get_ylim()[1] * 0.96, colors="#555555", lw=2, ls="-", zorder=1, alpha=0.2)
        ax.text(i, ax.get_ylim()[1], "{0:+0.1f}°".format(j), ha="center", va="top", color=tone_color, size="small")

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


def append_images(images, direction='horizontal', bg_color=(255, 255, 255), aligment='center'):
    """ Appends images in horizontal/vertical direction.

    Parameters
    ----------
    images : [PIL.Image]
        List of PIL images
    direction : str
        Direction of concatenation: 'horizontal' or 'vertical'
    bg_color : [r, g, b]
        Background color (0-255) (default: white)
    aligment : str
        Alignment mode if images need padding: 'left', 'right', 'top', 'bottom', or 'center'

    Returns
    -------
    image : PIL.Image
        Concatenated image as a new PIL image object
    """

    widths, heights = zip(*(i.size for i in images))

    if direction == 'horizontal':
        new_width = sum(widths)
        new_height = max(heights)
    else:
        new_width = max(widths)
        new_height = sum(heights)

    new_im = Image.new('RGB', (new_width, new_height), color=bg_color)

    offset = 0
    for im in images:
        if direction == 'horizontal':
            y = 0
            if aligment == 'center':
                y = int((new_height - im.size[1]) / 2)
            elif aligment == 'bottom':
                y = new_height - im.size[1]
            new_im.paste(im, (offset, y))
            offset += im.size[0]
        else:
            x = 0
            if aligment == 'center':
                x = int((new_width - im.size[0]) / 2)
            elif aligment == 'right':
                x = new_width - im.size[0]
            new_im.paste(im, (x, offset))
            offset += im.size[1]

    return new_im


def get_aggregate_day(utci_values, lower=9, upper=28, months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], percentile=0.95):
    # Create month of year mask and number of potential values
    mask = utci_values.index.month.isin(months)
    count = pd.Series(mask).groupby(utci_values.index.hour).sum()

    # Filter the input to only use the masked data
    temp = utci_values[mask]

    # Get the elements where the comfort limits are fulfilled
    temp = temp[(temp >= lower) & (temp <= upper)]

    # Get the number of hours where the "sample" point meets the comfort limit in the sample period
    temp = temp.groupby(temp.index.hour).count()

    # Get the nth percentile within the sample points
    try:
        temp = temp.quantile(percentile, axis=1)
    except Exception as e:
        pass

    # Get the proportion of comfortable hours in filtered data
    temp = temp / count
    temp.fillna(0, inplace=True)

    temp.index = pd.date_range("2018-05-15 00:00:00", freq="60T", periods=24, closed="left")

    return temp

def utci_reduction(utci_a, utci_b, months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], hours=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], threshold=4, prnt=False):
    months = np.array(months)
    hours = np.array(hours) - 1  # Needed to add -1 here to fix the indexing issue for hour of year
    utci_difference = utci_a - utci_b
    mask = utci_difference.index.hour.isin(hours) & utci_difference.index.month.isin(months)
    filtered_difference = utci_difference[mask]
    time_achieved = (filtered_difference >= threshold).mean()
    if prnt:
        print("Months: [{0:}]\nHours: [{1:}]\nThreshold: {2:}\n% UTCI reduction: {3:0.2%}".format(", ".join(months.astype(str)), ", ".join(hours.astype(str)), threshold, time_achieved))
    return time_achieved