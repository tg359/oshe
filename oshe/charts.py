import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.dates import date2num, DateFormatter
from psychrolib import SetUnitSystem, SI, GetHumRatioFromRelHum, GetMoistAirEnthalpy
from windrose import WindroseAxes

SetUnitSystem(SI)
humidityratio_relativehumidity = np.vectorize(GetHumRatioFromRelHum)
def temperature_enthalpy_humidityratio(enthalpy: float, humidity_ratio: float) -> float:
    return (enthalpy - 2.5 * (humidity_ratio * 1000)) / (1.01 + (0.00189 * humidity_ratio * 1000))

ANNUAL_DATETIME = pd.date_range(start="2018-01-01 00:30:00", freq="60T", periods=8760, closed="left")

MASKS = {
    "Daily": ((ANNUAL_DATETIME.hour >= 0) & (ANNUAL_DATETIME.hour <= 24)),
    "Morning": ((ANNUAL_DATETIME.hour >= 5) & (ANNUAL_DATETIME.hour <= 10)),
    "Midday": ((ANNUAL_DATETIME.hour >= 11) & (ANNUAL_DATETIME.hour <= 13)),
    "Afternoon": ((ANNUAL_DATETIME.hour >= 14) & (ANNUAL_DATETIME.hour <= 18)),
    "Evening": ((ANNUAL_DATETIME.hour >= 19) & (ANNUAL_DATETIME.hour <= 22)),
    "Night": ((ANNUAL_DATETIME.hour >= 23) | (ANNUAL_DATETIME.hour <= 4)),
    "MorningShoulder": ((ANNUAL_DATETIME.hour >= 7) & (ANNUAL_DATETIME.hour <= 10)),
    "AfternoonShoulder": ((ANNUAL_DATETIME.hour >= 16) & (ANNUAL_DATETIME.hour <= 19)),

    "Annual": ((ANNUAL_DATETIME.month >= 1) & (ANNUAL_DATETIME.month <= 12)),
    "Spring": ((ANNUAL_DATETIME.month >= 3) & (ANNUAL_DATETIME.month <= 5)),
    "Summer": ((ANNUAL_DATETIME.month >= 6) & (ANNUAL_DATETIME.month <= 8)),
    "Autumn": ((ANNUAL_DATETIME.month >= 9) & (ANNUAL_DATETIME.month <= 11)),
    "Winter": ((ANNUAL_DATETIME.month <= 2) | (ANNUAL_DATETIME.month >= 12)),
    "Shoulder": ((ANNUAL_DATETIME.month == 3) | (ANNUAL_DATETIME.month == 10))
}


def variablerose(
        annual_values: np.ndarray,
        annual_wind_direction: np.ndarray,
        save_path: str = None,
        close: bool = True,
        bins: np.ndarray = 10,
        season_period: str = "Annual",
        time_period: str = "Daily",
        n_sector: int = 16,
        title: str = None,
        unit: str = None,
        cmap: str = "GnBu",
        tone_color: str = "black",
        transparency: str = False
):
    """ Generates a pseudo-windrose plot displaying other variables from a set of that variable and directions, with the ability to filter for specific time-periods.

    Parameters
    ----------
    annual_values : np.ndarray
        List of annual hourly values to map.
    annual_wind_direction : np.ndarray
        List of annual hourly wind directions.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    bins : np.ndarray or None
        A list of values into which the variable data passed will be binned
    season_period : str
        Choose from ["Annual", "Spring", "Summer", "Autumn", "Winter", "Shoulder"].
    time_period : str
        Choose from [Daily, "Morning", "Midday", "Afternoon", "Evening", "Night", "MorningShoulder", "AfternoonShoulder"].
    n_sector : int
        The number of directions to bin wind directions into.
    title : str or None
        Adds a title to the plot.
    unit : str or None
        Adds a unit to the plot legend.
    cmap : str or None
        A Matplotlib valid colormap name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image
    """

    # Run checks on input data
    assert len(annual_values) == len(annual_wind_direction), \
        "Length of annual_values {} does not match length of annual_wind_direction {}".format(len(annual_values), len(
            annual_wind_direction))

    assert len(annual_values) == 8760, \
        "Number of hourly values passed ({}) does not equal 8760".format(len(annual_values))

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    assert season_period in MASKS.keys(), \
        "\"{}\" not known as a filter for season periods. Choose from {}".format(season_period, list(MASKS.keys()))

    assert time_period in MASKS.keys(), \
        "\"{}\" not known as a filter for time periods. Choose from {}".format(time_period, list(MASKS.keys()))

    # Construct a DataFrame containing the annual wind speeds and directions
    df = pd.DataFrame(index=ANNUAL_DATETIME)
    df["variable"] = annual_values
    df["wind_direction"] = annual_wind_direction

    # Create a set of masks to remove unwanted (null or zero) hours of the year, and for different time periods
    speed_mask = (df.variable != 0)
    direction_mask = (df.wind_direction != 0)
    mask = np.array([MASKS[time_period], MASKS[season_period], speed_mask, direction_mask]).all(axis=0)

    # Weird bug fix here to create curved ends to polar bars
    plt.hist([0, 1]);
    plt.close()

    # Instantiate figure
    plt.figure(figsize=(6, 6))
    ax = WindroseAxes.from_ax()
    ax.bar(
        df.wind_direction[mask],
        df.variable[mask],
        normed=True,
        bins=bins,
        opening=0.95,
        edgecolor='White',
        lw=0.1,
        nsector=n_sector,
        cmap=get_cmap(cmap)
    )

    # Generate and format legend
    lgd = ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', frameon=False, title=unit)
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    [plt.setp(text, color=tone_color) for text in lgd.get_texts()]
    plt.setp(lgd.get_title(), color=tone_color)

    for i, leg in enumerate(lgd.get_texts()):
        b = leg.get_text().replace('[', '').replace(')', '').split(' : ')
        lgd.get_texts()[i].set_text(b[0] + ' to ' + b[1])

    # Format plot canvas
    ax.grid(linestyle=':', color=tone_color, alpha=0.5)
    ax.spines['polar'].set_visible(False)
    plt.setp(ax.get_xticklabels(), color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)

    # Add title if provided
    if title is None:
        plt_title = "{0:} - {1:}".format(season_period, time_period)
    else:
        plt_title = "{0:} - {1:}\n{2:}".format(season_period, time_period, title)
    ax.set_title(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.05, transform=ax.transAxes)

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()


def windrose(
        annual_wind_speed: np.ndarray,
        annual_wind_direction: np.ndarray,
        save_path: str = None,
        close: bool = True,
        season_period: str = "Annual",
        time_period: str = "Daily",
        n_sector: int = 16,
        title: str = None,
        cmap: str = "GnBu",
        tone_color: str = "black",
        transparency: str = False
):
    """ Generates a windrose plot from a set of wind speeds and directions, with the ability to filter for specific time-periods.

    Parameters
    ----------
    annual_wind_speed : np.ndarray
        List of annual hourly wind speeds.
    annual_wind_direction : np.ndarray
        List of annual hourly wind directions.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    season_period : str
        Choose from ["Annual", "Spring", "Summer", "Autumn", "Winter", "Shoulder"].
    time_period : str
        Choose from [Daily, "Morning", "Midday", "Afternoon", "Evening", "Night", "MorningShoulder", "AfternoonShoulder"].
    n_sector : int
        The number of directions to bin wind directions into.
    title : str or None
        Adds a title to the plot.
    cmap : str or None
        A Matplotlib valid colormap name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image
    """

    # Run checks on input data
    assert len(annual_wind_speed) == len(annual_wind_direction), \
        "Length of annual_wind_speed {} does not match length of annual_wind_direction {}".format(
            len(annual_wind_speed), len(annual_wind_direction))

    assert len(annual_wind_speed) == 8760, \
        "Number of hourly values passed ({}) does not equal 8760".format(len(annual_wind_speed))

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    assert season_period in MASKS.keys(), \
        "\"{}\" not known as a filter for season periods. Choose from {}".format(season_period, list(MASKS.keys()))

    assert time_period in MASKS.keys(), \
        "\"{}\" not known as a filter for time periods. Choose from {}".format(time_period, list(MASKS.keys()))


    # Construct a DataFrame containing the annual wind speeds and directions
    df = pd.DataFrame(index=ANNUAL_DATETIME)
    df["wind_speed"] = annual_wind_speed
    df["wind_direction"] = annual_wind_direction

    # Create a set of masks to remove unwanted (null or zero) hours of the year, and for different time periods
    speed_mask = (df.wind_speed != 0)
    direction_mask = (df.wind_direction != 0)
    mask = np.array([MASKS[time_period], MASKS[season_period], speed_mask, direction_mask]).all(axis=0)

    # Weird bug fix here to create curved ends to polar bars
    plt.hist([0, 1]);
    plt.close()

    # Instantiate figure
    plt.figure(figsize=(6, 6))
    ax = WindroseAxes.from_ax()
    ax.bar(
        df.wind_direction[mask],
        df.wind_speed[mask],
        normed=True,
        bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        opening=0.95,
        edgecolor='White',
        lw=0.1,
        nsector=n_sector,
        cmap=get_cmap(cmap)
    )

    # Generate and format legend
    lgd = ax.legend(bbox_to_anchor=(1.1, 0.5), loc='center left', frameon=False, title="m/s")
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    [plt.setp(text, color=tone_color) for text in lgd.get_texts()]
    plt.setp(lgd.get_title(), color=tone_color)

    for i, leg in enumerate(lgd.get_texts()):
        b = leg.get_text().replace('[', '').replace(')', '').split(' : ')
        lgd.get_texts()[i].set_text(b[0] + ' to ' + b[1])

    # Format plot canvas
    ax.grid(linestyle=':', color=tone_color, alpha=0.5)
    ax.spines['polar'].set_visible(False)
    plt.setp(ax.get_xticklabels(), color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)

    # Add title if provided
    if title is None:
        plt_title = "{0:} - {1:}".format(season_period, time_period)
    else:
        plt_title = "{0:} - {1:}\n{2:}".format(season_period, time_period, title)
    ax.set_title(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.05, transform=ax.transAxes)

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()

def utci(
    annual_values: np.ndarray,
    save_path: str = None,
    close: bool = True,
    detailed: bool = False,
    title: str = None,
    tone_color: str = "black",
    invert_y: bool = False,
    transparency: bool = False
    ):
    """ Generate a UTCI heatmap from a set of annual hourly values

    Parameters
    ----------
    annual_values : np.ndarray
        List of 8760 annual hourly values.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    detailed : bool
        Set to True to include monthly stacked charts with proportion of time comfortable.
    title : str or None
        Adds a title to the plot.
    tone_color : str or None
        Text and border colours throughout the plot.
    invert_y : bool
        Reverse the y-axis so that 0-24 hours runs from top to bottom.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image
    """

    # Run input data checks
    assert len(annual_values) == 8760, \
        "Number of hourly values passed ({}) does not equal 8760".format(len(annual_values))

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    # Create DataFrame from passed values
    df = pd.DataFrame(data=annual_values, index=ANNUAL_DATETIME, columns=[title])

    # Reshape data into time/day matrix for heatmap plotting
    annual_matrix = df.pivot_table(columns=df.index.date, index=df.index.time).values[::-1]

    # Create UTCI colormap
    utci_cmap = ListedColormap(['#0D104B', '#262972', '#3452A4', '#3C65AF', '#37BCED', '#2EB349', '#F38322', '#C31F25', '#7F1416', '#580002'])
    utci_cmap_bounds = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
    utci_cmap_norm = BoundaryNorm(utci_cmap_bounds, utci_cmap.N)

    bounds = np.arange(-41, 48, 1)
    norm = BoundaryNorm(bounds, utci_cmap.N)

    if not detailed:
        # Instantiate figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))

        # Plot data
        heatmap = ax.imshow(
            annual_matrix,
            extent=[date2num(df.index.min()), date2num(df.index.max()), 726449, 726450],
            aspect='auto',
            cmap=utci_cmap,
            interpolation='none',
            vmin=-40,
            vmax=46,
            norm=utci_cmap_norm
        )

        # Axis formatting
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter('%b'))
        ax.yaxis_date()
        ax.yaxis.set_major_formatter(DateFormatter('%H:%M'))
        if invert_y:
            ax.invert_yaxis()
        ax.tick_params(length=0, labelleft=True, labelright=True, labeltop=False, labelbottom=True, color=tone_color)
        plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
        plt.setp(ax.get_yticklabels(), color=tone_color)

        # Spine formatting
        [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]

        # Grid formatting
        ax.grid(b=True, which='major', color='white', linestyle=':', alpha=1)

        # Add colorbar legend and text descriptors for comfort bands
        # cb = fig.colorbar(heatmap, orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.075)
        cb = fig.colorbar(heatmap, cmap=utci_cmap, norm=norm, boundaries=bounds, orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.15, extend='both', ticks=[-40, -27, -13, 0, 9, 26, 32, 38, 46])
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
        cb.set_label("°C", color=tone_color)
        cb.outline.set_visible(False)
        y_move = -0.135
        ax.text(0, y_move, 'Extreme\ncold stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(-27 + (-40 - -27) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Very strong\ncold stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(-13 + (-27 - -13) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Strong\ncold stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(0 + (-13 - 0) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Moderate\ncold stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(0 + (9 - 0) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Slight\ncold stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(9 + (26 - 9) / 2, [-44.319, 50.319], [0, 1]), y_move, 'No thermal stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(26 + (32 - 26) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Moderate\nheat stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(32 + (38 - 32) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Strong\nheat stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(np.interp(38 + (46 - 38) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Very strong\nheat stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')
        ax.text(1, y_move, 'Extreme\nheat stress', ha='center', va='center', transform=ax.transAxes, color=tone_color, fontsize='small')

        # Add title if provided
        if title is None:
            plt_title = "Universal Thermal Climate Index"
        else:
            plt_title = "{0:}\nUniversal Thermal Climate Index".format(title)
        ti = plt.title(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=ax.transAxes)

    else:
        fig = plt.figure(figsize=(15, 6), constrained_layout=True)
        spec = fig.add_gridspec(ncols=1, nrows=2, width_ratios=[1], height_ratios=[2, 1], hspace=0.1)

        hmap = fig.add_subplot(spec[0, 0])
        hbar = fig.add_subplot(spec[1, 0])

        # Plot heatmap
        heatmap = hmap.imshow(pd.pivot_table(df, index=df.index.time, columns=df.index.date,
                                             values=df.columns[0]).values[::-1], norm=utci_cmap_norm,
                              extent=[date2num(df.index.min()), date2num(df.index.max()), 726449,
                                      726450],
                              aspect='auto', cmap=utci_cmap, interpolation='none', vmin=-40, vmax=46)
        hmap.xaxis_date()
        hmap.xaxis.set_major_formatter(DateFormatter('%b'))
        hmap.yaxis_date()
        hmap.yaxis.set_major_formatter(DateFormatter('%H:%M'))
        if invert_y:
            hmap.invert_yaxis()
        hmap.tick_params(length=0, labelleft=True, labelright=True, labeltop=False, labelbottom=True, color=tone_color)
        plt.setp(hmap.get_xticklabels(), ha='left', color=tone_color)
        plt.setp(hmap.get_yticklabels(), color=tone_color)
        for spine in ['top', 'bottom', 'left', 'right']:
            hmap.spines[spine].set_visible(False)
            hmap.spines[spine].set_color(tone_color)
        hmap.grid(b=True, which='major', color="white", linestyle=':', alpha=1)

        # Add colorbar legend and text descriptors for comfort bands
        cb = fig.colorbar(heatmap, cmap=utci_cmap, norm=norm, boundaries=bounds,
                          orientation='horizontal', drawedges=False, fraction=0.01, aspect=50,
                          pad=-0.0, extend='both', ticks=[-40, -27, -13, 0, 9, 26, 32, 38, 46])
        plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
        cb.outline.set_visible(False)
        cb.set_label("°C", color=tone_color)

        y_move = -0.4
        hbar.text(0, y_move, 'Extreme\ncold stress', ha='center', va='center', transform=hbar.transAxes,
                  color=tone_color,
                  fontsize='small')
        hbar.text(np.interp(-27 + (-40 - -27) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Very strong\ncold stress',
                  ha='center', va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(-13 + (-27 - -13) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Strong\ncold stress',
                  ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(0 + (-13 - 0) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Moderate\ncold stress', ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(0 + (9 - 0) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Slight\ncold stress', ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(9 + (26 - 9) / 2, [-44.319, 50.319], [0, 1]), y_move, 'No thermal stress', ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(26 + (32 - 26) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Moderate\nheat stress',
                  ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(32 + (38 - 32) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Strong\nheat stress', ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(np.interp(38 + (46 - 38) / 2, [-44.319, 50.319], [0, 1]), y_move, 'Very strong\nheat stress',
                  ha='center',
                  va='center', transform=hbar.transAxes, color=tone_color, fontsize='small')
        hbar.text(1, y_move, 'Extreme\nheat stress', ha='center', va='center', transform=hbar.transAxes,
                  color=tone_color,
                  fontsize='small')

        # Add stacked plot
        bins = [-100, -40, -27, -13, 0, 9, 26, 32, 38, 46, 100]
        tags = ["Extreme cold stress", "Very strong cold stress", "Strong cold stress", "Moderate cold stress",
                "Slight cold stress", "No thermal stress", "Moderate heat stress", "Strong heat stress",
                "Very strong heat stress", "Extreme heat stress"]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        clrs = utci_cmap.colors

        adf = pd.DataFrame()
        for mnth_n, mnth in enumerate(months):
            # Filter the series to return only the month
            a = df[df.index.month == mnth_n + 1].dropna().values
            a = pd.Series(index=tags, name=mnth,
                          data=[((a > i) & (a <= j)).sum() / len(a) for n, (i, j) in
                                enumerate(zip(bins[:-1], bins[1:]))])
            adf = pd.concat([adf, a], axis=1)
        adf = adf.T[tags]
        adf.plot(kind="bar", ax=hbar, stacked=True, color=clrs, width=1, legend=False)
        hbar.set_xlim(-0.5, 11.5)
        #
        # # Major ticks
        hbar.set_xticks(np.arange(-0.5, 11, 1))

        # Labels for major ticks
        hbar.set_xticklabels(months)

        plt.setp(hbar.get_xticklabels(), ha='center', rotation=0, color=tone_color)
        plt.setp(hbar.get_xticklabels(), ha='left', color=tone_color)
        plt.setp(hbar.get_yticklabels(), color=tone_color)
        for spine in ['top', 'right']:
            hbar.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            hbar.spines[spine].set_color(tone_color)
        hbar.grid(b=True, which='major', color="white", linestyle=':', alpha=1)
        hbar.set_yticklabels(['{:,.0%}'.format(x) for x in hbar.get_yticks()])

        # Add header percentages for bar plot
        cold_percentages = adf.iloc[:, :5].sum(axis=1).values
        comfortable_percentages = adf.iloc[:, 5]
        hot_percentages = adf.iloc[:, 6:].sum(axis=1).values
        for n, (i, j, k) in enumerate(zip(*[cold_percentages, comfortable_percentages, hot_percentages])):
            hbar.text(n, 1.02, "{0:0.1f}%".format(i * 100), va="bottom", ha="center", color="#3C65AF", fontsize="small")
            hbar.text(n, 1.02, "{0:0.1f}%\n".format(j * 100), va="bottom", ha="center", color="#2EB349",
                      fontsize="small")
            hbar.text(n, 1.02, "{0:0.1f}%\n\n".format(k * 100), va="bottom", ha="center", color="#C31F25",
                      fontsize="small")
        hbar.set_ylim(0, 1)

        # Add title if provided
        if title is None:
            plt_title = "Universal Thermal Climate Index"
        else:
            plt_title = "{0:}\nUniversal Thermal Climate Index".format(title)
        ti = hmap.set_title(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=hmap.transAxes)

    # Tidy figure
    plt.tight_layout()

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_extra_artists=(ti,), bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()


def psychrometric(dry_bulb_temperature: np.ndarray, relative_humidity: np.ndarray, save_path: str = None, close: bool = True, atmospheric_station_pressure: np.ndarray = None, title: str = None, bins: int = 50, cmap: str = "viridis", tone_color: str = "black",
                 transparency: bool = False):
    """ Generate a psychrometric chart from a set of annual hourly values

    Parameters
    ----------
    dry_bulb_temperature : np.ndarray
        List of 8760 annual hourly dry-bulb temperature values.
    relative_humidity : np.ndarray
        List of 8760 annual hourly relative humidity values.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    atmospheric_station_pressure : np.ndarray or double or None
        List of 8760 annual hourly atmospheric station pressure values, or a single pressure value, or None.
    title : str or None
        Adds a title to the plot.
    bins : int
        Number of bins into which the dry-bulb temperature and calculated humidity ratio values will be placed
    cmap : str or None
        A Matplotlib valid colormap name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image
    """

    # Define ranges for plot/value limits
    lims_dbt = [-20, 50]
    lims_hr = [0, 0.03]
    lims_enth = [-10, 110]
    lims_rh = [0, 100]

    # If no atmoatmospheric_station_pressure passed, use sea level
    if atmospheric_station_pressure is None:
        atmospheric_station_pressure = 101350

    # Run input data checks
    assert len(dry_bulb_temperature) == len(relative_humidity) == 8760, \
        "Number of hourly values passed do not match each other"

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    # Calculate Humidity Ratio from Relative Humidity
    assert max(relative_humidity) <= 1, \
        "Relative humidity must be passed as a ratio between 0 and 1"
    humidity_ratio = humidityratio_relativehumidity(dry_bulb_temperature, relative_humidity, atmospheric_station_pressure)

    # Instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    # Plot values
    counts, xedges, yedges, im = ax.hist2d(dry_bulb_temperature, humidity_ratio, bins=bins, cmin=1, alpha=0.85, density=False, cmap=cmap, edgecolor=None, lw=0, zorder=0)

    # Y-axis formatting
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylim(lims_hr)
    ax.set_yticks(np.arange(lims_hr[0], lims_hr[1] + 0.001, 0.0025))
    ax.set_ylabel("Humidity ratio (kg$_{water}$/kg$_{air}$)", color=tone_color, fontsize="x-large")

    # X-axis formatting
    ax.set_xlim(lims_dbt)
    ax.set_xticks(np.arange(lims_dbt[0], lims_dbt[1] + 1, 2))
    ax.set_xlabel("Dry-bulb temperature (°C)", color=tone_color, fontsize="x-large")
    ax.tick_params(axis='both', colors=tone_color)

    # Canvas formatting
    ax.tick_params(axis="both", color=tone_color, grid_color=tone_color, grid_alpha=1, grid_lw=0.5)
    for edge in ["right", "bottom"]:
        ax.spines[edge].set_alpha(1)
        ax.spines[edge].set_color(tone_color)
        ax.spines[edge].set_lw(1)
    for edge in ["left", "top"]:
        ax.spines[edge].set_visible(False)

        # Add relative humidity grid/curves
        dry_bulb_temperatures = np.linspace(-20, 50, 71)
        humidity_ratios = [i / 10000 for i in range(0, 301, 1)]
        enthalpys = np.linspace(-10, 120, 14)
        relative_humiditys = np.linspace(0, 1, 11)

    for rh in relative_humiditys:
        h_r = [humidityratio_relativehumidity(i, rh, 101350) for i in dry_bulb_temperatures]
        ax.plot(dry_bulb_temperatures, h_r, color=tone_color, alpha=1, lw=0.2)
        # Fill the top part of the plot
        if rh == 1:
            ax.fill_between(dry_bulb_temperatures, h_r, 0.031, interpolate=True, color='w', lw=0, edgecolor=None,
                            zorder=4)
        # Add curve label
        ax.text(30, humidityratio_relativehumidity(30, rh, 101350), "{0:0.0f}% RH".format(rh * 100), ha="right",
                va="bottom", rotation=0, zorder=9, fontsize="small", color=tone_color)

    for enthalpy in enthalpys:
        ys = [0, 0.030]
        xs = [temperature_enthalpy_humidityratio(enthalpy, i) for i in lims_hr]
        ax.plot(xs, ys, color=tone_color, alpha=1, lw=0.2)
        # Add curve label
        if (enthalpy <= 50) & (enthalpy != 30):
            ax.text(xs[0], 0.0002, "{}kJ/kg".format(enthalpy), ha="right", va="bottom", color=tone_color, zorder=9,
                    fontsize="small")
        else:
            pass

    # Grid formatting
    ax.grid(b=True, which='major', color=tone_color, linestyle=':', alpha=0.5, zorder=3)

    # Add title if provided
    if title is None:
        plt_title = "Psychrometric chart"
    else:
        plt_title = "{0:}\nPsychrometric chart".format(title)
    ti = plt.title(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=ax.transAxes)

    # Colorbar
    cb = plt.colorbar(im, ax=ax, shrink=1, pad=0.071)
    cb.ax.set_title('Hours', color=tone_color)
    cb.outline.set_visible(False)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=tone_color)
    cb.ax.yaxis.set_tick_params(color=tone_color)
    # cb.tick_params(length=0, labelleft=False, labelright=False, labeltop=False, labelbottom=True, color=tone_color)

    # Tidy plot

    # Tidy figure
    plt.tight_layout()

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_extra_artists=(ti,), bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()


def heatmap(
    annual_values: np.ndarray,
    save_path: str = None,
    close: bool = True,
    title: str = None,
    unit: str = None,
    vrange: np.ndarray = None,
    cmap: str = 'viridis',
    tone_color: str = "black",
    invert_y: bool = False,
    transparency: bool = False
    ):
    """ Generate a heatmap from a set of annual hourly values

    Parameters
    ----------
    annual_values : np.ndarray
        List of 8760 annual hourly values.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    title : str or None
        Adds a title to the plot.
    unit : str or None
        Adds a unit string to the color-bar associated with the heatmap.
    vrange : np.ndarray or None
        Sets the range of values within which the color-bar will be applied.
    cmap : str or None
        A Matplotlib valid colormap name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/_color_data.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    invert_y : bool
        Reverse the y-axis so that 0-24 hours runs from top to bottom.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image
    """

    # Run input data checks
    assert len(annual_values) == 8760, \
        "Number of hourly values passed ({}) does not equal 8760".format(len(annual_values))

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    # Create DataFrame from passed values
    df = pd.DataFrame(data=annual_values, index=ANNUAL_DATETIME, columns=[title])

    # Reshape data into time/day matrix
    annual_matrix = df.pivot_table(columns=df.index.date, index=df.index.time).values[::-1]

    # Instantiate figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    # Plot data
    heatmap = ax.imshow(
        annual_matrix,
        extent=[date2num(df.index.min()), date2num(df.index.max()), 726449, 726450],
        aspect='auto',
        cmap=cmap,
        interpolation='none',
        vmin=vrange[0] if vrange is not None else None,
        vmax=vrange[-1] if vrange is not None else None,
    )

    # Axis formatting
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(DateFormatter('%b'))
    ax.yaxis_date()
    ax.yaxis.set_major_formatter(DateFormatter('%H:%M'))
    if invert_y:
        ax.invert_yaxis()
    ax.tick_params(labelleft=True, labelright=True, labelbottom=True, color=tone_color)
    plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)

    # Spine formatting
    [ax.spines[spine].set_visible(False) for spine in ['top', 'bottom', 'left', 'right']]

    # Grid formatting
    ax.grid(b=True, which='major', color='white', linestyle=':', alpha=1)

    # Color-bar formatting
    cb = fig.colorbar(heatmap, orientation='horizontal', drawedges=False, fraction=0.05, aspect=100, pad=0.075)
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=tone_color)
    cb.set_label(unit, color=tone_color)
    cb.outline.set_visible(False)

    # Add title if provided
    if title is not None:
        plt.title(title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=ax.transAxes)

    # Tidy plot
    plt.tight_layout()

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()



def frequency(values: np.ndarray, save_path: str = None, close: bool = True, title: str=None, unit: str=None, vrange: np.ndarray=None, bins: int=10, color: str="black", tone_color: str="black", transparency=False):
    """ Create a histogram with summary table for the data passed

    Parameters
    ----------
    values : np.ndarray
        List of values to bin.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    title : str or None
        Adds a title to the plot.
    unit : str or None
        Adds a unit string to the x-axis of the histogram.
    vrange : np.ndarray or None
        Sets the range of values within which the bins will be distributed.
    bins : int
        Number of bins into which the data should be sorted
    color : str
        A Matplotlib valid color name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image

    """

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    # Convert passed values into a pd.Series object
    series = pd.Series(values)

    # Instantiate the plot and add sub-plots
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    # Plot histogram
    if vrange is None:
        vrange = [series.min(), series.max()]
    ax.hist(series, bins=np.linspace(vrange[0], vrange[1], bins), color=color, alpha=0.9)

    # Format histogram
    ax.grid(b=True, which='major', color="k", linestyle=':', alpha=0.5, zorder=3)
    [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
    [ax.spines[j].set_color(tone_color) for j in ['bottom', 'left']]
    ax.tick_params(length=0, labelleft=True, labelright=False, labeltop=False, labelbottom=True, color=tone_color)
    ax.set_ylabel("Hours", color=tone_color)
    if vrange is None:
        ax.set_xlim(vrange)
    if unit is not None:
        ax.set_xlabel(unit, color=tone_color)
    plt.setp(ax.get_xticklabels(), ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)

    if title is not None:
        ax.set_title(title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=ax.transAxes)

    # Plot summary statistics
    _vals = np.round(series.describe().values, 2)
    _ids = ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
    summary_text = ""
    max_id_length = max([len(i) for i in _ids])
    max_val_length = max([len(str(i)) for i in _vals])
    string_length = max_id_length + max_val_length
    for i, j in list(zip(*[_ids, _vals])):
        _id_length = len(i)
        _val_length = len(str(j))
        space_length = string_length - _id_length - _val_length
        summary_text += "{0:}: {1:}{2:}\n".format(i, " " * space_length, j)
    txt = ax.text(1.01, 1, summary_text, fontsize=12, color=tone_color, horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, **{'fontname':'Courier New'})

    # Tidy plot
    plt.tight_layout()

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_inches='tight', bbox_extra_artists=(txt,), dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()



def diurnal(annual_values: np.ndarray, save_path: str = None, close: bool = True, grouping: str="Daily", months: np.ndarray=range(1, 13), title: str=None, unit: str=None, color: str="black", tone_color: str="black", transparency=False):
    """ Create an aggregate diurnal profile across either a day, week or year.

    Parameters
    ----------
    annual_values : np.ndarray
        List of annual hourly values to be aggregated.
    save_path : str
        The full path where the plot will be saved.
    close : bool
        Close the plot if you want. Or don't. I don't mind.
    grouping : str
        The method of grouping the aggregate diurnal values. Choose from ["Daily", "Weekly", "Monthly"].
    months : np.ndarray
        A list of integers denoting the months to include in the summary
    title : str or None
        Adds a title to the plot.
    unit : str or None
        Adds a unit string to the y-axis of the plot.
    color : str
        A Matplotlib valid color name. An up-to-date list of values is available from https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/colors.py.
    tone_color : str or None
        Text and border colours throughout the plot.
    transparency : bool
        Sets transparency of saved plot.

    Returns
    -------
    imagePath : str
        Path to the saved image

    """

    # Run input data checks
    assert len(annual_values) == 8760, \
        "Number of hourly values passed ({}) does not equal 8760".format(len(annual_values))

    assert ((min(months) >= 1) & (max(months) <= 12)), \
        "Month integers must be between 1 and 12 (inclusive)"

    # Convert passed values into a pd.Series object (and filter to remove unwanted months)
    series = pd.Series(annual_values, index=ANNUAL_DATETIME)[ANNUAL_DATETIME.month.isin(months)]

    if grouping == "Monthly":
        assert (len(months) == 12), \
            "Month filtering is not possible when grouping = \"Monthly\""

    if save_path is not None:
        assert os.path.exists(os.path.dirname(save_path)), \
            "\"{}\" does not exist".format(os.path.abspath(os.path.dirname(save_path)))

    # Define grouping methodologies
    groupings = {
        "Daily": {
            "grp": series.index.hour,
            "periods": 24,
            "xlabels": ["{0:02d}:00".format(i) for i in range(24)],
            "xticks": np.arange(0, 24, 1),
            "skip_n": [0, 1]
        },
        "Weekly": {
            "grp": [series.index.dayofweek, series.index.hour],
            "periods": 24 * 7,
            "xlabels": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            "xticks": np.arange(0, 24 * 7, 24),
            "skip_n": [0, 7]
        },
        "Monthly": {
            "grp": [series.index.month, series.index.hour],
            "periods": 24 * 12,
            "xlabels": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            "xticks": np.arange(0, 24 * 12, 24),
            "skip_n": [0, 12]
        }
    }

    assert grouping in groupings.keys(), \
        "\"{}\" not available as a filter for grouping annual hourly data. Choose from {}".format(grouping, list(groupings.keys()))

    _grouping = series.groupby(groupings[grouping]["grp"])
    _min = _grouping.min().reset_index(drop=True)
    _mean = _grouping.mean().reset_index(drop=True)
    _max = _grouping.max().reset_index(drop=True)
    # TODO: Add end value to each day/week/month to fill gaps in Series

    # Instantiate plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))

    # Plot aggregate
    [ax.plot(_mean.iloc[i:i + 24], color=color, lw=2, label='Average') for i in np.arange(0, groupings[grouping]["periods"])[::24]]
    [ax.fill_between(np.arange(i, i + 24), _min.iloc[i:i + 24], _max.iloc[i:i + 24], color=color, alpha=0.2,
                     label='Range') for i in np.arange(0, groupings[grouping]["periods"])[::24]]
    ax.set_ylabel(unit, labelpad=2, color=tone_color)
    ax.yaxis.set_major_locator(MaxNLocator(7))

    # Format plot area
    [ax.spines[spine].set_visible(False) for spine in ['top', 'right']]
    [ax.spines[j].set_color(tone_color) for j in ['bottom', 'left']]
    ax.set_xlim([0, groupings[grouping]["periods"]])
    ax.xaxis.set_ticks(groupings[grouping]["xticks"])
    ax.set_xticklabels(groupings[grouping]["xlabels"], ha='left', color=tone_color)
    plt.setp(ax.get_yticklabels(), color=tone_color)
    ax.grid(b=True, which='major', axis='both', c=tone_color, ls='--', lw=1, alpha=0.3)
    ax.tick_params(length=0, labelleft=True, labelright=False, labeltop=False, labelbottom=True, color=tone_color)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(bbox_to_anchor=(0.5, -0.2), loc=8, ncol=3, borderaxespad=0., frameon=False,
                    handles=[handles[groupings[grouping]["skip_n"][0]], handles[groupings[grouping]["skip_n"][1]]], labels=[labels[groupings[grouping]["skip_n"][0]], labels[groupings[grouping]["skip_n"][1]]])
    lgd.get_frame().set_facecolor((1, 1, 1, 0))
    [plt.setp(text, color=tone_color) for text in lgd.get_texts()]

    # Add a title
    if title is None:
        plt_title = "{0:} diurnal profile".format(grouping)
    else:
        plt_title = "{0:}\n{1:} diurnal profile".format(title, grouping)
    title = plt.suptitle(plt_title, color=tone_color, ha="left", va="bottom", x=0, y=1.01, transform=ax.transAxes)

    # Tidy plot
    plt.tight_layout()

    if save_path is not None:
        # Save figure
        plt.savefig(save_path, bbox_inches='tight', dpi=300, transparent=transparency)
        print("Plot saved to {}".format(save_path))

    if close:
        plt.close()
