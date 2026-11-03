from typing import Generator, Tuple, Union
import json
from itertools import chain
from IPython import get_ipython

import ipykernel
from jupyter_core.paths import jupyter_runtime_dir
from traitlets.config import MultipleInstanceError

from datetime import datetime
import sys
import os
from pathlib import Path, PurePath

import matplotlib.pyplot as plt
import urllib
import numpy as np
import pandas as pd
from pathlib import Path

import logging

FILE_ERROR = "Can't identify the notebook {}."
CONN_ERROR = "Unable to access server;\n" \
    + "ipynbname requires either no security or token based security."


def add_footer_to_fig(
    fig: plt.Figure,
    fn_fig: str | None = None,
    dpi: int = 250, font_size=8,
    add_pc_name=True,
    add_script=True,
    add_env=True,
    add_time_stamp=True,
    do_tight_layout=True,
) -> str:
    """
    Add footer to figure.

    Args:
        fig: Existing Matplotlib figure
        fn_fig: file path to save the figure to. If None, the figure is not saved
        dpi: dpi
        font_size: The font size
        add_pc_name: Add name of computer to the footer
        add_env: Add name of virtual environment to the footer
        add_script: Add name of script to the footer
        add_time_stamp: Add a time stamp to the footer

    Returns:
        fig_stamp: string with metadata
    """
    fig_stamp = "| "

    if add_time_stamp:
        fig_stamp += datetime.now().strftime("%Y-%m-%d %H:%M") + " | "

    if fn_fig is not None:
        # remove standard path, and add to string
        fig_stamp += strip_path(fn_fig) + " | "

    if add_script:
        # get filename, method depends on whether notebook or py-file is used
        try:
            fn_script = notebook_path()
        except RuntimeError:
            fn_script = str(Path(__file__))

        # remove standard path, and add to string
        fig_stamp += strip_path(fn_script) + " | "

    if add_pc_name:
        fig_stamp += os.environ['COMPUTERNAME'] + " | "

    if add_env:
        env_name, _ = get_venv_name_and_path()
        fig_stamp += env_name + " | "

    fig.text(0.01, 0.01, fig_stamp, fontsize=font_size, color='gray')

    if fn_fig is not None:
        if do_tight_layout:
            fig.tight_layout()
        fig.savefig(fn_fig, dpi=dpi)

    return fig_stamp


def get_venv_name_and_path() -> tuple:
    """
    his function supports various configurations
    such as Jupyter Notebook and any IDE (PyCharm, Spyder, etc.) with is of a conda environment or a
    Python environment created with virtualenv.
    """
    env = os.environ
    # Fetch the environment name and path. Depends on use of conda env en jupyter notebook
    try:
        # If conda is used and jupyter is started from the anaconda prompt
        env_name = env['CONDA_DEFAULT_ENV']
        env_path = env['CONDA_PREFIX']
    except KeyError:
        # If jupyter is started from the (regular) command line
        python_path = Path(sys.executable)
        env_path = python_path.parent

        # If the module virtualenv is used, then the folder 'Scripts' is between python and the folder 'env_name'
        if env_path.name == 'Scripts':
            env_name = env_path.parent.name
        else:
            env_name = env_path.name

    return env_name, env_path


def strip_path(str_to_strip: str, path_abbreviations={r'data\python\cloned': 'python_cache'}) -> str:
    """
    Function to abbreviate the full path of a file. The abbreviations are available as a dict in PATH_ABBREVIATIONS.

    Args:
        str_to_strip: full path of a file

    Returns:
        abbreviated_path: abbreviated path
    """
    # check input
    if not isinstance(str_to_strip, str):
        raise TypeError("Input should be a string")

    # Remove drive letter
    if str_to_strip.lower().startswith(r"\\ws.local\share\wk"):
        drive = r"\\ws.local\share\wk"
        str_to_strip = str_to_strip[len(drive) + 1:]

    else:
        drive, _ = os.path.splitdrive(str_to_strip)
        if len(drive) > 0:
            str_to_strip = str_to_strip[len(drive) + 1:]

    # Remove leading backslash if present
    if str_to_strip[0] == "\\":
        str_to_strip = str_to_strip[1:]

    new_path = str_to_strip

    # Abbreviate path if part of the path is in PATH_ABBREVIATIONS
    for full_path in path_abbreviations.keys():
        if str_to_strip.lower().startswith(full_path.lower()):
            new_path = path_abbreviations[full_path] + \
                str_to_strip[len(full_path):]
            break

    return new_path


def _get_kernel_id() -> str:
    """ Returns the kernel ID of the ipykernel.
    """
    connection_file = Path(ipykernel.get_connection_file()).stem
    kernel_id = connection_file.split('-', 1)[1]

    return kernel_id


def _list_maybe_running_servers(runtime_dir=None) -> Generator[dict, None, None]:
    """ Iterate over the server info files of running notebook servers.
    """
    if runtime_dir is None:
        runtime_dir = jupyter_runtime_dir()
    runtime_dir = Path(runtime_dir)

    if runtime_dir.is_dir():
        # Get notebook configuration files, sorted to check the more recently modified ones first
        for file_name in sorted(
            chain(
                # jupyter notebook (or lab 2)
                runtime_dir.glob('nbserver-*.json'),
                runtime_dir.glob('jpserver-*.json'),  # jupyterlab 3
            ),
            key=os.path.getmtime,
            reverse=True,
        ):
            try:
                yield json.loads(file_name.read_bytes())
            except json.JSONDecodeError as err:
                # Sometimes we encounter empty JSON files. Ignore them.
                pass


def _find_nb_path() -> Union[Tuple[dict, PurePath], Tuple[None, None]]:
    # Handle VS Code notebooks
    ip = get_ipython()
    if '__vsc_ipynb_file__' in ip.user_ns:
        return None, PurePath(ip.user_ns['__vsc_ipynb_file__'])

    try:
        kernel_id = _get_kernel_id()
    except (MultipleInstanceError, RuntimeError):
        return None, None  # Could not determine

    for srv in _list_maybe_running_servers():
        try:
            sessions = _get_sessions(srv)
            for sess in sessions:
                if sess['kernel']['id'] == kernel_id:
                    return srv, PurePath(sess['path'])
        except Exception:
            pass  # There may be stale entries in the runtime directory

    return None, None


def _get_sessions(srv):
    """ Given a server, returns sessions, or HTTPError if access is denied.
        NOTE: Works only when either there is no security or there is token
        based security. An HTTPError is raised if unable to connect to a
        server.
    """
    try:
        qry_str = ""
        token = srv['token']
        if not token and "JUPYTERHUB_API_TOKEN" in os.environ:
            token = os.environ["JUPYTERHUB_API_TOKEN"]
        qry_str = f"?token={token}" if token else ""
        url = f"{srv['url']}api/sessions{qry_str}"
        # Use a timeout in case this is a stale entry.
        with urllib.request.urlopen(url, timeout=0.5) as req:
            return json.load(req)
    except Exception:
        raise urllib.error.HTTPError(CONN_ERROR)


def name() -> str:
    """ Returns the short name of the notebook w/o the .ipynb extension,
        or raises a FileNotFoundError exception if it cannot be determined.
    """
    _, path = _find_nb_path()
    if path:
        return path.stem

    raise FileNotFoundError(FILE_ERROR.format('name'))


def notebook_path() -> str | None:
    """ Returns the absolute path of the notebook,
        or raises a FileNotFoundError exception if it cannot be determined.
    """
    srv, path = _find_nb_path()

    if srv and path:
        root_dir = Path(srv.get('root_dir') or srv['notebook_dir'])
        return root_dir / path

    if path:
        return str(path)

    raise FileNotFoundError(FILE_ERROR.format('path'))


def update_ylims(axes, ymin_threshold=None, ymax_threshold=None):
    # update ylims for multiple axes to have the same limits
    axes_unique = list(dict.fromkeys(axes))
    ymins, ymaxs = zip(*(ax.get_ylim() for ax in axes_unique))
    ymin, ymax = min(ymins), max(ymaxs)
    if ymin_threshold is not None:
        if ymin < ymin_threshold:
            ymin = ymin_threshold
    if ymax_threshold is not None:
        if ymax > ymax_threshold:
            ymax = ymax_threshold
    for ax in axes_unique:
        ax.set_ylim(ymin, ymax)


def add_logo_to_fig(fig, fn_logo=r'c:\data\logo\logo-horizontaal.png'):
    logo_ax = fig.add_axes([0.05, 0.93, 0.12, 0.08], zorder=10)
    logo_ax.imshow(plt.imread(fn_logo))
    logo_ax.axis('off')


def plot_sensor_4_graphs(
    obs_raai,
    title=None,
    plot_col_wc='water_content',
    plot_col_temp='soil_temperature',
    temp_offset=25,
    fn=None,
    close_fig=True,
    ylines=None,
    start=None,
    end=None,
):
    """Plot sensor data for water content and soil temperature.

    Parameters
    ----------
    obs_raai : pd.DataFrame
        DataFrame containing sensor observations.
    title : str
        Title for the plot.
    """

    vmin_temp = 999.0
    vmax_temp = -999.0
    for index, row in obs_raai.iterrows():
        vmin_temp = np.min(
            [vmin_temp, row.obs[start:end][plot_col_temp].min()])
        vmax_temp = np.max(
            [vmax_temp, row.obs[start:end][plot_col_temp].max()])
    logging.debug(f'vmintemp: {vmin_temp}, vmaxtemp: {vmax_temp}')

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(
        nrows=4,
        ncols=2,
        width_ratios=[19, 1],
        wspace=0.05,
        hspace=0.2,
    )
    axes = np.empty(4, dtype=object)
    axes[0] = fig.add_subplot(grid[0, 0])

    for row in range(1, 4):
        axes[row] = fig.add_subplot(grid[row, 0], sharex=axes[0])
    cbar_axes = np.array([fig.add_subplot(grid[row, 1]) for row in range(4)])

    for index, row in obs_raai.sort_values('screen_top', ascending=True).iterrows():
        label = f'{row.screen_top:0.2f} cm-mv'

        row.obs[start:end][plot_col_wc].plot(
            ax=axes[0],
            ylabel='bodemvocht [-]',
            label=label,
        )

        row_max = row.obs[start:end][plot_col_wc].max()
        series_relative = row.obs[start:end][plot_col_wc] / row_max

        sc_sm = axes[1].scatter(
            series_relative.index,
            np.ones_like(series_relative) * -row.screen_top,
            20,
            marker='|',
            c=series_relative.values,
            cmap='jet_r',
            vmin=0,
            vmax=1
        )

        row.obs[start:end][plot_col_temp].plot(
            ax=axes[2],
            ylabel='temperatuur [oC]',
            label=label,
        )

        sc_temp = axes[3].scatter(
            row.obs[start:end][plot_col_temp].index,
            np.ones_like(row.obs[start:end][plot_col_temp]) * -row.screen_top,
            20,
            marker='|',
            c=row.obs[start:end][plot_col_temp],
            cmap='jet',
            vmin=vmin_temp,
            vmax=vmax_temp,
        )

        df_daily = row.obs[start:end][plot_col_temp].resample('D').max()
        df_daily = df_daily.to_frame()
        df_daily['year'] = df_daily.index.year
        moments_above_offset = df_daily[plot_col_temp] > temp_offset
        df_label = df_daily[moments_above_offset].groupby('year').count()
        for i, r in df_label.iterrows():
            axes[3].text(
                pd.Timestamp(i, 6, 1),
                -row.screen_top + 0.03,
                f'{r[plot_col_temp]:.0f} dagen > {temp_offset} oC'
            )
    axes[0].set_title('meetreeks bodemvocht')
    axes[0].legend()

    axes[1].set_ylabel('sensor [m mv]')
    axes[1].set_title('bodemvocht [% van volledige verzadiging]')
    axes[1].text(
        series_relative.index[0],
        -row.screen_top+0.07,
        'per sensor niveau berekend als percentage van de maximale waarneming voor die sensor',
        va='center'
    )
    cbar_sm = fig.colorbar(sc_sm, cax=cbar_axes[1])
    cbar_sm.set_label('relatieve bodemvocht [-]')

    axes[2].set_title('meetreeks bodemtemperatuur')

    axes[2].axhline(y=temp_offset, color='k', ls='--',
                    label=f'{temp_offset} oC')
    axes[3].set_ylabel('sensor [m mv]')
    axes[3].set_title('temperatuur')
    cbar_temp = fig.colorbar(sc_temp, cax=cbar_axes[3])
    cbar_temp.set_label('temperatuur [°C]')

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, zorder=-10)
        if ylines is not None:
            for yline in ylines:
                ax.axvline(x=yline, color='k', ls='--', lw=4,
                           alpha=0.3, zorder=0, label=f'{yline.date()}')

    for id in [0, 2]:
        axes[id].legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
        )
        cbar_axes[id].remove()
    for id in [1, 3]:
        # set yticks to obs_raai.screen_top unique values
        axes[id].set_yticks(-obs_raai.screen_top.unique())
        axes[id].set_yticklabels(
            [f'-{st:0.2f}' for st in obs_raai.screen_top.unique()])

    fig.suptitle(title)
    add_logo_to_fig(fig)
    if fn is not None:
        add_footer_to_fig(fig, fn.replace(',', '_'))
    if close_fig:
        plt.close(fig)

    return fig, axes, cbar_axes


def prepare_soil_moisture_animation_data(oc, freq='D'):
    """Build the data needed for the section animation.

    Returns
    -------
    loc : pd.DataFrame
        Index is the (subset of the) soil moisture sensors in `oc`.
        Columns 'x' and 'y' hold the plotting location of each sensor.
    rel : pd.DataFrame
        Same index as `loc`, one column per day in the observation period,
        holding the relative soil moisture (daily mean / max) of that sensor.
    """
    oc_sm = oc.loc[oc.sensor_type == 'soil moisture'].copy()

    loc = pd.DataFrame(index=oc_sm.index)
    loc['x'] = oc_sm.distance_to_ref.astype(float)
    loc['y'] = oc_sm.screen_top.astype(float)

    days = pd.date_range(
        oc_sm.stats.dates_first_obs.min().normalize(),
        oc_sm.stats.dates_last_obs.max().normalize(),
        freq=freq,
    )

    rel = pd.DataFrame(index=oc_sm.index, columns=days, dtype=float)
    for index, row in oc_sm.iterrows():
        daily = row.obs.water_content.resample('D').mean()
        rel.loc[index] = (daily / row.obs.water_content.max()
                          ).reindex(days).to_numpy(dtype=float)

    return loc, rel


def animate_section(
    loc,
    rel,
    title='alle locaties',
    inverse_y_values=True,
    fn=None,
    fps=15,
):
    """Animate the relative soil moisture in the cross section, one frame per column of `rel`."""
    days = rel.columns

    x = loc.x.values
    y = (loc.y * -1 if inverse_y_values else loc.y).values

    cmap = plt.get_cmap('jet_r').copy()
    cmap.set_bad(color='lightgray', alpha=0.4)

    def masked_rel(i):
        return np.ma.masked_invalid(rel.iloc[:, i].values)

    # non-interactive timeline bar baked into each frame, showing progress through `days`
    fig, (ax, ax_slider) = plt.subplots(
        2, 1, figsize=(10, 4.8), gridspec_kw={'height_ratios': [8, 1], 'hspace': 0.5},
    )
    ax.scatter(x, y, marker='x', color='darkgray',
               s=100, label='soil moisture sensor')
    sc = ax.scatter(x, y, c=masked_rel(0), marker='o',
                    s=50, cmap=cmap, vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('relatieve bodemvochtigheid [-]')

    ax.set_xlabel('locatie sensor, ten opzichte van referentielijn')
    ax.set_ylabel('sensor niveau [m mv]')
    ax.grid(True)

    ax_slider.plot([days[0], days[-1]], [0, 0], color='gray',
                   lw=4, solid_capstyle='round')
    slider_marker = ax_slider.scatter(
        [days[0]], [0], s=120, color='crimson', zorder=5)
    ax_slider.set_xlim(days[0], days[-1])
    ax_slider.set_ylim(-1, 1)
    ax_slider.set_yticks([])
    ax_slider.set_xlabel('moment in tijd')

    add_logo_to_fig(fig)

    # set_array() alone did not refresh the rendered colors reliably, so
    # each frame removes the previous scatter and draws a brand new one.
    def update(i):
        nonlocal sc
        sc.remove()
        sc = ax.scatter(x, y, c=masked_rel(i), marker='o',
                        s=50, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(
            f'{title}, relatieve bodemvochtigheid per sensor, '
            f'{days[i].strftime("%d %b %Y")}'
        )
        slider_marker.set_offsets([[days[i], 0]])
        return (sc, slider_marker)

    from matplotlib.animation import (
        FuncAnimation,
        FFMpegWriter,
        PillowWriter,
        writers,
    )

    anim = FuncAnimation(
        fig,
        update,
        frames=len(days),
        interval=1000 / fps,
        blit=False,
    )

    if fn is not None:
        fn = Path(fn)
        # always write a gif, plus a movie file when ffmpeg is available
        anim.save(fn.with_suffix('.gif'), writer=PillowWriter(fps=fps))
        logging.info(f'saved animation to {fn.with_suffix(".gif")}')
        if writers.is_available('ffmpeg'):
            movie_fn = fn if fn.suffix.lower() != '.gif' else fn.with_suffix('.mp4')
            anim.save(movie_fn, writer=FFMpegWriter(fps=fps))
            logging.info(f'saved animation to {movie_fn}')

    plt.close(fig)
    return anim


def plot_per_depth(oc, screen_top_between=[0.6, 0.7], sensor_position_contains=None, raai_dp=None, sensor_type='soil moisture', tmin=None, tmax=None):
    # plot timeseries with same sensor depth
    fig, ax = plt.subplots(1, figsize=(12, 5))

    if sensor_position_contains is not None:
        oc_plot = oc.loc[
            (oc.screen_top >= screen_top_between[0]) &
            (oc.screen_top <= screen_top_between[1]) &
            (oc.sensor_position_name.str.contains(sensor_position_contains)) &
            (oc.sensor_type == sensor_type)
        ].sort_values('raai_dp', ascending=False)
        # assign unique color per raai_dp
        raai_dp_list = oc_plot.raai_dp.unique()
        color_map = dict(
            zip(raai_dp_list, plt.cm.tab10.colors[:len(raai_dp_list)]))

        for index, row in oc_plot.iterrows():
            color = color_map.get(row.raai_dp, 'blue')
            df = row.obs[tmin:tmax].water_content
            if df.empty:
                continue
            df.plot(
                ax=ax,
                label=f'{row.region}, raai {row.raai_dp}, pos {row.sensor_position_name}',
                linestyle='-',
                linewidth=1,
                color=color
            )
        title_str = f'{sensor_type.capitalize()} sensors in {sensor_position_contains} with depth {screen_top_between[0]}-{screen_top_between[1]} m-surface level'
    elif raai_dp is not None:
        oc_plot = oc.loc[
            (oc.screen_top >= screen_top_between[0]) &
            (oc.screen_top <= screen_top_between[1]) &
            (oc.raai_dp == raai_dp) &
            (oc.sensor_type == sensor_type)
        ].sort_values('distance_to_ref', ascending=True)

        # assign unique color per sensor_position_name
        sensor_positions = oc_plot.sensor_position_name.unique()
        color_map = dict(
            zip(sensor_positions, plt.cm.tab10.colors[:len(sensor_positions)]))

        for index, row in oc_plot.iterrows():
            color = color_map.get(row.sensor_position_name, 'blue')
            df = row.obs[tmin:tmax].water_content
            if df.empty:
                continue
            df.plot(
                ax=ax,
                label=f'{row.region}, raai {row.raai_dp}, pos {row.sensor_position_name}',
                linestyle='-',
                linewidth=1,
                color=color
            )
        title_str = f'{sensor_type.capitalize()} sensors in raai {raai_dp} with depth {screen_top_between[0]}-{screen_top_between[1]} m-surface level'

    ax.grid()
    ax.set_ylim(0, 0.5)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax.set_title(title_str)

    return fig, ax


def fig_location_in_cols_depths_in_rows(oc, region=None, dp=None, ylim=[0, 0.45], suptitle_prefix=None, vspans=None, vspan_colors=None, vspan_labels=None, suptitle_message=None, tmin=None, tmax=None):
    if (region is not None) & (dp is not None):
        oc = oc.loc[
            (oc.raai_dp == dp) &
            (oc.region == region)]
        suptitle_prefix = f', {region} dp={dp}'
    else:
        suptitle_prefix = ''

    locations = oc.sort_values('distance_to_ref').name_location.unique()
    depths = np.sort(oc.screen_top.unique())

    fig, axes = plt.subplots(len(depths), len(locations), figsize=(
        4*len(locations), 3*len(depths)), sharex=True, sharey=True)

    for i, depth in enumerate(depths):
        for j, loc in enumerate(locations):
            ax = axes[i, j] if len(depths) > 1 else axes[j]
            subset = oc[(oc.name_location == loc) & (oc.screen_top == depth)]
            for idx, row in subset.iterrows():
                df = row.obs[tmin:tmax].water_content
                if df.empty:
                    continue
                df.plot(ax=ax, label=idx)
            if i == 0:
                ax.set_title(loc)
            if j == 0:
                ax.set_ylabel(f"Depth: {depth:.2f} m")
            ax.grid()
            if i == len(depths)-1:
                ax.set_xlabel("")
            # if j == len(locations)-1:
            #    ax.legend(fontsize='small')

            if vspans is not None:
                for vspan, color, label in zip(vspans, vspan_colors, vspan_labels):
                    ax.axvspan(pd.Timestamp(vspan[0]), pd.Timestamp(
                        vspan[1]), color=color, alpha=0.1, label=label)

    xmin = np.max([oc.stats.get_first_last_obs_date().iloc[:, 0].min(), pd.Timestamp(
        tmin)]) if tmin is not None else oc.stats.get_first_last_obs_date().iloc[:, 0].min()
    xmax = np.min([oc.stats.get_first_last_obs_date().iloc[:, 1].max(), pd.Timestamp(
        tmax)]) if tmax is not None else oc.stats.get_first_last_obs_date().iloc[:, 1].max()
    axes[0, 0].set_xlim(xmin, xmax)
    axes[0, 0].set_ylim(ylim)

    fig.suptitle(
        f"Soil moisture timeseries per location and depth{suptitle_prefix}{suptitle_message if suptitle_message is not None else ''}", fontsize=16, y=1.02)

    return fig, axes


def plot_locations(oc):
    fig, ax = plt.subplots(1, figsize=(12, 5))
    ax.axvline(x=0, color='gray', linestyle='--', label='kruin', zorder=0)

    for (region, raai_dp), df_raai_dp in oc.groupby(['region', 'raai_dp']):
        ax.scatter(
            df_raai_dp.distance_to_ref,
            df_raai_dp.screen_top,
            40,
            label=f'{region}, raai {raai_dp}',
            edgecolor='black',
            linewidth=0.8,
        )
        # tensiometers
        df_raai_dp_tensiometer = df_raai_dp[df_raai_dp.sensor_type == 'tensiometer']
        if not df_raai_dp_tensiometer.empty:
            ax.scatter(
                df_raai_dp_tensiometer.distance_to_ref,
                df_raai_dp_tensiometer.screen_top,
                80,
                label=f'{region}, raai {raai_dp} (tensiometers)',
                marker='o',
                color='magenta',
                zorder=0,
            )

    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel('afstand tot referentielijn (m)')
    ax.text(0.01, 1.01, 'buitenwater', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left')
    ax.text(0.99, 1.01, 'achterland', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right')
    ax.set_ylabel('diepte (m-mv)')
    ax.legend(loc='lower right', ncols=2, fontsize='small')
    ax.set_title(
        'Locaties van sensoren in doorsnede over de dijk', fontsize=16)
    return fig, ax


def plot_section_at_date(
        oc_loc,
        plot_date=pd.Timestamp('2026-01-15'),  # type:   pd.Timestamp,
        inverse_y_values=True,
        title=None,
):
    fig, ax = plt.subplots(1, figsize=(10, 4))
    oc_loc['screen_top_plot'] = oc_loc.screen_top * - \
        1 if inverse_y_values else oc_loc.screen_top

    oc_loc.loc[oc_loc.sensor_type == 'soil moisture'].plot.scatter(
        x='distance_to_ref',
        y='screen_top_plot',
        marker='x',
        color='darkgray',
        s=100,
        label='soil moisture sensor',
        ax=ax)

    oc_loc[f'soil_moisture_{plot_date}'] = None
    for index, row in oc_loc.iterrows():
        if row.sensor_type == 'soil moisture':
            # check in plot_date is within row.obs.index
            if plot_date not in row.obs.index:
                continue
            day_value = row.obs[plot_date:plot_date +
                                pd.Timedelta('1D')].water_content.mean()
            max_value = row.obs.water_content.max()
            oc_loc.at[index,
                      f'rel_soil_moisture_{plot_date}'] = day_value / max_value

    oc_loc.loc[oc_loc.sensor_type == 'soil moisture'].plot.scatter(
        x='distance_to_ref',
        y='screen_top_plot',
        c=f'rel_soil_moisture_{plot_date}',
        marker='o',
        s=50,
        cmap='jet_r',
        vmin=0,
        vmax=1,
        ax=ax)
    ax.set_title(
        f'{title}, relatieve bodemvochtigheid per sensor, gemiddeld op {plot_date.date()}')
    ax.set_xlabel('doorsnede haaks op de dijk')
    ax.set_ylabel('sensor niveau [m mv]')

    tick_data = oc_loc[['distance_to_ref',
                        'sensor_position_name']].drop_duplicates()
    ax.set_xticks(tick_data['distance_to_ref'])
    ax.set_xticklabels(tick_data['sensor_position_name'])

    ax.set_xlabel('locatie sensor, ten opzichte van referentielijn')
    ax.set_ylabel('sensor niveau [m mv]')

    add_logo_to_fig(fig)

    add_footer_to_fig(
        fig,
        fr"plots\{title}_section_at_{plot_date.strftime('%Y%m%d')}.png",
    )
    return fig, ax
