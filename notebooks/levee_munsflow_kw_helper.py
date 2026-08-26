import os
import hydropandas as hpd
import pandas as pd
import logging
import json
import tempfile
import matplotlib.pyplot as plt

import matplotlib.dates as mdates

# META DATA for additional meta data in ObsCollection
META_DATA_KEYS = [
    "region",
    "dp",
    "position",
    "position_short",
    "ref_distance",
    "distance_to_ref",
    "filter_letter",
    "tube_bottom",
    "traject",
    "timeseries_id",
    "in_analysis",
    "remark_obs",
    "raai_plot",
    "plot_dp",
    "plot_x",
]


def _restore_timezone(obs, tz_name):
    """Restore a timezone on an observation index when possible.

    Args:
        obs: Hydropandas observation-like object with an ``index`` attribute.
        tz_name: Timezone name (for example ``Europe/Amsterdam``) or ``None``.

    Returns:
        The same observation object with a timezone-aware ``DatetimeIndex`` when conversion succeeds.
    """
    idx = getattr(obs, "index", None)
    if tz_name is None or not isinstance(idx, pd.DatetimeIndex):
        return obs

    try:
        idx_utc = pd.to_datetime(idx, utc=True)
        idx_tz = idx_utc.tz_convert(tz_name)
        if hasattr(idx_tz, "as_unit"):
            idx_tz = idx_tz.as_unit("ns")
        obs.index = idx_tz
    except Exception as err:
        logging.warning("Kon timezone '%s' niet herstellen voor '%s': %s",
                        tz_name, getattr(obs, 'name', '?'), err)
    return obs


def _normalize_payload_for_from_dict(payload, obs_label="?"):
    """Normalize JSON payload fields before hydropandas ``from_dict``.

    Args:
        payload: Dictionary loaded from a JSON observation file.
        obs_label: Observation label used in warning messages.

    Returns:
        A shallow copy of ``payload`` with ``obs`` and ``meta`` decoded when they are JSON strings.
    """
    normalized = dict(payload)

    obs_raw = normalized.get("obs")
    if isinstance(obs_raw, str):
        obs_text = obs_raw.strip()
        if obs_text.startswith("{") or obs_text.startswith("["):
            try:
                normalized["obs"] = json.loads(obs_text)
            except Exception as err:
                logging.warning(
                    "Kon 'obs' niet decoderen voor '%s': %s",
                    obs_label,
                    err,
                )

    meta_raw = normalized.get("meta")
    if isinstance(meta_raw, str):
        meta_text = meta_raw.strip()
        if meta_text.startswith("{") or meta_text.startswith("["):
            try:
                normalized["meta"] = json.loads(meta_text)
            except Exception as err:
                logging.warning(
                    "Kon 'meta' niet decoderen voor '%s': %s",
                    obs_label,
                    err,
                )

    return normalized


def json_to_hpd(
    json_zip: str,
    tz_meta_fn: str = "timezone_meta.json",
):
    """Read zipped JSON observation files into a hydropandas ``ObsCollection``.

    Args:
        json_zip: Path to a zip file containing observation JSON files.
        tz_meta_fn: Expected timezone metadata filename inside the zip.

    Returns:
        A hydropandas ``ObsCollection`` with metadata columns added when available.
    """
    obs_list = []

    import zipfile
    import io

    with zipfile.ZipFile(json_zip, "r") as zip_ref:
        zip_members = sorted(
            fn for fn in zip_ref.namelist()
            if not fn.endswith("/")
        )

        tz_meta_members = [
            fn for fn in zip_members
            if fn.lower().endswith(tz_meta_fn.lower())
            or "timezone" in os.path.basename(fn).lower()
        ]

        timezone_map = {}
        for tz_meta_member in tz_meta_members:
            with zip_ref.open(tz_meta_member, "r") as f:
                tz_payload = json.load(io.TextIOWrapper(f, encoding="utf-8"))
                if isinstance(tz_payload, dict):
                    timezone_map.update(tz_payload)
                else:
                    logging.warning(
                        "Timezone-bestand '%s' bevat geen dictionary en wordt overgeslagen",
                        tz_meta_member,
                    )

        json_files = [
            fn for fn in zip_members
            if fn.lower().endswith(".json")
            and fn not in tz_meta_members
        ]

        for json_path in json_files:
            with zip_ref.open(json_path, "r") as f:
                json_text = io.TextIOWrapper(f, encoding="utf-8").read()

            tmp_json = None
            try:
                # Primary route: parse JSON through hydropandas.
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    encoding="utf-8",
                    delete=False,
                ) as tmpf:
                    tmpf.write(json_text)
                    tmp_json = tmpf.name
                obs = hpd.read_json(tmp_json)
            except AssertionError as err:
                if "obstype" in str(err):
                    logging.warning(
                        "JSON zonder 'obstype' overgeslagen: %s",
                        json_path,
                    )
                    continue
                raise
            except Exception:
                # Fallback for legacy or slightly deviating payload structures.
                payload = json.loads(json_text)
                payload = _normalize_payload_for_from_dict(
                    payload,
                    os.path.splitext(os.path.basename(json_path))[0],
                )

                if "obstype" not in payload:
                    logging.warning(
                        "JSON zonder 'obstype' overgeslagen: %s",
                        json_path,
                    )
                    continue

                obstype = payload.get("obstype", "Obs")
                obs_cls = getattr(hpd, obstype, hpd.Obs)

                if hasattr(obs_cls, "from_dict"):
                    obs = obs_cls.from_dict(payload)
                else:
                    raise AttributeError(
                        f"{obs_cls.__name__} heeft geen from_dict voor zip-inhoud")
            finally:
                if tmp_json and os.path.exists(tmp_json):
                    os.remove(tmp_json)

            obs_name = str(getattr(obs, "name", os.path.splitext(
                os.path.basename(json_path))[0]))
            obs = _restore_timezone(obs, timezone_map.get(obs_name))
            obs_list.append(obs)

    logging.info(f"JSON files read: {len(json_files)}")
    logging.info(f"Obs created: {len(obs_list)}")

    try:
        oc_gwl = hpd.ObsCollection(obs_list)
    except TypeError:
        oc_gwl = hpd.ObsCollection.from_list(obs_list)

    if len(oc_gwl) >= 1:
        for key in META_DATA_KEYS:
            oc_gwl = oc_gwl.add_meta_to_df(key)
    else:
        logging.critical('geen observaties in oc, dat is niet verwacht')

    logging.info(f"ObsCollection created with {len(oc_gwl)} Obs objects")

    return oc_gwl


def calc_cummulative_precip(
        precip_df,
        cumulative_start_month=8,
        month_names=['januari', 'februari', 'maart', 'april', 'mei', 'juni',
                     'juli', 'augustus', 'september', 'oktober', 'november', 'december']
):
    """
    Calculate cumulative precipitation per specific season.

    Args:
        precip_df: DataFrame with a datetime index and precipitation values in the first column.
        cumulative_start_month: Month number (1-12) that defines the season start.
        month_names: Month labels used to return the selected start month name.

    Returns:
        Tuple of ``(precip_df, month_name)`` where ``precip_df`` includes date, year,
        month, season_year and cumulative columns.
    """

    precip_df['date'] = precip_df.index
    precip_df['year'] = precip_df['date'].dt.year
    precip_df['month'] = precip_df['date'].dt.month
    precip_df['season_year'] = precip_df['year']
    precip_df.loc[precip_df['month'] <
                  cumulative_start_month, 'season_year'] -= 1
    precip_df['cumulative'] = precip_df.groupby(
        'season_year')[precip_df.columns[0]].cumsum()

    return precip_df, month_names[cumulative_start_month - 1]


def plot_cummulative_precip(
        precip_df,
        location=None,
        fig=None,
        ax=None,
        ls='-',
        colors=['r', 'g', 'b', 'c', 'm', 'y', 'k'],
        location_in_legend=None,
        cumulative_start_month=8,
        month_names=['jan', 'feb', 'mrt', 'apr', 'mei', 'jun',
                     'jul', 'aug', 'sep', 'okt', 'nov', 'dec']
):
    """Plot cumulative precipitation curves per season year.

    Args:
        precip_df: DataFrame returned by ``calc_cummulative_precip``.
        location: Optional location name for the title.
        fig: Optional Matplotlib figure to draw on.
        ax: Optional Matplotlib axes to draw on.
        ls: Line style for plotted season curves.
        colors: List of colors cycled per season.
        location_in_legend: Optional location text prefix in legend labels.
        cumulative_start_month: Month number (1-12) used in labels.
        month_names: Month labels used for legend and title text.

    Returns:
        Tuple ``(fig, ax)`` of the Matplotlib figure and axes.
    """
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    if colors is None:
        colors = list(plt.get_cmap('tab10').colors)

    if location_in_legend:
        location_in_legend = location_in_legend + ', '

    for i, (year, group) in enumerate(precip_df.groupby('season_year')):
        days_since_start = (
            group['date'] - group['date'].iloc[0]).dt.total_seconds() / 86400
        if year < 2026:
            ax.plot(days_since_start, group['cumulative'],
                    label=f'{month_names[cumulative_start_month - 1]} {year} - {month_names[cumulative_start_month - 2]} {year+1}, {location_in_legend} som:{group["cumulative"].iloc[-1]:.1f} m',
                    color=colors[i % len(colors)],
                    ls=ls)

    ax.set_xticks([0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334])
    ax.set_xticklabels(['1 aug', '1 sep', '1 okt', '1 nov', '1 dec',
                        '1 jan', '1 feb', '1 mrt', '1 apr', '1 mei', '1 jun', '1 jul'])

    if location_in_legend:
        ax.set_title(
            f'Cumulatieve neerslag sinds 1 {month_names[cumulative_start_month - 1]}')
    else:
        ax.set_title(
            f'Cumulatieve neerslag sinds 1 {month_names[cumulative_start_month - 1]} voor {location}')
    ax.set_ylabel('(m)')
    handles, labels = ax.get_legend_handles_labels()
    legend_items = sorted(zip(labels, handles), key=lambda item: item[0])
    if legend_items:
        sorted_labels, sorted_handles = zip(*legend_items)
        ax.legend(sorted_handles, sorted_labels)
    ax.set_xlim(0, 366)
    ax.set_ylim(bottom=0)
    ax.grid(True)

    return fig, ax


def add_offset_for_close_points(oc, col_offset='distance_to_ref', suffix='plot', dup_offset=1.0):
    """Add a plotting offset for duplicate x-values in an observation collection.

    Args:
        oc: DataFrame-like object containing the offset column.
        col_offset: Source column with base x-coordinate values.
        suffix: Suffix used for the generated plotting column name.
        dup_offset: Offset increment added to duplicate values.

    Returns:
        Updated object with an extra column ``{col_offset}_{suffix}``.
    """
    col_offset_plot = f"{col_offset}_{suffix}"
    oc[col_offset_plot] = oc[col_offset].copy()
    oc = oc.sort_values(col_offset_plot)
    while True:
        dup_offset = oc.groupby(col_offset_plot).cumcount() * dup_offset

        if dup_offset.gt(0).any():
            oc[col_offset_plot] = oc[col_offset_plot] + dup_offset
            oc = oc.sort_values(col_offset_plot)
            continue
        else:
            break
    return oc


def ax_lim_as_dates(ax):
    """Return x-axis limits as timezone-naive pandas timestamps.

    Args:
        ax: Matplotlib axes object.

    Returns:
        Tuple ``(x_start, x_end)`` ordered from early to late timestamp.
    """
    x0, x1 = ax.get_xlim()
    x_start, x_end = pd.to_datetime(
        mdates.num2date([x0, x1])).tz_localize(None)
    if x_start > x_end:
        x_start, x_end = x_end, x_start

    return x_start, x_end


def plot_dates_as_vline(
        df,
        ax,
        date_colname='date',
        color='tab:orange',
        label=None
):
    """Draw vertical lines for dates that fall within current axis limits.

    Args:
        df: DataFrame containing datetime values in ``date_colname``.
        ax: Single Matplotlib axes or list of axes.
        date_colname: Name of the datetime column in ``df``.
        color: Line color for the vertical markers.
        label: Optional legend label for the markers.

    Returns:
        None. The function plots directly on the provided axes.
    """
    # check if ax is a list
    if isinstance(ax, list):
        plot_axes = ax
    else:
        plot_axes = [ax]
    x_start, x_end = ax_lim_as_dates(plot_axes[0])

    # gebruik alle relevante neerslagmomenten opnieuw en beperk tot zichtbare periode
    dates_all = (
        df[date_colname]
        .dt.normalize()
        .drop_duplicates()
    )
    dates_in_xlim = dates_all[(dates_all >= x_start) & (dates_all <= x_end)]

    if dates_in_xlim.empty:
        plot_axes[0].text(
            0.02, 0.95, "Geen relevante neerslagmomenten binnen x-limieten",
            transform=plot_axes[0].transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="0.7")
        )
        df_plot = pd.DataFrame({"date": pd.to_datetime([])})
    else:
        df_plot = pd.DataFrame({"date": dates_in_xlim})
    for date in df_plot['date']:
        for ax in plot_axes:
            ax.axvline(x=date, color=color, linewidth=0.5, label=label)


def short_legend_precip(ax):
    """Create a compact legend with unique non-empty labels for precipitation plots.

    Args:
        ax: Matplotlib axes containing plotted artists.

    Returns:
        None. The legend is created and attached to ``ax``.
    """
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in unique:
            unique[l] = h

    leg = ax.legend(
        unique.values(),
        unique.keys(),
        loc='center right',
        bbox_to_anchor=(-0.05, 0.3),
        borderaxespad=0.0,
        framealpha=1.0
    )
    # keep tight_layout from resizing/repositioning other axes
    leg.set_in_layout(False)
    leg.set_zorder(10_000)
    leg.set_clip_on(False)


def select_surfacelevelprofile(dp_center, df_profiles, delta_dp=1, region='Os'):
    """
    Select the closest available surface level profile in a region.

    Args:
        dp_center: Central dike pole number.
        df_profiles: DataFrame with surface level profile coordinates.
        delta_dp: Search range around the central dike pole.
        region: Region name.

    Returns:
        Tuple ``(x_vals, y_vals, closest_dp_profile)`` where coordinates can be
        ``None`` when insufficient data is available.
    """

    # select data
    df_profiles_subset = df_profiles[
        df_profiles.dp.between(dp_center - delta_dp, dp_center + delta_dp) &
        (df_profiles.region == region)
    ]
    # find closest dp in df_profiles
    closest_dp_profile = df_profiles_subset.dp.iloc[(
        df_profiles_subset.dp - dp_center).abs().argsort()[:1]].values[0]
    logging.debug(
        f'Closest dp with surface level profile to {dp_center} is {closest_dp_profile}')

    # prepare plotting
    lst_plot_profiel = ['mv.bin', 'sloot.1a', 'sloot.1c', 'sloot.1d', 'sloot.1b', 'weg.1', 'teen.1', 'berm.1a',
                        'berm.1b', 'kruin.1', 'kruin.2', 'berm.2a', 'berm.2b', 'teen.2', 'weg.2', 'sloot.2', 'mvb.bui']
    lst_plot_profiel_x = ['x' + s for s in lst_plot_profiel]
    lst_plot_profiel_y = ['y' + s for s in lst_plot_profiel]
    lst_plot_profiel_y[-1] = 'ymv.bui'
    lst_plot_profiel_y[3] = 'ysloot.1d'

    if len(df_profiles_subset) > 0:
        for index, row in df_profiles_subset.iterrows():
            # Plot the profile line in the figure
            x_vals = [row[x]
                      for x in lst_plot_profiel_x if x in row and pd.notnull(row[x])]
            y_vals = [row[y]
                      for y in lst_plot_profiel_y if y in row and pd.notnull(row[y])]
            if len(x_vals) > 1 and len(y_vals) > 1:
                return x_vals, y_vals, closest_dp_profile
            else:
                logging.debug(
                    f"Not enough data to get profile for dp{row['dp']:.1f}")
                return None, None, closest_dp_profile
    else:
        logging.debug(
            f"No surface level profile found for dp{dp_center:.1f} in region {region}")
        return None, None, closest_dp_profile


def plot_doodtij(
        ax,
        df_doodtij,
        x_start=None,
        x_end=None
):
    """Plot vertical lines for dead-tide moments within a selected period.

    Args:
        ax: Matplotlib axes to draw on.
        df_doodtij: DataFrame with column ``doodtij_verwachting_localize``.
        x_start: Optional period start. If omitted, derived from axis limits.
        x_end: Optional period end. If omitted, derived from axis limits.

    Returns:
        None. The function plots directly on the provided axes.
    """
    if x_start is None and x_end is None:
        x_start, x_end = ax_lim_as_dates(ax)

    # make doodtij_verwachting_localize timezone-naive
    df_doodtij['doodtij_verwachting_localize'] = df_doodtij['doodtij_verwachting_localize'].apply(
        lambda x: x.tz_localize(None) if getattr(
            x, "tzinfo", None) is not None else x
    )

    for i, (_, row) in enumerate(df_doodtij.loc[(df_doodtij.doodtij_verwachting_localize >= x_start) & (df_doodtij.doodtij_verwachting_localize <= x_end)].iterrows()):
        t_doodtij = row.doodtij_verwachting_localize

        ax.axvline(
            t_doodtij,
            color='red',
            linestyle='--',
            alpha=0.5,
            label='doodtij' if i == 0 else None
        )


def plot_pb_precip(
        obs_gws,
        precip_df,
        df_doodtij=None,
        start=None,
        end=None,
        col_gwl='gwl_mnap',
        precip_col='RH'
):
    """Plot groundwater series and daily precipitation in a two-panel figure.

    Args:
        obs_gws: Observation collection/DataFrame with ``obs`` time series per row.
        precip_df: Precipitation DataFrame with an ``RH`` column and ``meta`` location.
        df_doodtij: Optional DataFrame with dead-tide moments for annotation.
        start: Optional start timestamp for selecting groundwater data.
        end: End timestamp for both subplots.
        col_gwl: Groundwater level column name inside each observation.

    Returns:
        Tuple ``(fig, axes)`` with the generated Matplotlib figure and axes array.
    """
    fig, axes = plt.subplots(nrows=2, figsize=(
        12, 4), gridspec_kw={"hspace": 0.35})

    # bovenste plot
    # peilbuis
    for index, row in obs_gws.iterrows():
        row.obs[col_gwl].loc[start:end].plot(
            ax=axes[0], linewidth=1.5, label=f"peilbuis {row.name.split('_')[-2]}, BKF:{row.screen_top:.2f} m NAP")
    first_obs = row.obs[col_gwl].loc[start:end].dropna().index[0]
    axes[0].axvline(x=first_obs, color='darkgray', ls=':',
                    linewidth=1.5, label=f'eerste waarneming {first_obs:%d-%m-%Y}')

    # doodtij
    if df_doodtij is not None:
        plot_doodtij(axes[0], df_doodtij)

    # layout
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].tick_params(axis='x', labelrotation=0)
    axes[0].set_ylabel('m NAP')

    # onderste plot
    # neerslag
    # precip_df.loc[first_obs:end, 'cumulative'].plot(ax=axes[1], color='tab:red', linewidth=1.5, label=f'cummulatief sinds 1 {month_name}')
    daily_precip = precip_df.loc[first_obs:end, precip_col].resample(
        'D').sum() * 1000  # convert to mm
    axes[1].bar(
        daily_precip.index,
        daily_precip.values,
        width=0.9,
        color='tab:blue',
        label=f'Dagelijkse neerslagsom {precip_df.meta["location"].capitalize()}'
    )

    # layout
    axes[1].set_ylabel('neerslag (mm)')
    axes[1].legend(loc='upper left', fontsize=8)

    for ax in axes:
        ax.set_xlim([first_obs-pd.Timedelta('2d'), end])
        ax.grid(True)

    return fig, axes
