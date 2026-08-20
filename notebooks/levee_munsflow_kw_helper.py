import os
import hydropandas as hpd
import pandas as pd
import logging
import json
import tempfile
from tqdm.auto import tqdm

import matplotlib.dates as mdates

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
    """Decode string-encoded JSON fields so hydropandas from_dict can parse them."""
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

        for json_path in tqdm(json_files, desc="JSON inlezen", unit="bestand"):
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


def ax_lim_as_dates(ax):
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


def add_offset_for_close_points(oc, col_offset='distance_to_ref', suffix='plot', dup_offset=1.0):
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


def short_legend_precip(ax):
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
