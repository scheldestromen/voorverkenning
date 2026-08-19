import os
import hydropandas as hpd
import pandas as pd
import logging
import json
import tempfile

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


# if __name__ == "__main__":
#    obs = json_to_hpd(
#        r'c:\data\python\cloned\N27-2\data\obs\json\json.zip'
#    )
