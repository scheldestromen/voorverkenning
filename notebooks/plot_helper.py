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
