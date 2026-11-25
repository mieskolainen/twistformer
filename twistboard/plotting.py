# Shared plot style and exports for training and particle diagnostics
#
# mikael.mieskolainen@cern.ch (2026)

from functools import wraps
from pathlib import Path

PNG_DPI = 200

PLOT_COLORS = ("#0057B8", "#D62728", "#228833", "#AA3377", "#222222", "#0099BB", "#8A7900")

def save_figure(figure, path):
    """
    Export PNG figures at higher resolution and preserve vector export settings

    Args:
        figure:  Matplotlib figure
        path:  Output file path
    Returns:
        None: Figure is written to disk
    """
    options = {"dpi": PNG_DPI} if Path(path).suffix.lower() == ".png" else {}
    figure.savefig(path, **options)

def plot_style(function):
    """
    Apply the shared color cycle only while rendering diagnostic figures

    Args:
        function:  Plotting function with lazy Matplotlib imports
    Returns:
        wrapped: Plotting function that restores the caller's style on exit
    """

    @wraps(function)
    def wrapped(*args, **kwargs):
        """
        Render figures with distinct blue, red and green series colors

        Args:
            args:  Positional plotting arguments
            kwargs:  Named plotting arguments
        Returns:
            result: Return value from the plotting function
        """
        import matplotlib
        from cycler import cycler

        with matplotlib.rc_context({"axes.prop_cycle": cycler(color=PLOT_COLORS)}):
            return function(*args, **kwargs)

    return wrapped
