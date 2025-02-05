import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from narf import ioutils

from combinetf2 import io_tools

plt.rcParams.update({"font.size": 14})


def writeOutput(fig, outfile, extensions=[], postfix=None, args=None, meta_info=None):
    name, _ = os.path.splitext(outfile)

    if postfix:
        name += f"_{postfix}"

    for ext in extensions:
        if ext[0] != ".":
            ext = "." + ext
        output = name + ext
        print(f"Write output file {output}")
        plt.savefig(output)

        output = name.rsplit("/", 1)
        output[1] = os.path.splitext(output[1])[0]
        if len(output) == 1:
            output = (None, *output)
    if args is None and meta_info is None:
        return
    ioutils.write_logfile(
        *output,
        args=args,
        meta_info=meta_info,
    )


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputFile",
        type=str,
        help="fitresults output",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        default="./test",
        help="Folder path for output",
    )
    parser.add_argument(
        "-p", "--postfix", type=str, help="Postfix for output file name"
    )
    parser.add_argument(
        "--params",
        type=str,
        nargs=2,
        action="append",
        help="Parameters to plot the likelihood scan",
    )
    parser.add_argument(
        "--title",
        default="CombineTF2",
        type=str,
        help="Title to be printed in upper left",
    )
    parser.add_argument(
        "--subtitle",
        default=None,
        type=str,
        help="Subtitle to be printed after title",
    )
    return parser.parse_args()


def ellipse(cov, mu0, mu1, cl, cartesian_angle=False):

    a = cov[0, 0]
    b = cov[1, 0]
    c = cov[1, 1]

    l1 = (a + c) / 2 + np.sqrt((a - c) ** 2 / 4 + b**2)
    l2 = (a + c) / 2 - np.sqrt((a - c) ** 2 / 4 + b**2)

    theta = np.arctan2(l1 - a, b)

    def func(t):
        x = (
            mu0
            + np.sqrt(l1) * np.cos(theta) * np.cos(t)
            - np.sqrt(l2) * np.sin(theta) * np.sin(t)
        )
        y = (
            mu1
            + np.sqrt(l1) * np.sin(theta) * np.cos(t)
            + np.sqrt(l2) * np.cos(theta) * np.sin(t)
        )
        return x, y

    return func


def plot_scan(
    px,
    py,
    cov,
    h_scan,
    h_contour,
    xlabel="x",
    ylabel="y",
    confidence_levels=[
        1.0,
    ],
    n_points=100,
    title=None,
    subtitle=None,
):

    # Parameterize ellipse
    t = np.linspace(0, 2 * np.pi, n_points)

    if h_scan is not None:
        right = 0.97
    else:
        right = 0.99

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.subplots_adjust(left=0.16, bottom=0.14, right=right, top=0.94)

    if h_scan is not None:
        x_scan = np.array(h_scan.axes["scan_x"]).astype(float)
        y_scan = np.array(h_scan.axes["scan_y"]).astype(float)
        nll_values = h_scan.values()
        plt.pcolormesh(x_scan, y_scan, 2 * nll_values, shading="auto", cmap="plasma")
        plt.colorbar(label=r"$-2\,\Delta \log L$")

    for cl in confidence_levels:
        xy = ellipse(cov, px, py, cl)(t)
        ax.plot(xy[0], xy[1], label=f"{cl}σ")

        if h_contour is not None and str(cl) in h_contour.axes["confidence_level"]:
            x_contour = h_contour[{"confidence_level": str(cl), "params": 0}].values()
            y_contour = h_contour[{"confidence_level": str(cl), "params": 1}].values()
            ax.plot(
                x_contour,
                y_contour,
                marker="o",
                markerfacecolor="none",
                color="black",
                label=f"Contour scan {cl}σ",
            )

    # Plot mean point
    ax.scatter(px, py, color="red", marker="x", label="Best fit")

    textsize = ax.xaxis.label.get_size()

    if title:
        ax.text(
            0.0,
            1.01,
            title,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=1.2 * textsize,
        )
    if subtitle:
        ax.text(0.45, 1.01, subtitle, transform=ax.transAxes, fontstyle="italic")

    # Formatting
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(py, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(px, color="gray", linestyle="--", alpha=0.5)

    ax.legend(loc="upper right")

    return fig


if __name__ == "__main__":
    args = parseArgs()
    fitresult, meta = io_tools.get_fitresult(args.inputFile, meta=True)

    meta = {
        "combinetf2": meta["meta_info"],
    }

    h_params = fitresult["parms"].get()

    h_cov = None
    if "cov" in fitresult.keys():
        h_cov = fitresult["cov"].get()

    h_contour = None
    if "contour_scans2D" in fitresult.keys():
        h_contour = fitresult["contour_scans2D"].get()

    for px, py in args.params:
        px_value = h_params[{"parms": px}].value
        py_value = h_params[{"parms": py}].value

        cov = None
        if h_cov is not None:
            cov = h_cov[{"parms_x": [px, py], "parms_y": [px, py]}].values()

        h_contour_params = None
        if h_contour is not None and f"{px}-{py}" in h_contour.axes["param_tuple"]:
            h_contour_params = h_contour[{"param_tuple": f"{px}-{py}"}]

        h_scan = None
        if f"nll_scan2D_{px}_{py}" in fitresult.keys():
            h_scan = fitresult[f"nll_scan2D_{px}_{py}"].get()

        fig = plot_scan(
            px_value,
            py_value,
            cov,
            h_scan,
            h_contour_params,
            xlabel=px,
            ylabel=py,
            title=args.title,
            subtitle=args.subtitle,
        )
        os.makedirs(args.outpath, exist_ok=True)
        outfile = os.path.join(args.outpath, f"nll_scan2D_{px}_{py}")
        writeOutput(
            fig,
            outfile=outfile,
            extensions=["png", "pdf"],
            meta_info=meta,
            args=args,
            postfix=args.postfix,
        )
