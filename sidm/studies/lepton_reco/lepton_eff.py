"""
Lepton reconstruction efficiency vs. dark photon Lxy.

Works on the existing "<obj>_nearGenA_n_genA_lxy" histograms: for each gen
dark photon, they record (Lxy, number of reco objects within dR of it).

    efficiency(Lxy) = (# dark photons with >= min_n matched reco objects)
                      / (# dark photons)
"""
import numpy as np
import matplotlib.pyplot as plt
import hist
import hist.intervals

OBJECTS = {
    "electron": "Electrons",
    "photon": "Photons",
    "muon": "PF muons",
    "dsaMuon": "DSA muons",
}


def _inner(ax_):
    """Slice that strips the flow bins from one axis of values(flow=True)."""
    lo = 1 if ax_.traits.underflow else 0
    hi = -1 if ax_.traits.overflow else None
    return slice(lo, hi)


def reco_efficiency(h, min_n=1, channel="base_sr", lxy_axis=0, n_axis=1):
    """
    Return (lxy_edges, eff, err_low, err_high) for one histogram.

    The n axis overflow is included, so "n >= min_n" really means "or more".
    Lxy over/underflow bins are dropped.
    """
    if channel is not None:
        h = h[channel, ...]

    vals = h.values(flow=True)
    vals = np.moveaxis(vals, (lxy_axis, n_axis), (0, 1))
    vals = vals[_inner(h.axes[lxy_axis]), :]        # drop Lxy flow bins only

    # n bin index -> n value (flow=True puts underflow first if it exists)
    n_ax = h.axes[n_axis]
    offset = 1 if n_ax.traits.underflow else 0
    first = offset + n_ax.index(min_n)

    num = vals[:, first:].sum(axis=1)                # n >= min_n (incl. overflow)
    den = vals.sum(axis=1)                           # every dark photon

    eff = np.divide(num, den, out=np.full_like(num, np.nan, dtype=float),
                    where=den > 0)
    # Clopper-Pearson interval; appropriate for unweighted counts
    with np.errstate(invalid="ignore", divide="ignore"):
        lo, hi = hist.intervals.ratio_uncertainty(
            num, np.where(den > 0, den, 1), uncertainty_type="efficiency")
    lo[den == 0] = hi[den == 0] = 0
    return h.axes[lxy_axis].edges, eff, lo, hi


def plot_efficiency_vs_lxy(outputs, sample, min_n=1, channel="base_sr",
                           lxy_max=None, title=None, savepath=None):
    """
    outputs: dict {legend label: coffea output}, e.g.
        {"LJ source cuts": output_300cm,
         "No LJ source cuts": output_300cm_noLjSourceCuts}
    One panel per reco object type, one curve per output.
    """
    fig, axes = plt.subplots(len(OBJECTS), 1, figsize=(30, 10),
                             sharex=True)

    for ax, (obj, obj_label) in zip(axes, OBJECTS.items()):
        for label, out in outputs.items():
            h = out[sample]["hists"][f"{obj}_nearGenA_n_genA_lxy"]
            edges, eff, lo, hi = reco_efficiency(h, min_n, channel)
            centers = 0.5 * (edges[1:] + edges[:-1])
            ax.errorbar(centers, eff, yerr=[lo, hi],
                        xerr=0.5 * np.diff(edges), fmt="o", ms=3, label=label)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel(f"{obj_label}\neff. (n ≥ {min_n})")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[-1].set_xlabel("Dark photon $L_{xy}$ [cm]")
    if lxy_max is not None:
        axes[-1].set_xlim(0, lxy_max)
    fig.suptitle(title or sample)
    fig.subplots_adjust(hspace=0)
    if savepath:
        fig.savefig(savepath, bbox_inches="tight")
    return fig, axes
