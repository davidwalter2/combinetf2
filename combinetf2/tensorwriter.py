import math
import os
from collections import defaultdict

import h5py
import narf.ioutils
import numpy as np

from combinetf2 import common, h5pyutils


class TensorWriter:
    def __init__(
        self,
        sparse=False,
        exponential_transform=False,
        allow_negative_expectation=False,
    ):
        self.allow_negative_expectation = allow_negative_expectation
        self.exponential_transform = exponential_transform
        self.binByBinStatScale = 1.0
        self.symmetric_tensor = True  # If all shape systematics are symmetrized the systematic tensor is symmetric leading to reduced memory and improved efficiency

        self.signals = set()
        self.bkgs = set()

        self.channels = {}
        self.nbins = 0
        self.nbinschan = {}
        self.pseudodata_names = set()

        self.dict_noigroups = defaultdict(lambda: set())
        self.dict_systgroups = defaultdict(lambda: set())

        self.systsstandard = set()
        self.systsnoi = set()
        self.systsnoconstraint = set()
        self.systsnoprofile = set()

        self.sparse = sparse
        self.idxdtype = "int64"

        # temporary data
        self.dict_data_obs = {}  # [channel]
        self.data_covariance_inverted = None
        self.dict_pseudodata = {}  # [channel][pseudodata]
        self.dict_norm = {}  # [channel][process]
        self.dict_sumw2 = {}  # [channel]
        self.dict_logkavg = {}  # [channel][proc][syst]
        self.dict_logkhalfdiff = {}  # [channel][proc][syst]
        self.dict_logkavg_indices = {}
        self.dict_logkhalfdiff_indices = {}

        self.clipSystVariations = False
        if self.clipSystVariations > 0.0:
            self.clip = np.abs(np.log(self.clipSystVariations))

        self.logkepsilon = math.log(
            1e-3
        )  # numerical cutoff in case of zeros in systematic variations

        # settings for writing out hdf5 files
        self.dtype = "float64"
        self.chunkSize = 4 * 1024**2

    def get_flat_values(self, h):
        return h.values().flatten().astype(self.dtype)

    def get_flat_variances(self, h):
        return h.variances().flatten().astype(self.dtype)

    def add_data(self, h, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if channel in self.dict_data_obs.keys():
            raise RuntimeError(f"Data histogram for channel '{channel}' already set.")
        self.dict_data_obs[channel] = self.get_flat_values(h)

    def add_data_covariance(self, cov):
        cov = cov if isinstance(cov, np.ndarray) else cov.values()
        cov = np.reshape(cov, (self.nbins, self.nbins))
        self.data_covariance_inverted = np.linalg.inv(cov)

    def add_data_covariance_inverted(self, cov_inv):
        cov_inv = cov_inv if isinstance(cov_inv, np.ndarray) else cov_inv.values()
        cov_inv = np.reshape(cov_inv, (self.nbins, self.nbins))
        self.data_covariance_inverted = cov_inv

    def add_pseudodata(self, h, name=None, channel="ch0"):
        self._check_hist_and_channel(h, channel)
        if name is None:
            name = f"pseudodata_{len(self.pseudodata_names)}"
        self.pseudodata_names.add(name)
        if channel not in self.dict_pseudodata.keys():
            self.dict_pseudodata[channel] = {}
        if name in self.dict_pseudodata[channel].keys():
            raise RuntimeError(
                f"Pseudodata histogram '{name}' for channel '{channel}' already set."
            )
        self.dict_pseudodata[channel][name] = self.get_flat_values(h)

    def add_process(self, h, name, channel="ch0", signal=False):
        self._check_hist_and_channel(h, channel)

        if name in self.dict_norm[channel].keys():
            raise RuntimeError(
                f"Nominal histogram for process '{name}' for channel '{channel}' already set."
            )

        if signal:
            self.signals.add(name)
        else:
            self.bkgs.add(name)

        self.dict_logkavg[channel][name] = {}
        self.dict_logkhalfdiff[channel][name] = {}
        if self.sparse:
            self.dict_logkavg_indices[channel][name] = {}
            self.dict_logkhalfdiff_indices[channel][name] = {}

        norm = self.get_flat_values(h)
        sumw2 = self.get_flat_variances(h)

        if not self.allow_negative_expectation:
            norm = np.maximum(norm, 0.0)
        if not np.all(np.isfinite(sumw2)):
            raise RuntimeError(
                f"{len(sumw2)-sum(np.isfinite(sumw2))} NaN or Inf values encountered in variances for {name}!"
            )
        if not np.all(np.isfinite(norm)):
            raise RuntimeError(
                f"{len(norm)-sum(np.isfinite(norm))} NaN or Inf values encountered in nominal histogram for {name}!"
            )

        self.dict_norm[channel][name] = norm
        self.dict_sumw2[channel] += sumw2

    def add_channel(self, axes, name=None):
        if name is None:
            name = f"ch{len(self.channels)}"
        self.channels[name] = {"axes": axes}
        ibins = np.product([len(a) for a in axes])
        self.nbins += ibins
        self.nbinschan[name] = ibins
        self.dict_norm[name] = {}
        self.dict_sumw2[name] = np.zeros(ibins)
        self.dict_logkavg[name] = {}
        self.dict_logkhalfdiff[name] = {}
        if self.sparse:
            self.dict_logkavg_indices[name] = {}
            self.dict_logkhalfdiff_indices[name] = {}

    def _check_hist_and_channel(self, h, channel, add=True):
        axes = [a for a in h.axes]
        if channel not in self.channels.keys():
            if add:
                print(f"Add new channel {channel}")
                self.add_channel(axes, channel)
            else:
                raise RuntimeError(f"Channel {channel} not known!")
        elif axes != self.channels[channel]["axes"]:
            raise RuntimeError(
                f"Histogram axes {axes} different from channel axes {self.channels[channel]["axes"]} of channel {channel}"
            )

    def add_lnN_systematic(
        self,
        name,
        process,
        channel,
        uncertainty,
        profile=True,
        groups=None,
    ):
        norm = self.dict_norm[channel][process]
        syst = norm * uncertainty

        logk = self.get_logk(syst, norm)
        self.book_logk_avg(logk, channel, process, name)
        self.book_systematic(name, groups=groups, profile=profile)

    def add_systematic(
        self,
        h,
        name,
        process,
        channel,
        kfactor=1,
        mirror=True,
        symmetrize="average",
        **kargs,
    ):
        """
        h: either a single histogram with the systematic variation if mirror=True or a list of two histograms with the up and down variation
        """

        norm = self.dict_norm[channel][process]

        if isinstance(h, (list, tuple, np.ndarray)):
            self._check_hist_and_channel(h[0], channel, add=False)
            self._check_hist_and_channel(h[1], channel, add=False)

            syst_up = self.get_flat_values(h[0])
            syst_down = self.get_flat_values(h[1])
            if symmetrize is not None:

                logkup_proc = self.get_logk(syst_up, norm, kfactor)
                logkdown_proc = -self.get_logk(syst_down, norm, kfactor)

                if symmetrize == "conservative":
                    # symmetrize by largest magnitude of up and down variations
                    logkavg_proc = np.where(
                        np.abs(logkup_proc) > np.abs(logkdown_proc),
                        logkup_proc,
                        logkdown_proc,
                    )
                elif symmetrize == "average":
                    # symmetrize by average of up and down variations
                    logkavg_proc = 0.5 * (logkup_proc + logkdown_proc)
                elif symmetrize in ["linear", "quadratic"]:
                    # "linear" corresponds to a piecewise linear dependence of logk on theta
                    # while "quadratic" corresponds to a quadratic dependence and leads
                    # to a large variance
                    diff_fact = np.sqrt(3.0) if symmetrize == "quadratic" else 1.0

                    # split asymmetric variation into two symmetric variations
                    logkavg_proc = 0.5 * (logkup_proc + logkdown_proc)
                    logkdiffavg_proc = 0.5 * diff_fact * (logkup_proc - logkdown_proc)

                    var_name_out = name + "SymAvg"
                    var_name_out_diff = name + "SymDiff"

                    # special case, book the extra systematic
                    self.book_logk_avg(
                        logkdiffavg_proc, channel, process, var_name_out_diff
                    )
                    self.book_systematic(var_name_out_diff, **kargs)
            else:
                self.symmetric_tensor = False
                logkup_proc = self.get_logk(syst_up, norm, kfactor)
                logkdown_proc = -self.get_logk(syst_down, norm, kfactor)

                logkavg_proc = 0.5 * (logkup_proc + logkdown_proc)
                logkhalfdiff_proc = 0.5 * (logkup_proc - logkdown_proc)

                logkup_proc = None
                logkdown_proc = None

                self.book_logk_halfdiff(logkhalfdiff_proc, channel, process, name)
        elif mirror:
            self._check_hist_and_channel(h, channel, add=False)
            syst = self.get_flat_values(h)
            logkavg_proc = self.get_logk(syst, norm, kfactor)
        else:
            raise RuntimeError(
                "Only one histogram given but mirror=False, can not construct a variation"
            )

        self.book_logk_avg(logkavg_proc, channel, process, name)
        self.book_systematic(name, **kargs)

    def get_logk(self, syst, norm, kfac=1.0):
        if not np.all(np.isfinite(syst)):
            raise RuntimeError(
                f"{len(syst)-sum(np.isfinite(syst))} NaN or Inf values encountered in systematic!"
            )

        # check if there is a sign flip between systematic and nominal
        if self.exponential_transform:
            _logk = kfac * (syst - norm) / 100000
        else:
            _logk = kfac * np.log(syst / norm)
        _logk_view = np.where(
            np.equal(np.sign(norm * syst), 1),
            _logk,
            self.logkepsilon * np.ones_like(_logk),
        )

        if self.clipSystVariations > 0.0:
            _logk = np.clip(_logk, -self.clip, self.clip)
        if self.exponential_transform:
            return _logk
        else:
            return _logk_view

    def book_logk_avg(self, *args):
        self.book_logk(
            self.dict_logkavg,
            self.dict_logkavg_indices,
            *args,
        )

    def book_logk_halfdiff(self, *args):
        self.book_logk(
            self.dict_logkhalfdiff,
            self.dict_logkhalfdiff_indices,
            *args,
        )

    def book_logk(
        self,
        dict_logk,
        dict_logk_indices,
        logk,
        channel,
        process,
        syst_name,
    ):
        norm = self.dict_norm[channel][process]
        # ensure that systematic tensor is sparse where normalization matrix is sparse
        logk = np.where(np.equal(norm, 0.0), 0.0, logk)
        if self.sparse:
            indices = np.transpose(np.nonzero(logk))
            dict_logk_indices[channel][process][syst_name] = indices
            dict_logk[channel][process][syst_name] = np.reshape(logk[indices], [-1])
        else:
            dict_logk[channel][process][syst_name] = logk

    def book_systematic(
        self,
        name,
        profile=True,
        noi=False,
        constrained=True,
        groups=None,
    ):
        print(f"Book systematic {name}")
        if not profile:
            self.systsnoprofile.add(name)
        elif noi:
            self.systsnoi.add(name)
        elif not constrained:
            self.systsnoconstraint.add(name)
        else:
            self.systsstandard.add(name)

        if groups is None:
            groups = [name]

        if noi:
            target_dict = self.dict_noigroups
        else:
            target_dict = self.dict_systgroups

        for group in groups:
            target_dict[group].add(name)

    def write(self, outfolder="./", outfilename="combinetf2_input.hdf5", args={}):

        if self.signals.intersection(self.bkgs):
            raise RuntimeError(
                f"Processes '{self.signals.intersection(self.bkgs)}' found as signal and background"
            )

        procs = list(self.signals) + list(self.bkgs)
        nproc = len(procs)

        print(f"Write out nominal arrays")
        sumw = np.zeros([self.nbins], self.dtype)
        sumw2 = np.zeros([self.nbins], self.dtype)
        data_obs = np.zeros([self.nbins], self.dtype)
        pseudodata = np.zeros([self.nbins, len(self.pseudodata_names)], self.dtype)
        ibin = 0
        for chan, axes in self.channels.items():
            nbinschan = self.nbinschan[chan]
            data_obs[ibin : ibin + nbinschan] = self.dict_data_obs[chan]
            sumw2[ibin : ibin + nbinschan] = (
                self.binByBinStatScale**2 * self.dict_sumw2[chan]
            )

            for idx, name in enumerate(self.pseudodata_names):
                pseudodata[ibin : ibin + nbinschan, idx] = self.dict_pseudodata[chan][
                    name
                ]

            for iproc, proc in enumerate(procs):
                if proc not in self.dict_norm[chan]:
                    continue

                sumw[ibin : ibin + nbinschan] += self.dict_norm[chan][proc]

            ibin += nbinschan

        systs = self.get_systs()
        nsyst = len(systs)

        if self.symmetric_tensor:
            print("No asymmetric systematics - write fully symmetric tensor")

        ibin = 0
        if self.sparse:
            print(f"Write out sparse array")
            norm_sparse_size = 0
            norm_sparse_indices = np.zeros([norm_sparse_size, 2], self.idxdtype)
            norm_sparse_values = np.zeros([norm_sparse_size], self.dtype)

            logk_sparse_size = 0
            logk_sparse_normindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_systindices = np.zeros([logk_sparse_size, 1], self.idxdtype)
            logk_sparse_values = np.zeros([logk_sparse_size], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]
                dict_logkavg_chan_indices = self.dict_logkavg_indices[chan]
                dict_logkavg_chan_values = self.dict_logkavg[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue
                    norm_proc = dict_norm_chan[proc]

                    if self.exponential_transform:
                        norm_proc = np.exp(norm_proc / 100000)

                    norm_indices = np.transpose(np.nonzero(norm_proc))
                    norm_values = np.reshape(norm_proc[norm_indices], [-1])

                    nvals = len(norm_values)
                    oldlength = norm_sparse_size
                    norm_sparse_size = oldlength + nvals
                    norm_sparse_indices.resize([norm_sparse_size, 2])
                    norm_sparse_values.resize([norm_sparse_size])

                    out_indices = np.array([[ibin, iproc]]) + np.pad(
                        norm_indices, ((0, 0), (0, 1)), "constant"
                    )
                    norm_indices = None

                    norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
                    out_indices = None

                    norm_sparse_values[oldlength:norm_sparse_size] = norm_values
                    norm_values = None

                    norm_idx_map = (
                        np.cumsum(np.not_equal(norm_proc, 0.0)) - 1 + oldlength
                    )

                    dict_logkavg_proc_indices = dict_logkavg_chan_indices[proc]
                    dict_logkavg_proc_values = dict_logkavg_chan_values[proc]

                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc_indices.keys():
                            continue

                        logkavg_proc_indices = dict_logkavg_proc_indices[syst]
                        logkavg_proc_values = dict_logkavg_proc_values[syst]

                        nvals_proc = len(logkavg_proc_values)
                        oldlength = logk_sparse_size
                        logk_sparse_size = oldlength + nvals_proc
                        logk_sparse_normindices.resize([logk_sparse_size, 1])
                        logk_sparse_systindices.resize([logk_sparse_size, 1])
                        logk_sparse_values.resize([logk_sparse_size])

                        # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                        # second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
                        # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                        out_normindices = norm_idx_map[logkavg_proc_indices]
                        logkavg_proc_indices = None

                        logk_sparse_normindices[oldlength:logk_sparse_size] = (
                            out_normindices
                        )
                        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
                        out_normindices = None

                        logk_sparse_values[oldlength:logk_sparse_size] = (
                            logkavg_proc_values
                        )
                        logkavg_proc_values = None

                        if syst in self.dict_logkhalfdiff_indices[chan][proc].keys():
                            logkhalfdiff_proc_indices = self.dict_logkhalfdiff_indices[
                                chan
                            ][proc][syst]
                            logkhalfdiff_proc_values = self.dict_logkhalfdiff[chan][
                                proc
                            ][syst]

                            nvals_proc = len(logkhalfdiff_proc_values)
                            oldlength = logk_sparse_size
                            logk_sparse_size = oldlength + nvals_proc
                            logk_sparse_normindices.resize([logk_sparse_size, 1])
                            logk_sparse_systindices.resize([logk_sparse_size, 1])
                            logk_sparse_values.resize([logk_sparse_size])

                            # out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_proc_indices,((0,0),(0,3)),'constant')
                            # first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
                            # second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
                            # two dimensions are kept in separate arrays for now to reduce the number of copies needed later
                            out_normindices = norm_idx_map[logkhalfdiff_proc_indices]
                            logkhalfdiff_proc_indices = None

                            logk_sparse_normindices[oldlength:logk_sparse_size] = (
                                out_normindices
                            )
                            logk_sparse_systindices[oldlength:logk_sparse_size] = (
                                nsyst + isyst
                            )
                            out_normindices = None

                            logk_sparse_values[oldlength:logk_sparse_size] = (
                                logkhalfdiff_proc_values
                            )
                            logkhalfdiff_proc_values = None

                    # free memory
                    dict_logkavg_proc_indices = None
                    dict_logkavg_proc_values = None

                # free memory
                norm_proc = None
                norm_idx_map = None

                ibin += nbinschan

            print(f"Resize and sort sparse arrays into canonical order")
            # resize sparse arrays to actual length
            norm_sparse_indices.resize([norm_sparse_size, 2])
            norm_sparse_values.resize([norm_sparse_size])
            logk_sparse_normindices.resize([logk_sparse_size, 1])
            logk_sparse_systindices.resize([logk_sparse_size, 1])
            logk_sparse_values.resize([logk_sparse_size])

            # straightforward sorting of norm_sparse into canonical order
            norm_sparse_dense_shape = (self.nbins, nproc)
            norm_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(norm_sparse_indices), norm_sparse_dense_shape
                )
            )
            norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
            norm_sparse_values = norm_sparse_values[norm_sort_indices]

            # now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse

            # compute the inverse permutation from the sorting of norm_sparse
            # since the final indices are filled from here, need to ensure it has the correct data type
            logk_permute_indices = np.argsort(norm_sort_indices).astype(self.idxdtype)
            norm_sort_indices = None
            logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
            logk_permute_indices = None
            logk_sparse_indices = np.concatenate(
                [logk_sparse_normindices, logk_sparse_systindices], axis=-1
            )

            # now straightforward sorting of logk_sparse into canonical order
            if self.symmetric_tensor:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], nsyst)
            else:
                logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2 * nsyst)
            logk_sort_indices = np.argsort(
                np.ravel_multi_index(
                    np.transpose(logk_sparse_indices), logk_sparse_dense_shape
                )
            )
            logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
            logk_sparse_values = logk_sparse_values[logk_sort_indices]
            logk_sort_indices = None

        else:
            print(f"Write out dense array")
            # initialize with zeros, i.e. no variation
            norm = np.zeros([self.nbins, nproc], self.dtype)
            if self.symmetric_tensor:
                logk = np.zeros([self.nbins, nproc, nsyst], self.dtype)
            else:
                logk = np.zeros([self.nbins, nproc, 2, nsyst], self.dtype)

            for chan in self.channels.keys():
                nbinschan = self.nbinschan[chan]
                dict_norm_chan = self.dict_norm[chan]

                for iproc, proc in enumerate(procs):
                    if proc not in dict_norm_chan:
                        continue

                    norm_proc = dict_norm_chan[proc]
                    if self.exponential_transform:
                        norm_proc = np.exp(norm_proc / 100000)

                    norm[ibin : ibin + nbinschan, iproc] = norm_proc

                    dict_logkavg_proc = self.dict_logkavg[chan][proc]
                    dict_logkhalfdiff_proc = self.dict_logkhalfdiff[chan][proc]
                    for isyst, syst in enumerate(systs):
                        if syst not in dict_logkavg_proc.keys():
                            continue

                        if self.symmetric_tensor:
                            logk[ibin : ibin + nbinschan, iproc, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                        else:
                            logk[ibin : ibin + nbinschan, iproc, 0, isyst] = (
                                dict_logkavg_proc[syst]
                            )
                            if syst in dict_logkhalfdiff_proc.keys():
                                logk[ibin : ibin + nbinschan, iproc, 1, isyst] = (
                                    dict_logkhalfdiff_proc[syst]
                                )

                ibin += nbinschan

        # compute poisson parameter for Barlow-Beeston bin-by-bin statistical uncertainties
        kstat = np.square(sumw) / sumw2
        # numerical protection to avoid poorly defined constraint
        kstat = np.where(np.equal(sumw, 0.0), 1.0, kstat)
        kstat = np.where(np.equal(sumw2, 0.0), 1.0, kstat)

        # write results to hdf5 file
        procSize = nproc * np.dtype(self.dtype).itemsize
        systSize = 2 * nsyst * np.dtype(self.dtype).itemsize
        amax = np.max([procSize, systSize])
        if amax > self.chunkSize:
            print(
                f"Maximum chunk size in bytes was increased from {self.chunkSize} to {amax} to align with tensor sizes and allow more efficient reading/writing."
            )
            self.chunkSize = amax

        # create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
        if not os.path.isdir(outfolder):
            os.makedirs(outfolder)
        outpath = f"{outfolder}/{outfilename}"
        if len(outfilename.split(".")) < 2:
            outpath += ".hdf5"
        print(f"Write output file {outpath}")
        f = h5py.File(outpath, rdcc_nbytes=self.chunkSize, mode="w")

        # propagate meta info into result file
        meta = {
            "meta_info": narf.ioutils.make_meta_info_dict(
                args=args, wd=common.base_dir
            ),
            "channel_info": self.channels,
            "symmetric_tensor": self.symmetric_tensor,
        }

        narf.ioutils.pickle_dump_h5py("meta", meta, f)

        systsnoprofile = self.get_systsnoprofile()
        systsnoconstraint = self.get_systsnoconstraint()
        noigroups, noigroupidxs = self.get_noigroups()
        systgroups, systgroupidxs = self.get_systgroups()

        # save some lists of strings to the file for later use
        def create_dataset(
            name,
            content,
            length=None,
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",
        ):
            dimension = [len(content), length] if length else [len(content)]
            ds = f.create_dataset(
                f"h{name}", dimension, dtype=dtype, compression=compression
            )
            ds[...] = content

        create_dataset("procs", procs)
        create_dataset("signals", [s for s in self.signals])
        create_dataset("systs", systs)
        create_dataset("systsnoprofile", systsnoprofile)
        create_dataset("systsnoconstraint", systsnoconstraint)
        create_dataset("systgroups", systgroups)
        create_dataset(
            "systgroupidxs",
            systgroupidxs,
            dtype=h5py.special_dtype(vlen=np.dtype("int32")),
        )
        create_dataset("noigroups", noigroups)
        create_dataset("noigroupidxs", noigroupidxs, dtype="int32")
        create_dataset("pseudodatanames", [n for n in self.pseudodata_names])

        # create h5py datasets with optimized chunk shapes
        nbytes = 0

        constraintweights = self.get_constraintweights(self.dtype)
        nbytes += h5pyutils.writeFlatInChunks(
            constraintweights, f, "hconstraintweights", maxChunkBytes=self.chunkSize
        )
        constraintweights = None

        nbytes += h5pyutils.writeFlatInChunks(
            data_obs, f, "hdata_obs", maxChunkBytes=self.chunkSize
        )
        data_obs = None

        nbytes += h5pyutils.writeFlatInChunks(
            pseudodata, f, "hpseudodata", maxChunkBytes=self.chunkSize
        )
        pseudodata = None

        if self.data_covariance_inverted is not None:
            nbytes += h5pyutils.writeFlatInChunks(
                self.data_covariance_inverted,
                f,
                "hdata_cov_inv",
                maxChunkBytes=self.chunkSize,
            )

        nbytes += h5pyutils.writeFlatInChunks(
            kstat, f, "hkstat", maxChunkBytes=self.chunkSize
        )
        kstat = None

        if self.sparse:
            nbytes += h5pyutils.writeSparse(
                norm_sparse_indices,
                norm_sparse_values,
                norm_sparse_dense_shape,
                f,
                "hnorm_sparse",
                maxChunkBytes=self.chunkSize,
            )
            norm_sparse_indices = None
            norm_sparse_values = None
            nbytes += h5pyutils.writeSparse(
                logk_sparse_indices,
                logk_sparse_values,
                logk_sparse_dense_shape,
                f,
                "hlogk_sparse",
                maxChunkBytes=self.chunkSize,
            )
            logk_sparse_indices = None
            logk_sparse_values = None
        else:
            nbytes += h5pyutils.writeFlatInChunks(
                norm, f, "hnorm", maxChunkBytes=self.chunkSize
            )
            norm = None
            nbytes += h5pyutils.writeFlatInChunks(
                logk, f, "hlogk", maxChunkBytes=self.chunkSize
            )
            logk = None

        print(f"Total raw bytes in arrays = {nbytes}")

    def get_systsstandard(self):
        return list(common.natural_sort(self.systsstandard))

    def get_systsnoprofile(self):
        return list(common.natural_sort(self.systsnoprofile))

    def get_systsnoi(self):
        return list(common.natural_sort(self.systsnoi))

    def get_systsnoconstraint(self):
        return self.get_systsnoi() + list(common.natural_sort(self.systsnoconstraint))

    def get_systs(self):
        return (
            self.get_systsnoconstraint()
            + self.get_systsstandard()
            + self.get_systsnoprofile()
        )

    def get_constraintweights(self, dtype):
        systs = self.get_systs()
        constraintweights = np.ones([len(systs)], dtype=dtype)
        for syst in self.get_systsnoconstraint():
            constraintweights[systs.index(syst)] = 0.0
        return constraintweights

    def get_groups(self, group_dict):
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(group_dict).items():
            groups.append(group)
            idx = []
            for syst in members:
                idx.append(systs.index(syst))
            idxs.append(idx)
        return groups, idxs

    def get_noigroups(self):
        # list of groups of systematics to be treated as additional outputs for impacts, etc (aka "nuisances of interest")
        systs = self.get_systs()
        groups = []
        idxs = []
        for group, members in common.natural_sort_dict(self.dict_noigroups).items():
            groups.append(group)
            for syst in members:
                idxs.append(systs.index(syst))
        return groups, idxs

    def get_systgroups(self):
        # list of groups of systematics (nuisances) and lists of indexes
        return self.get_groups(self.dict_systgroups)