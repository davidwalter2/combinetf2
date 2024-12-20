
import hist
import numpy as np
import scipy
import tensorflow as tf

class Fitter:
    def __init__(self, indata, workspace, options):
        self.indata = indata
        self.binByBinStat = options.binByBinStat
        self.normalize = options.normalize
        self.systgroupsfull = self.indata.systgroups.tolist()
        self.systgroupsfull.append("stat")
        if self.binByBinStat:
            self.systgroupsfull.append("binByBinStat")

        self.nsystgroupsfull = len(self.systgroupsfull)

        self.pois = []

        if options.POIMode == "mu":
            self.npoi = self.indata.nsignals
            poidefault = options.POIDefault*tf.ones([self.npoi],dtype=self.indata.dtype)
            for signal in self.indata.signals:
                self.pois.append(signal)
        elif options.POIMode == "none":
            self.npoi = 0
            poidefault = tf.zeros([],dtype=dtype)
        else:
            raise Exception("unsupported POIMode")

        self.parms = np.concatenate([self.pois, self.indata.systs])

        self.allowNegativePOI = options.allowNegativePOI

        if self.allowNegativePOI:
            xpoidefault = poidefault
        else:
            xpoidefault = tf.sqrt(poidefault)

        # tf variable containing all fit parameters
        thetadefault = tf.zeros([self.indata.nsyst],dtype=self.indata.dtype)
        if self.npoi>0:
            xdefault = tf.concat([xpoidefault,thetadefault], axis=0)
        else:
            xdefault = thetadefault

        self.x = tf.Variable(xdefault, trainable=True, name="x")

        # observed number of events per bin
        self.nobs = tf.Variable(self.indata.data_obs, trainable=False, name="nobs")

        if self.chisqFit:
            if self.externalCovariance:
                if self.indata.data_cov_inv is None:
                    raise RuntimeError("No external covariance found in input data.")
                # provided covariance
                self.data_cov_inv = self.indata.data_cov_inv
            else:
                # covariance from data stat
                if tf.math.reduce_any(self.nobs <= 0).numpy():
                    raise RuntimeError("Bins in 'nobs <= 0' encountered, chi^2 fit can not be performed.")
                self.data_cov_inv = tf.reciprocal(tf.linalg.diag(self.nobs))

        # constraint minima for nuisance parameters
        self.theta0 = tf.Variable(tf.zeros([self.indata.nsyst], dtype=self.indata.dtype), trainable=False, name="theta0")

        # global observables for mc stat uncertainty
        self.beta0 = tf.ones_like(self.indata.data_obs)

        nexpfullcentral = self.expected_events(profile=False)
        self.nexpnom = tf.Variable(nexpfullcentral, trainable=False, name="nexpnom")

        self.workspace = workspace
        self.hist = self.workspace.hist

    def prefit_covariance(self, unconstrained_err=0.):
        # free parameters are taken to have zero uncertainty for the purposes of prefit uncertainties
        var_poi = tf.zeros([self.npoi], dtype=self.indata.dtype)

        # nuisances have their uncertainty taken from the constraint term, but unconstrained nuisances
        # are set to a placeholder uncertainty (zero by default) for the purposes of prefit uncertainties
        var_theta = tf.where(self.indata.constraintweights == 0., unconstrained_err, tf.math.reciprocal(self.indata.constraintweights))

        invhessianprefit = tf.linalg.diag(tf.concat([var_poi, var_theta], axis = 0))
        return invhessianprefit

    @tf.function
    def val_jac(self, fun, *args, **kwargs):
        with tf.GradientTape() as t:
            val = fun(*args, **kwargs)
        jac = t.jacobian(val, self.x)

        return val, jac

    def bayesassign(self):
        if self.npoi > 0:
            raise NotImplementedError("Assignment for Bayesian toys is not currently supported in the presence of explicit POIs")
        self.x.assign(tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype))

    def frequentistassign(self):
        self.theta0.assign(tf.random.normal(shape=self.theta0.shape, dtype=self.theta0.dtype))

    def parms_hist(self, cov, hist_name="parms"):
        parms = list(self.parms.astype(str))
        axis_parms = hist.axis.StrCategory(parms, name="parms")

        values = self.x.numpy()
        variances = tf.linalg.diag_part(cov)

        h = self.hist(hist_name, axis_parms, values=values, variances=variances)

        return h

    def cov_hist(self, cov, hist_name="cov"):
        parms = list(self.parms.astype(str))
        axis_parms_x = hist.axis.StrCategory(parms, name="parms_x")
        axis_parms_y = hist.axis.StrCategory(parms, name="parms_y")

        h_cov = self.hist(hist_name, [axis_parms_x, axis_parms_y], values=cov)

        return h_cov

    @tf.function(reduce_retracing=True)
    def _compute_impact_group(self, cov, nstat, idxs):
        cov_reduced = tf.gather(cov, idxs, axis=0)
        cov_reduced = tf.gather(cov_reduced, idxs, axis=1)
        v = tf.gather(cov[:nstat], idxs, axis=1)
        invC_v = tf.linalg.solve(cov_reduced, tf.transpose(v))
        v_invC_v = tf.einsum('ij,ji->i', v, invC_v)
        return tf.sqrt(v_invC_v)

    @tf.function
    def _impacts_parms(self, nstat, cov, hess):
        #impact for poi at index i in covariance matrix from nuisance with index j is C_ij/sqrt(C_jj) = <deltax deltatheta>/sqrt(<deltatheta^2>)
        impacts = cov[:nstat]/tf.reshape(tf.sqrt(tf.linalg.diag_part(cov)),[1,-1])

        impacts_grouped = tf.map_fn(
            lambda idxs: self._compute_impact_group(cov, nstat, idxs), 
            tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32), 
            fn_output_signature=tf.TensorSpec(shape=(impacts.shape[0],), dtype=tf.float64)
        )
        impacts_grouped = tf.transpose(impacts_grouped)

        # impact data stat
        hess_stat = hess[:nstat,:nstat]
        identity = tf.eye(nstat, dtype=hess_stat.dtype)
        inv_hess_stat = tf.linalg.solve(hess_stat, identity)  # Solves H * X = I

        if self.binByBinStat:
            # impact bin-by-bin stat
            val_no_bbb, grad_no_bbb, hess_no_bbb = self.loss_val_grad_hess(profile_grad=False)

            hess_stat_no_bbb = hess_no_bbb[:nstat,:nstat]
            inv_hess_stat_no_bbb = tf.linalg.solve(hess_stat_no_bbb, identity)

            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat_no_bbb))
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))

            impacts_bbb_sq = tf.linalg.diag_part(inv_hess_stat - inv_hess_stat_no_bbb)
            impacts_bbb = tf.sqrt(tf.nn.relu(impacts_bbb_sq)) # max(0,x)
            impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = tf.concat([impacts_grouped, impacts_data_stat, impacts_bbb], axis=1)
        else:
            impacts_data_stat = tf.sqrt(tf.linalg.diag_part(inv_hess_stat))
            impacts_data_stat = tf.reshape(impacts_data_stat, (-1, 1))
            impacts_grouped = tf.concat([impacts_grouped, impacts_data_stat], axis=1)

        return impacts, impacts_grouped

    def impacts_hists(self, cov, hess):
        # store impacts for all POIs and unconstrained nuisances
        nstat = self.npoi + self.indata.nsystnoconstraint

        impacts, impacts_grouped = self._impacts_parms(nstat, cov, hess)

        parms = list(self.parms.astype(str))[:nstat]

        # write out histograms
        axis_parms = hist.axis.StrCategory(parms, name="parms")
        axis_impacts = self.indata.getImpactsAxes()
        axis_impacts_grouped = self.indata.getImpactsAxesGrouped(self.binByBinStat)

        h = self.hist("impacts", [axis_parms, axis_impacts], values=impacts)
        h_grouped = self.hist("impacts_grouped", [axis_parms, axis_impacts_grouped], values=impacts_grouped)

        return h, h_grouped

    def _global_impacts(self, cov):
        with tf.GradientTape() as t2:
            t2.watch([self.theta0, self.nobs, self.beta0])
            with tf.GradientTape() as t1:
                t1.watch([self.theta0, self.nobs, self.beta0])
                val = self._compute_loss()
            grad = t1.gradient(val, self.x, unconnected_gradients="zero")
        pd2ldxdtheta0, pd2ldxdnobs, pd2ldxdbeta0 = t2.jacobian(grad, [self.theta0, self.nobs, self.beta0], unconnected_gradients="zero")

        dxdtheta0 = -cov @ pd2ldxdtheta0
        dxdnobs = -cov @ pd2ldxdnobs
        dxdbeta0 = -cov @ pd2ldxdbeta0

        return dxdtheta0, dxdnobs, dxdbeta0

    @tf.function(reduce_retracing=True)
    def _compute_global_impact_group(self, d_squared, idxs):
        gathered = tf.gather(d_squared, idxs, axis=-1)
        d_squared_summed = tf.reduce_sum(d_squared, axis=-1)
        return tf.sqrt(d_squared_summed)

    @tf.function
    def _global_impacts_parms(self, cov):

        dxdtheta0, dxdnobs, dxdbeta0 = self._global_impacts(cov)
        
        dxdtheta0_squared = tf.square(dxdtheta0)
        impacts_grouped = tf.map_fn(
            lambda idxs: self._compute_global_impact_group(dxdtheta0_squared, idxs), 
            tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32), 
            fn_output_signature=tf.TensorSpec(shape=(dxdtheta0_squared.shape[0],), dtype=tf.float64)
        )
        impacts_grouped = tf.transpose(impacts_grouped)

        # global impact data stat
        if self.externalCovariance:
            data_cov = tf.linalg.inv(self.data_cov_inv)
            data_stat = tf.einsum('ij,jk,ik->i', dxdnobs, data_cov, dxdnobs)
        else:
            data_stat = tf.reduce_sum(tf.square(dxdnobs) * self.nobs, axis=-1)
        data_stat = tf.sqrt(data_stat)
        impacts_data_stat = tf.reshape(data_stat, (-1, 1))
        impacts_grouped = tf.concat([impacts_grouped, impacts_data_stat], axis=1)

        if self.binByBinStat:
            # global impact bin-by-bin stat
            impacts_bbb = tf.sqrt(tf.reduce_sum(tf.square(dxdbeta0) * tf.math.reciprocal(self.indata.kstat), axis=-1))
            impacts_bbb = tf.reshape(impacts_bbb, (-1, 1))
            impacts_grouped = tf.concat([impacts_grouped, impacts_bbb], axis=1)

        # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
        impacts = dxdtheta0[:,self.indata.nsystnoconstraint:]

        return impacts, impacts_grouped

    def global_impacts_hists(self, cov):
        # store impacts for all POIs and unconstrained nuisances
        nstat = self.npoi + self.indata.nsystnoconstraint

        impacts, impacts_grouped = self._global_impacts_parms(cov[:nstat])

        parms = list(self.parms.astype(str))[:nstat]

        # write out histograms
        axis_parms = hist.axis.StrCategory(parms, name="parms")
        axis_impacts = self.indata.getGlobalImpactsAxes()
        axis_impacts_grouped = self.indata.getImpactsAxesGrouped(self.binByBinStat)

        h = self.hist("global_impacts", [axis_parms, axis_impacts], values=impacts)
        h_grouped = self.hist("global_impacts_grouped", [axis_parms, axis_impacts_grouped], values=impacts_grouped)

        return h, h_grouped

    @tf.function
    def _expvar_profiled(self, fun_exp, cov, compute_cov=False, compute_global_impacts=False):
        dxdtheta0, dxdnobs, dxdbeta0 = self._global_impacts(cov)

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))

        pdexpdx, pdexpdtheta0, pdexpdnobs, pdexpdbeta0 = t.jacobian(expected_flat, [self.x, self.theta0, self.nobs, self.beta0], unconnected_gradients="zero")

        dexpdtheta0 = pdexpdtheta0 + pdexpdx @ dxdtheta0
        dexpdnobs = pdexpdnobs + pdexpdx @ dxdnobs
        dexpdbeta0 = pdexpdbeta0 + pdexpdx @ dxdbeta0

        # FIXME factorize this part better with the global impacts calculation

        var_theta0 = tf.where(self.indata.constraintweights == 0., tf.zeros_like(self.indata.constraintweights), tf.math.reciprocal(self.indata.constraintweights))

        dtheta0 = tf.math.sqrt(var_theta0)
        dnobs = tf.math.sqrt(self.nobs)
        dbeta0 = tf.math.sqrt(tf.math.reciprocal(self.indata.kstat))

        dexpdtheta0 *= dtheta0[None, :]
        dexpdnobs *= dnobs[None, :]
        dexpdbeta0 *= dbeta0[None, :]

        if compute_cov:
            expcov_stat = dexpdnobs @ tf.transpose(dexpdnobs)
            expcov = dexpdtheta0 @ tf.transpose(dexpdtheta0) + expcov_stat
            if self.binByBinStat:
                expcov_binByBinStat = dexpdbeta0 @ tf.transpose(dexpdbeta0)
                expcov += expcov_binByBinStat

            expvar = tf.linalg.diag_part(expcov)
        else:
            expcov = None
            expvar_stat = tf.reduce_sum(tf.square(dexpdnobs), axis=-1)
            expvar = tf.reduce_sum(tf.square(dexpdtheta0), axis=-1) + expvar_stat
            if self.binByBinStat:
                expvar_binByBinStat = tf.reduce_sum(tf.square(dexpdbeta0), axis=-1)
                expvar += expvar_binByBinStat
        
        expvar = tf.reshape(expvar, expected.shape)

        if compute_global_impacts:
            dexpdtheta0_squared = tf.square(dexpdtheta0)
            impacts_grouped = tf.map_fn(
                lambda idxs: self._compute_global_impact_group(dexpdtheta0_squared, idxs), 
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32), 
                fn_output_signature=tf.TensorSpec(shape=(dexpdtheta0_squared.shape[0],), dtype=tf.float64)
            )

            # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
            impacts = dexpdtheta0[:,self.indata.nsystnoconstraint:]

            impacts_grouped = tf.transpose(impacts_grouped)

            if compute_cov:
                expvar_stat = tf.linalg.diag_part(expcov_stat)
            impacts_stat = tf.sqrt(expvar_stat)
            impacts_stat = tf.reshape(impacts_stat, (-1, 1))
            impacts_grouped = tf.concat([impacts_grouped, impacts_stat], axis=1)

            if self.binByBinStat:
                if compute_cov:
                    expvar_binByBinStat = tf.linalg.diag_part(expcov_binByBinStat)
                impacts_binByBinStat = tf.sqrt(expvar_binByBinStat)
                impacts_binByBinStat = tf.reshape(impacts_binByBinStat, (-1, 1))
                impacts_grouped = tf.concat([impacts_grouped, impacts_binByBinStat], axis=1)
        else:
            impacts = None
            impacts_grouped = None

        return expected, expvar, expcov, impacts, impacts_grouped

    def _expvar_optimized(self, fun_exp, cov, skipBinByBinStat = False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix

        #FIXME this doesn't actually work for the positive semi-definite case
        invhesschol = tf.linalg.cholesky(cov)

        # since the full covariance matrix with respect to the bin counts is given by J^T R^T R J, then summing RJ element-wise squared over the parameter axis gives the diagonal elements

        expected = fun_exp()

        # dummy vector for implicit transposition
        u = tf.ones_like(expected)
        with tf.GradientTape(watch_accessed_variables=False) as t1:
            t1.watch(u)
            with tf.GradientTape() as t2:
                expected = fun_exp()
            # this returns dndx_j = sum_i u_i dn_i/dx_j
            Ju = t2.gradient(expected, self.x, output_gradients=u)
            Ju = tf.transpose(Ju)
            Ju = tf.reshape(Ju, [-1, 1])
            RJu = tf.matmul(tf.stop_gradient(invhesschol), Ju, transpose_a=True)
            RJu = tf.reshape(RJu, [-1])
        RJ = t1.jacobian(RJu, u)
        sRJ2 = tf.reduce_sum(RJ**2, axis=0)
        sRJ2 = tf.reshape(sRJ2, expected.shape)
        if self.binByBinStat and not skipBinByBinStat:
            # add MC stat uncertainty on variance
            sumw2 = tf.square(expected)/self.indata.kstat
            sRJ2 = sRJ2 + sumw2
        return expected, sRJ2

    @tf.function
    def _chi2(self, res, rescov):
        resv = tf.reshape(res, (-1,1))

        chi_square_value = tf.transpose(resv) @ tf.linalg.solve(rescov, resv)

        return chi_square_value[0,0]

    @tf.function
    def _expvar(self, fun_exp, invhess, compute_cov=False, compute_global_impacts=False):
        # compute uncertainty on expectation propagating through uncertainty on fit parameters using full covariance matrix
        #FIXME switch back to optimized version at some point?

        with tf.GradientTape() as t:
            t.watch([self.theta0, self.nobs, self.beta0])
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        pdexpdx, pdexpdnobs, pdexpdbeta0 = t.jacobian(expected_flat, [self.x, self.nobs, self.beta0])

        expcov = pdexpdx @ tf.matmul(invhess, pdexpdx, transpose_b = True)

        if pdexpdnobs is not None:
            varnobs = self.nobs
            exp_cov_stat = pdexpdnobs @ (varnobs[:, None] * tf.transpose(pdexpdnobs))
            expcov += exp_cov_stat
        
        expcov_noBBB = expcov
        if self.binByBinStat:
            varbeta0 = tf.math.reciprocal(self.indata.kstat)
            exp_cov_BBB = pdexpdbeta0 @ (varbeta0[:, None] * tf.transpose(pdexpdbeta0))
            expcov += exp_cov_BBB

        if compute_global_impacts:
            #FIXME This is not correct
            print("WARNING: Global impacts on observables without profiling is under development and probably wrong!")

            dxdtheta0, dxdnobs, dxdbeta0 = self._global_impacts(invhess)
            dexpdtheta0 = pdexpdx @ dxdtheta0

            var_theta0 = tf.where(self.indata.constraintweights == 0., tf.zeros_like(self.indata.constraintweights), tf.math.reciprocal(self.indata.constraintweights))

            dtheta0 = tf.math.sqrt(var_theta0)
            dexpdtheta0 *= dtheta0[None, :]
            dexpdtheta0_squared = tf.square(dexpdtheta0)

            impacts_grouped = tf.map_fn(
                lambda idxs: self._compute_global_impact_group(dexpdtheta0_squared, idxs), 
                tf.ragged.constant(self.indata.systgroupidxs, dtype=tf.int32), 
                fn_output_signature=tf.TensorSpec(shape=(dexpdtheta0_squared.shape[0],), dtype=tf.float64)
            )

            # global impacts of unconstrained parameters are always 0, only store impacts of constrained ones
            impacts = dexpdtheta0[:,self.indata.nsystnoconstraint:]

            impacts_grouped = tf.transpose(impacts_grouped)

            # stat global impact from all unconstrained parameters, not sure if this is correct TODO: check
            impacts_stat = tf.sqrt(tf.linalg.diag_part(expcov_noBBB) - tf.reduce_sum(dexpdtheta0_squared, axis=-1))
            impacts_stat = tf.reshape(impacts_stat, (-1, 1))
            impacts_grouped = tf.concat([impacts_grouped, impacts_stat], axis=1)

            if self.binByBinStat:
                impacts_BBB_stat = tf.sqrt(tf.linalg.diag_part(exp_cov_BBB))
                impacts_BBB_stat = tf.reshape(impacts_BBB_stat, (-1, 1))
                impacts_grouped = tf.concat([impacts_grouped, impacts_BBB_stat], axis=1)
        else:
            impacts = None
            impacts_grouped = None

        expvar = tf.linalg.diag_part(expcov)
        expvar = tf.reshape(expvar, expected.shape)

        return expected, expvar, expcov, impacts, impacts_grouped

    @tf.function
    def _expvariations(self, fun_exp, cov, correlations):
        with tf.GradientTape() as t:
            expected = fun_exp()
            expected_flat = tf.reshape(expected, (-1,))
        dexpdx = t.jacobian(expected_flat, self.x)

        if correlations:
            # construct the matrix such that the columns represent
            # the variations associated with profiling a given parameter
            # taking into account its correlations with the other parameters
            dx = cov/tf.math.sqrt(tf.linalg.diag_part(cov))[None, :]

            dexp = dexpdx @ dx
        else:
            dexp = dexpdx*tf.math.sqrt(tf.linalg.diag_part(cov))[None, :]

        dexp = tf.reshape(dexp, (*expected.shape, -1))

        down = expected[..., None] - dexp
        up = expected[..., None] + dexp

        expvars = tf.stack([down, up], axis=-1)

        return expvars

    def _compute_yields_noBBB(self, compute_normfull=False):
        xpoi = self.x[:self.npoi]
        theta = self.x[self.npoi:]

        if self.allowNegativePOI:
            poi = xpoi
            gradr = tf.ones_like(poi)
        else:
            poi = tf.square(xpoi)
            gradr = 2.*xpoi

        rnorm = tf.concat([poi, tf.ones([self.indata.nproc - poi.shape[0]], dtype=self.indata.dtype)], axis=0)
        mrnorm = tf.expand_dims(rnorm,-1)
        ernorm = tf.reshape(rnorm,[1,-1])

        #interpolation for asymmetric log-normal
        twox = 2.*theta
        twox2 = twox*twox
        alpha =  0.125 * twox * (twox2 * (3.*twox2 - 10.) + 15.)
        alpha = tf.clip_by_value(alpha,-1.,1.)

        thetaalpha = theta*alpha

        mthetaalpha = tf.stack([theta,thetaalpha],axis=0) #now has shape [2,nsyst]
        mthetaalpha = tf.reshape(mthetaalpha,[2*self.indata.nsyst,1])

        if self.indata.sparse:
            logsnorm = tf.sparse.sparse_dense_matmul(self.indata.logk_sparse, mthetaalpha)
            logsnorm = tf.squeeze(logsnorm,-1)
            snorm = tf.exp(logsnorm)
            
            snormnorm_sparse = self.indata.norm_sparse.with_values(snorm * self.indata.norm_sparse.values)
            nexpfullcentral = tf.sparse.sparse_dense_matmul(snormnorm_sparse, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral,-1)

            if compute_normfull:
                snormnorm = tf.sparse.to_dense(snormnorm_sparse)

        else:
            mlogk = tf.reshape(self.indata.logk,[self.indata.nbins*self.indata.nproc,2*self.indata.nsyst])
            logsnorm = tf.matmul(mlogk,mthetaalpha)
            logsnorm = tf.reshape(logsnorm,[self.indata.nbins,self.indata.nproc])

            snorm = tf.exp(logsnorm)

            snormnorm = snorm*self.indata.norm
            nexpfullcentral = tf.matmul(snormnorm, mrnorm)
            nexpfullcentral = tf.squeeze(nexpfullcentral,-1)

        # if options.saveHists:
        if compute_normfull:
            normfullcentral = ernorm*snormnorm
        else: 
            normfullcentral = None

        if self.normalize:
            # FIXME this should be done per-channel ideally
            normscale = tf.reduce_sum(self.nobs)/tf.reduce_sum(nexpfullcentral)

            nexpfullcentral *= normscale
            if compute_normfull:
                normfullcentral *= normscale

        return nexpfullcentral, normfullcentral

    def _compute_yields_with_beta(self, profile=True, profile_grad=True, compute_normfull=False):
        nexpfullcentral, normfullcentral = self._compute_yields_noBBB(compute_normfull)

        nexpfull = nexpfullcentral
        normfull = normfullcentral

        beta = None
        if self.binByBinStat:
            if profile:
                beta = (self.nobs + self.indata.kstat)/(nexpfullcentral+self.indata.kstat)
                if not profile_grad:
                    beta = tf.stop_gradient(beta)
            else:
                beta = self.beta0
            nexpfull = beta*nexpfullcentral
            if compute_normfull:
                normfull = beta[..., None]*normfullcentral

            if self.normalize:
                # FIXME this is probably not fully consistent when combined with the binByBinStat
                normscale = tf.reduce_sum(self.nobs)/tf.reduce_sum(nexpfull)

                nexpfull *= normscale
                if compute_normfull:
                    normfull *= normscale

        return nexpfull, normfull, beta

    def _compute_yields(self, inclusive=True, profile=True, profile_grad=True):
        nexpfullcentral, normfullcentral, beta = self._compute_yields_with_beta(profile=profile, profile_grad=profile_grad, compute_normfull=not inclusive)
        if inclusive:
            return nexpfullcentral
        else:
            return normfullcentral

    @tf.function
    def expected_with_variance(self, fun, cov, compute_cov=False, compute_global_impacts=False, profile=True):
        if profile:
            return self._expvar_profiled(fun, cov, compute_cov, compute_global_impacts)
        else:
            return self._expvar(fun, cov, compute_cov, compute_global_impacts)

    @tf.function
    def expected_variations(self, fun, cov, correlations=False):
        return self._expvariations(fun, cov, correlations=correlations)

    def expected_hists(self, cov=None, inclusive=True, compute_variance=True, compute_cov=False, compute_global_impacts=False, compute_variations=False, correlated_variations=False, profile=True, profile_grad=True, compute_chi2=False, aux_info=False, name=None, label=None):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile=profile, profile_grad=profile_grad)

        if compute_variations and (compute_variance or compute_cov or compute_global_impacts):
            raise NotImplementedError()

        if compute_cov or compute_variance or compute_global_impacts:
            exp, expvar, expcov, exp_impacts, exp_impacts_grouped = self.expected_with_variance(fun, cov, compute_cov=compute_variance, compute_global_impacts=compute_global_impacts, profile=profile)
        elif compute_variations:
            exp = self.expected_variations(fun, cov, correlations=correlated_variations)
        else:
            exp = tf.function(fun)()

        hists = {}
        aux_dict = {}

        var_axes = []
        if compute_variations:
            axis_vars = hist.axis.StrCategory(self.parms, name="vars")
            axis_downUpVar = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")

            var_axes = [axis_vars, axis_downUpVar]

        for channel, info in self.indata.channel_info.items():
            axes = info["axes"]

            start = info["start"]
            stop = info["stop"]

            hist_axes = axes.copy()

            if not inclusive:
                hist_axes.append(self.indata.axis_procs)

            hists[channel] = self.hist(
                f"{name}_{channel}",
                [*hist_axes, *var_axes],
                values=exp[start:stop], 
                variances=expvar[start:stop] if compute_variance else None,
                label=label
                )
            
            if compute_global_impacts:
                if "hist_global_impacts" not in aux_dict.keys():
                    aux_dict["hist_global_impacts"] = {}
                    aux_dict["hist_global_impacts_grouped"] = {}

                axis_impacts = self.indata.getGlobalImpactsAxes()
                axis_impacts_grouped = self.indata.getImpactsAxesGrouped(self.binByBinStat)

                h_impacts = self.hist(
                    f"{name}_{channel}", 
                    [*hist_axes, axis_impacts], 
                    values=exp_impacts[start:stop], 
                    label=label,
                    )
                h_impacts_grouped = self.hist(
                    f"{name}_{channel}", 
                    [*hist_axes, axis_impacts_grouped], 
                    values=exp_impacts_grouped[start:stop], 
                    label=label,
                    )

                aux_dict["hist_global_impacts"][channel] = h_impacts
                aux_dict["hist_global_impacts_grouped"][channel] = h_impacts_grouped

        if compute_cov:
            # flat axes for covariance matrix, since it can go across channels
            flat_axis_x = hist.axis.Integer(0, expcov.shape[0], underflow=False, overflow=False, name="x")
            flat_axis_y = hist.axis.Integer(0, expcov.shape[1], underflow=False, overflow=False, name="y")

            h_expcov = self.hist(f"{name}_cov", [flat_axis_x, flat_axis_y], values=expcov, label=f"{label} covariance")
            aux_dict["hist_cov"] = h_expcov

        if compute_chi2:
            def fun_residual():
                return fun() - self.nobs

            if profile:
                res, resvar, rescov, _1, _2 = self._expvar_profiled(fun_residual, cov, compute_cov=True)
            else:
                res, resvar, rescov, _1, _2 = self._expvar(fun_residual, cov, compute_cov=True)

            chi2val = self.chi2(res, rescov).numpy()
            ndf = tf.size(exp).numpy() - self.normalize

            aux_dict["ndf"] = ndf
            aux_dict["chi2"] = chi2val

        if aux_info:
            return hists, aux_dict
        else:
            return hists

    def expected_projection_hist(self, channel, axes, cov=None, inclusive=True, compute_variance=True, compute_cov=False, compute_global_impacts=False, compute_variations=False, correlated_variations=False, profile=True, profile_grad=True, compute_chi2=False, aux_info=False, name=None, label=None):

        def fun():
            return self._compute_yields(inclusive=inclusive, profile=profile, profile_grad=profile_grad)

        info = self.indata.channel_info[channel]
        start = info["start"]
        stop = info["stop"]

        channel_axes = info["axes"]

        exp_axes = channel_axes.copy()
        hist_axes = [axis for axis in channel_axes if axis.name in axes]

        if len(hist_axes) != len(axes):
            raise ValueError("axis not found")

        extra_axes = []
        if not inclusive:
            exp_axes.append(self.indata.axis_procs)
            hist_axes.append(self.indata.axis_procs)
            extra_axes.append(self.indata.axis_procs)

        var_axes = []
        if compute_variations:
            axis_vars = hist.axis.StrCategory(self.parms, name="vars")
            axis_downUpVar = hist.axis.Regular(2, -2., 2., underflow=False, overflow=False, name = "downUpVar")
            var_axes = [axis_vars, axis_downUpVar]

        exp_shape = tuple([len(a) for a in exp_axes])

        channel_axes_names = [axis.name for axis in channel_axes]
        exp_axes_names = [axis.name for axis in exp_axes]
        extra_axes_names = [axis.name for axis in extra_axes]

        axis_idxs = [channel_axes_names.index(axis) for axis in axes]

        proj_idxs = [i for i in range(len(channel_axes)) if i not in axis_idxs]

        post_proj_axes_names = [axis for axis in channel_axes_names if axis in axes] + extra_axes_names

        transpose_idxs = [post_proj_axes_names.index(axis) for axis in axes] +  [post_proj_axes_names.index(axis) for axis in extra_axes_names]

        def make_projection_fun(fun_flat):
            def proj_fun():
                exp = fun_flat()[start:stop]
                exp = tf.reshape(exp, exp_shape)
                exp = tf.reduce_sum(exp, axis=proj_idxs)
                exp = tf.transpose(exp, perm=transpose_idxs)

                return exp

            return proj_fun

        projection_fun = make_projection_fun(fun)

        if compute_variations and (compute_variance or compute_cov or compute_global_impacts):
            raise NotImplementedError()

        if (compute_variance or compute_cov or compute_global_impacts):
            exp, expvar, expcov, exp_impacts, exp_impacts_grouped = self.expected_with_variance(projection_fun, cov, compute_cov=compute_cov, compute_global_impacts=compute_global_impacts, profile=profile)
        elif compute_variations:
            exp = self.expected_variations(projection_fun, cov, correlations=correlated_variations)
        else:
            exp = tf.function(projection_fun)()

        h = self.hist(
            name, 
            [*hist_axes, *var_axes], 
            values=exp, 
            variances=expvar if compute_variance else None, 
            label=label,
            )

        aux_dict = {}

        if compute_global_impacts:

            axis_impacts = self.indata.getGlobalImpactsAxes()
            axis_impacts_grouped = self.indata.getImpactsAxesGrouped(self.binByBinStat)

            h_impacts = self.hist(name, [*hist_axes, axis_impacts], values=exp_impacts[start:stop], label=label)
            h_impacts_grouped = self.hist(name, [*hist_axes, axis_impacts_grouped], values=exp_impacts_grouped[start:stop], label=label)

            aux_dict["hist_global_impacts"] = h_impacts
            aux_dict["hist_global_impacts_grouped"] = h_impacts_grouped

        if compute_cov:
            # flat axes for covariance matrix, since it can go across channels
            flat_axis_x = hist.axis.Integer(0, expcov.shape[0], underflow=False, overflow=False, name="x")
            flat_axis_y = hist.axis.Integer(0, expcov.shape[1], underflow=False, overflow=False, name="y")

            h_expcov = self.hist(f"{name}_cov", [flat_axis_x, flat_axis_y], values=expcov, label=f"{label} covariance")

            aux_dict["hist_cov"] = h_expcov

        if compute_chi2:
            def fun_residual():
                return fun() - self.nobs

            projection_fun_residual = make_projection_fun(fun_residual)

            if profile:
                res, resvar, rescov, _1, _2 = self._expvar_profiled(projection_fun_residual, cov, compute_cov=True)
            else:
                res, resvar, rescov, _1, _2 = self._expvar(projection_fun_residual, cov, compute_cov=True)

            chi2val = self.chi2(res, rescov).numpy()
            ndf = tf.size(exp).numpy() - self.normalize

            aux_dict["ndf"] = ndf
            aux_dict["chi2"] = chi2val

        if aux_info:
            return h, aux_dict
        else:
            return h

    def observed_hists(self):
        hists_data_obs = {}
        hists_nobs = {}

        for channel, info in self.indata.channel_info.items():
            axes = info["axes"]

            start = info["start"]
            stop = info["stop"]

            hists_data_obs[channel] = self.hist("data_obs", axes, values=self.indata.data_obs[start:stop], label="observed number of events in data")
            hists_nobs[channel] = self.hist("nobs", axes, values=self.nobs.value()[start:stop], label = "observed number of events for fit")

        return hists_data_obs, hists_nobs

    @tf.function
    def expected_events(self, profile=True):
        return self._compute_yields(inclusive=True, profile=profile)

    @tf.function
    def chi2(self, res, rescov):
        return self._chi2(res, rescov)

    @tf.function
    def saturated_nll(self):
        nobs = self.nobs

        nobsnull = tf.equal(nobs,tf.zeros_like(nobs))

        #saturated model
        nobssafe = tf.where(nobsnull, tf.ones_like(nobs), nobs)
        lognobs = tf.math.log(nobssafe)

        lsaturated = tf.reduce_sum(-nobs*lognobs + nobs, axis=-1)

        ndof = tf.size(nobs) - self.npoi - self.indata.nsystnoconstraint - self.normalize

        return lsaturated, ndof

    @tf.function
    def full_nll(self):
        l, lfull = self._compute_nll()
        return lfull

    def _compute_nll(self, profile=True, profile_grad=True):
        theta = self.x[self.npoi:]

        nexpfullcentral, _, beta = self._compute_yields_with_beta(profile=profile, profile_grad=profile_grad, compute_normfull=False)

        nexp = nexpfullcentral

        if self.chisqFit:
            residual = tf.reshape(self.nobs - nexp,[-1,1]) #chi2 residual
            # Solve the system without inverting
            ln = lnfull = 0.5 * tf.reduce_sum(tf.matmul(residual, tf.matmul(self.data_cov_inv, residual), transpose_a=True))
        else:
            nobsnull = tf.equal(self.nobs,tf.zeros_like(self.nobs))

        nexpsafe = tf.where(nobsnull, tf.ones_like(self.nobs), nexp)
        lognexp = tf.math.log(nexpsafe)

        nexpnomsafe = tf.where(nobsnull, tf.ones_like(self.nobs), self.nexpnom)
        lognexpnom = tf.math.log(nexpnomsafe)

        #final likelihood computation

        #poisson term
        lnfull = tf.reduce_sum(-self.nobs*lognexp + nexp, axis=-1)

        #poisson term with offset to improve numerical precision
        ln = tf.reduce_sum(-self.nobs*(lognexp-lognexpnom) + nexp-self.nexpnom, axis=-1)

        #constraints
        lc = tf.reduce_sum(self.indata.constraintweights*0.5*tf.square(theta - self.theta0))

        l = ln + lc
        lfull = lnfull + lc

        if self.binByBinStat:
            lbetavfull = -self.indata.kstat*tf.math.log(beta/self.beta0) + self.indata.kstat*beta/self.beta0

            lbetav = lbetavfull - self.indata.kstat
            lbeta = tf.reduce_sum(lbetav)

            l = l + lbeta
            lfull = lfull + lbeta

        return l, lfull

    def _compute_loss(self, profile=True, profile_grad=True):
        l, lfull = self._compute_nll(profile=profile, profile_grad=profile_grad)
        return l

    @tf.function
    def loss_val(self):
        val = self._compute_loss()
        return val

    @tf.function
    def loss_val_grad(self):
        with tf.GradientTape() as t:
            val = self._compute_loss()
        grad = t.gradient(val, self.x)

        return val, grad

    @tf.function
    def loss_val_grad_hessp(self, p):
        with tf.autodiff.ForwardAccumulator(self.x, p) as acc:
            with tf.GradientTape() as grad_tape:
                val = self._compute_loss()
            grad = grad_tape.gradient(val, self.x)
        hessp = acc.jvp(grad)

        return val, grad, hessp

    @tf.function
    def loss_val_grad_hess(self, profile=True, profile_grad=True):
        with tf.GradientTape() as t2:
            with tf.GradientTape() as t1:
                val = self._compute_loss(profile=profile, profile_grad=profile_grad)
            grad = t1.gradient(val, self.x)
        hess = t2.jacobian(grad, self.x)

        return val, grad, hess

    def minimize(self):

        def scipy_loss(xval):
            self.x.assign(xval)
            val, grad = self.loss_val_grad()
            print("scipy_loss", val)
            return val.numpy(), grad.numpy()

        def scipy_hessp(xval, pval):
            self.x.assign(xval)
            p = tf.convert_to_tensor(pval)
            val, grad, hessp = self.loss_val_grad_hessp(p)
            print("scipy_hessp", val)
            return hessp.numpy()

        def scipy_hess(xval):
            self.x.assign(xval)
            val, grad, hess = self.loss_val_grad_hess()
            print("scipy_hess", val)
            return hess.numpy()


        xval = self.x.numpy()

        res = scipy.optimize.minimize(scipy_loss, xval, method = "trust-krylov", jac = True, hessp = scipy_hessp)

        xval = res["x"]

        self.x.assign(xval)

        print(res)

        return res


