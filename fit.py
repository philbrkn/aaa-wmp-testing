import numpy as np
from baryrat import BarycentricRational

# on writing version 2 of this code.
# second argument is a plain old list of different function
# values. just expand that out.
# on return, a multi-valued barycentric object should be
# returned. this can be based heavily off the baryrat package.

# or... just be lazy for now!! this can be made to work.

# defines a barycentric rational expression, based heavily on baryrat
def aaa_mp(egrid, scattering,
            absorption,
            fission=None,
            tol=1e-13,
            mmax=100,
            use_relative_error=False):
    """

    This is a very slightly modified version of the aaa routine
    from the baryrat package. It is oriented towards getting
    the poles and residues of conventional WMP cross-sections.

    Arguments:
        egrid (array): the sampling points of the cross-section
        scattering (array): elastic scattering cross-section values
        absorption (array): absorption cross-section values
        fission    (array): (optional) fission values

    Returns:
        TODO what am I returning?
    """
    assert all([len(g.shape) == 1 for g in [egrid, scattering, absorption]])
    if fission:
        assert len(fission.shape) == 1

    # This is the set of point indices we're NOT interpolating at
    J = list(range(len(egrid)))

    # We exactly interpolate on this point set
    zj = np.empty(0, dtype=egrid.dtype)

    # saved values of the components to interpolate
    fj_scat = np.empty(0, dtype=float)
    fj_abs = np.empty(0, dtype=float)
    if fission:
        fj_fiss = np.empty(0, dtype=F.dtype)

    C = []

    # calculate the weights of the gridpoints so that we're
    # prioritizing extremely finely spaced areas equally to
    # the widely spaced ones.
    W = np.zeros_like(egrid)
    W[:-1] += 0.5 * np.diff(egrid)
    W[1:]  += 0.5 * (egrid[1:] - egrid[:-1])

    errors = []

    if use_relative_error:
        mul_s = np.copy(scattering)
        mul_s[mul_s == 0.0] = 1.0
        mul_a = np.copy(absorption)
        mul_a[mul_a == 0.0] = 1.0

    Rscat = np.mean(scattering) * np.ones_like(scattering)
    Rabs = np.mean(absorption) * np.ones_like(absorption)
    if fission:
        Rfiss = np.mean(fission) * np.ones_like(fission)

    for m in range(mmax):
        # find largest residual
        scat_err = np.abs(scattering - Rscat)
        abs_err = np.abs(absorption - Rabs)
        if use_relative_error:

            scat_err /= mul_a
            abs_err /= mul_a
        if fission:
            fiss_err = np.abs(fission - Rfiss)
            if use_relative_error:
                fiss_err /= fission

        jj_scat = np.argmax(scat_err)
        jj_abs = np.argmax(abs_err)
        max_scat_err = scat_err[jj_scat]
        max_abs_err = abs_err[jj_abs]
        jj = -1 # voting on the index to interpolate at
        if max_scat_err > max_abs_err:
            jj = jj_scat
        else:
            jj = jj_abs
        if fission:
            jj_fiss = np.argmax(fiss_err)
            max_fiss_err = fiss_err[jj_fiss]
            if max_fiss_err > max_scat_err and max_fiss_err > max_abs_err:
                jj = jj_fiss
        assert jj != -1

        # at this point, jj is the index of max error among all xs components

        zj = np.append(zj, (egrid[jj],))
        fj_scat = np.append(fj_scat, (scattering[jj],))
        fj_abs = np.append(fj_abs, (absorption[jj],))
        if fission:
            fj_fiss = np.append(fj_fiss, (fission[jj],))

        if jj not in J:
            jj += 1
        J.remove(jj)


        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (egrid[J,None] - zj[None,:])

        # Loewner matrix for each cross-section component

        # NOTE might want to weight so as to minimize the RELATIVE error.
        A_scat = W[J, None] * (scattering[J,None] - fj_scat[None,:]) * C
        A_abs = W[J, None] * (absorption[J,None] - fj_abs[None,:]) * C
        if use_relative_error:
            A_scat /= mul_s[J, None]
            A_abs /= mul_a[J, None]

        if fission:
            A_fiss = W[J, None] * (fission[J,None] - fj_fiss[None,:]) * C

        # Do multi-component
        matlist = [A_scat, A_abs]
        if fission:
            matlist.append(A_fiss) # this is by reference

        # UH. is it vstack to use here, or hstack?
        big_A = np.vstack(matlist)

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(big_A)
        wj = Vh[-1, :].conj()

        # approximation: numerator / denominator
        Nscat = C.dot(wj * fj_scat)
        Nabs = C.dot(wj * fj_abs)
        if fission:
            Nfiss = C.dot(wj * fj_fiss)
        D = C.dot(wj)

        # update residual
        Rscat = scattering.copy()
        Rabs = absorption.copy()
        Rscat[J] = Nscat / D
        Rabs[J] = Nabs / D
        if fission:
            Rfiss = fission.copy()
            Rfiss[J] = Nfiss / D

        # check for convergence
        w1 = scattering if use_relative_error else 1.0
        w2 = absorption if use_relative_error else 1.0
        w3 = fission if use_relative_error and fission else 1.0
        errors.append((np.linalg.norm((scattering - Rscat)/w1, np.inf),
                       np.linalg.norm((absorption - Rabs)/w2, np.inf),
                       np.linalg.norm((fission - Rfiss)/w3, np.inf) if fission else 0.0
        ))
        if max(errors[-1]) <= tol:
            break

    rscat = BarycentricRational(zj, fj_scat, wj)
    rabs = BarycentricRational(zj, fj_abs, wj)
    rfiss = BarycentricRational(zj, fj_fiss, wj) if fission else None
    return ((rscat, rabs, rfiss), errors)
