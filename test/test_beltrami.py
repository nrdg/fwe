import numpy as np
from dipy.core.gradients import gradient_table
from dipy.data.fetcher import get_fnames, read_bvals_bvecs
from dipy.reconst.dti import TensorModel, decompose_tensor, from_lower_triangular
from dipy.sims.voxel import all_tensor_evecs, multi_tensor, single_tensor
from fwe.beltrami import BeltramiModel
from numpy.testing import assert_almost_equal

_, fbvals, fbvecs = get_fnames(name="small_64D")
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab = gradient_table(bvals, bvecs=bvecs)


def setup_module():
    """Module-level setup"""
    global gtab, mevals, model_params_mv
    global DWI, FAref, GTF, MDref, FAdti, MDdti
    _, fbvals, fbvecs = get_fnames(name="small_64D")
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs=bvecs)

    # Simulation a typical DT and DW signal for no water contamination
    # S0 = np.array(100)
    dt = np.array([0.0017, 0, 0.0003, 0, 0, 0.0003])
    evals, evecs = decompose_tensor(from_lower_triangular(dt))
    S_tissue = single_tensor(gtab, S0=100, evals=evals, evecs=evecs, snr=None)
    dm = TensorModel(gtab, fit_method="WLS")
    dtifit = dm.fit(S_tissue)
    FAdti = dtifit.fa
    MDdti = dtifit.md

    # Simulation of 8 voxels tested
    DWI = np.zeros((2, 2, 2, len(gtab.bvals)))
    FAref = np.zeros((2, 2, 2))
    MDref = np.zeros((2, 2, 2))
    # Diffusion of tissue and water compartments are constant for all voxel
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    # volume fractions
    GTF = np.array([[[0.06, 0.71], [0.33, 0.91]], [[0.0, 0.0], [0.0, 0.0]]])
    # S0 multivoxel
    # S0m = 100 * np.ones((2, 2, 2))
    # model_params ground truth (to be fill)
    model_params_mv = np.zeros((2, 2, 2, 13))
    for i in range(2):
        for j in range(2):
            gtf = GTF[0, i, j]
            S, p = multi_tensor(
                gtab,
                mevals,
                S0=100,
                angles=[(90, 0), (90, 0)],
                fractions=[(1 - gtf) * 100, gtf * 100],
                snr=None,
            )
            DWI[0, i, j] = S
            FAref[0, i, j] = FAdti
            MDref[0, i, j] = MDdti
            R = all_tensor_evecs(p[0])
            R = R.reshape(9)
            model_params_mv[0, i, j] = np.concatenate(
                ([0.0017, 0.0003, 0.0003], R, [gtf]), axis=0
            )


def test_beltrami_model():
    global DWI, FAref, GTF, MDdti, FAdti
    fwdm = BeltramiModel(gtab)
    fwefit = fwdm.fit(DWI)
    FA = fwefit.fa
    FWF = fwefit.f
    MD = fwefit.md

    assert_almost_equal(FWF, GTF, decimal=3)
    assert_almost_equal(FA, FAref, decimal=3)
    assert_almost_equal(MD, MDref, decimal=3)
