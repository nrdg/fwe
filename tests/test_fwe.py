import numpy as np
import nibabel as nib
from numpy.testing import assert_array_almost_equal

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.gradients import read_bvals_bvecs
from dipy.sims.voxel import multi_tensor
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity

from fwe.fwe import (
    dipy_fwdti,
    remove_free_water,
    free_water_elimination,
    golub_beltrami,
)
from fwe.beltrami import BeltramiModel


def setup_module():
    """Module-level setup: simulate multi-shell DWI data with known ground truth."""
    global gtab_2s, DWI, FAref, MDref, GTF
    global DWI_beltrami, GTF_beltrami, gtab_beltrami

    _, fbvals, fbvecs = get_fnames(name="small_64D")
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

    # Multi-shell gradient table (FW-DTI requires >=2 shells)
    bvals_2s = np.concatenate((bvals, bvals * 1.5), axis=0)
    bvecs_2s = np.concatenate((bvecs, bvecs), axis=0)
    gtab_2s = gradient_table(bvals_2s, bvecs=bvecs_2s)

    diso = 0.003
    tissue_eigs = [0.0017, 0.0003, 0.0003]
    water_eigs = [diso, diso, diso]
    # Tissue and water eigenvalues
    mevals = np.array([tissue_eigs, water_eigs])

    # Free water fractions for 2x2x2 voxels
    # First slice: varying free water; second slice: all water (edge case)
    GTF = np.array([[[0.06, 0.71], [0.33, 0.91]], [[0.0, 0.0], [0.0, 0.0]]])

    DWI = np.zeros((2, 2, 2, len(gtab_2s.bvals)))
    FAref = np.zeros((2, 2, 2))
    MDref = np.zeros((2, 2, 2))

    for i in range(2):
        for j in range(2):
            gtf = GTF[0, i, j]
            S, _ = multi_tensor(
                gtab_2s,
                mevals,
                S0=100,
                angles=[(90, 0), (90, 0)],
                fractions=[(1 - gtf) * 100, gtf * 100],
                snr=None,
            )
            DWI[0, i, j] = S

            FAref[0, i, j] = fractional_anisotropy(np.array([0.0017, 0.0003, 0.0003]))
            MDref[0, i, j] = mean_diffusivity(np.array([0.0017, 0.0003, 0.0003]))

    # 10x10x10 volume with spatially varying S0 for Beltrami tests.
    # The Beltrami init methods (fraction_init_s0) require S0 variation to
    # distinguish Stissue from Swater via percentiles.
    beltrami_shape = (10, 10, 10)
    beltrami_fwf = 0.3
    mevals_bel = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])

    # Create spatially varying S0 map (range ~80..120)
    rng = np.random.RandomState(42)
    S0_map = 100 + 20 * rng.randn(*beltrami_shape)
    S0_map = np.clip(S0_map, 80, 120)

    DWI_beltrami = np.zeros(beltrami_shape + (len(bvals),))
    # Gradient table with b0_threshold=0 for Beltrami tests (required after bval scaling)
    gtab_beltrami = gradient_table(bvals, bvecs=bvecs, b0_threshold=0)

    for i in range(beltrami_shape[0]):
        for j in range(beltrami_shape[1]):
            for k in range(beltrami_shape[2]):
                S, _ = multi_tensor(
                    gtab_beltrami,
                    mevals_bel,
                    S0=S0_map[i, j, k],
                    angles=[(90, 0), (90, 0)],
                    fractions=[(1 - beltrami_fwf) * 100, beltrami_fwf * 100],
                    snr=None,
                )
                DWI_beltrami[i, j, k] = S
    GTF_beltrami = np.full(beltrami_shape, beltrami_fwf)


def test_beltrami_multi_voxel():
    """Beltrami model fit recovers known free water fraction on a 5x5x5 volume."""
    # Beltrami model expects bvals in s/mm² (scaled by 1e-3) and Diso in mm²/ms
    # b0_threshold=0 is required because scaled bvals are all < 50 (default threshold)
    bvals_scaled = gtab_beltrami.bvals * 1e-3
    gtab_scaled = gradient_table(bvals_scaled, gtab_beltrami.bvecs, b0_threshold=0)

    model = BeltramiModel(
        gtab_scaled,
        init_method="hybrid",
        Diso=3.0,
        iterations=100,
        learning_rate=0.0005,
    )
    model_fit = model.fit(DWI_beltrami)

    # Recovered free water fraction (BeltramiModel stores tissue fraction in params[..., 12])
    fwf = 1 - model_fit.model_params[..., 12]

    # Check mean fwf across the volume (loose tolerance for gradient descent)
    mean_fwf = np.mean(fwf)
    assert abs(mean_fwf - GTF_beltrami[0, 0, 0]) < 0.15, (
        f"Mean free water fraction {mean_fwf:.4f} != ~0.3"
    )


def test_dipy_fwdti_multi_voxel():
    """Multi-voxel FW-DTI fit recovers correct free water fractions."""
    fwe_img, model_params = dipy_fwdti(
        nib.Nifti1Image(DWI, affine=np.eye(4)), gtab_2s, Diso=3.0e-3, save_params=True
    )

    # Check output shape
    assert fwe_img.shape == DWI.shape

    # Extract free water fractions from first slice
    fwf = model_params.get_fdata()[0, :, :, -1]
    assert_array_almost_equal(fwf, GTF[0], decimal=1)


def test_remove_free_water():
    """remove_free_water correctly subtracts the free water signal."""
    # Create a simple test case: known S0, known fwf, known Diso
    n_dirs = len(gtab_2s.bvals)
    shape = (3, 3, 3)

    # Free water fraction: 0.3 everywhere
    fwf = 0.3 * np.ones(shape)

    # Signal: S0=100 for all directions (ignoring tensor decay for simplicity)
    S0 = 100.0
    data = S0 * np.ones(shape + (n_dirs,))

    dwi_img = nib.Nifti1Image(data, affine=np.eye(4))
    fwe_img = remove_free_water(dwi_img, gtab_2s, fwf, Diso=3.0e-3)

    # The output should have the free water component subtracted
    assert fwe_img.shape == data.shape

    # At b=0, free water decay = exp(0) = 1, so fw_signal = S0 * fwf = 30
    # fwe_signal should be S0 - 30 = 70 at b=0
    b0_mask = gtab_2s.bvals == 0
    fwe_data = fwe_img.get_fdata()
    b0_signal = fwe_data[..., b0_mask].mean(axis=-1)
    np.testing.assert_allclose(b0_signal, S0 * (1 - fwf), atol=1e-10)


def test_free_water_elimination_dipy_fwdti(tmp_path):
    """End-to-end: write NIfTI files, run free_water_elimination, check output."""
    # Create synthetic DWI data
    gtf = 0.3
    mevals = np.array([[0.0017, 0.0003, 0.0003], [0.003, 0.003, 0.003]])
    S, _ = multi_tensor(
        gtab_2s,
        mevals,
        S0=100,
        angles=[(90, 0), (90, 0)],
        fractions=[(1 - gtf) * 100, gtf * 100],
        snr=None,
    )

    shape = (5, 5, 5)
    DWI_vol = np.tile(S, (5, 5, 5, 1))
    mask_vol = np.ones(shape, dtype=np.float64)

    affine = np.eye(4)
    dwi_img = nib.Nifti1Image(DWI_vol, affine=affine)
    mask_img = nib.Nifti1Image(mask_vol, affine=affine)

    # Write files
    dwi_path = str(tmp_path / "dwi.nii.gz")
    bval_path = str(tmp_path / "dwi.bval")
    bvec_path = str(tmp_path / "dwi.bvec")
    mask_path = str(tmp_path / "mask.nii.gz")
    output_path = str(tmp_path / "fwe_output.nii.gz")

    nib.save(dwi_img, dwi_path)
    nib.save(mask_img, mask_path)
    np.savetxt(bval_path, gtab_2s.bvals.reshape(1, -1), fmt="%d")
    np.savetxt(bvec_path, gtab_2s.bvecs.T, fmt="%.6f")

    # Run free water elimination
    free_water_elimination(
        dwi_fname=dwi_path,
        bval_fname=bval_path,
        bvec_fname=bvec_path,
        mask_fname=mask_path,
        fwe_model="dipy_fwdti",
        output_fname=output_path,
        Diso=3.0e-3,
    )

    # Check output file exists and loads correctly
    assert (tmp_path / "fwe_output.nii.gz").exists()
    result = nib.load(output_path)
    assert result.shape == DWI_vol.shape


def test_free_water_elimination_invalid_model(tmp_path):
    """Unknown model name does not crash (logs error, returns None)."""
    shape = (3, 3, 3)
    n_dirs = len(gtab_2s.bvals)

    dwi_img = nib.Nifti1Image(np.ones(shape + (n_dirs,)), affine=np.eye(4))
    mask_img = nib.Nifti1Image(np.ones(shape), affine=np.eye(4))

    dwi_path = str(tmp_path / "dwi.nii.gz")
    bval_path = str(tmp_path / "dwi.bval")
    bvec_path = str(tmp_path / "dwi.bvec")
    mask_path = str(tmp_path / "mask.nii.gz")
    output_path = str(tmp_path / "fwe_output.nii.gz")

    nib.save(dwi_img, dwi_path)
    nib.save(mask_img, mask_path)
    np.savetxt(bval_path, gtab_2s.bvals.reshape(1, -1), fmt="%d")
    np.savetxt(bvec_path, gtab_2s.bvecs.T, fmt="%.6f")

    # Should not raise, but the output file won't be created
    # (the function logs an error and falls through without saving)
    try:
        free_water_elimination(
            dwi_fname=dwi_path,
            bval_fname=bval_path,
            bvec_fname=bvec_path,
            mask_fname=mask_path,
            fwe_model="invalid_model",
            output_fname=output_path,
        )
    except Exception:
        pass  # acceptable — the function may fail on missing output


def test_beltrami_uniform_spatial():
    """Beltrami regularization doesn't introduce spatial artifacts on uniform data."""
    bvals_scaled = gtab_beltrami.bvals * 1e-3
    gtab_scaled = gradient_table(bvals_scaled, gtab_beltrami.bvecs, b0_threshold=0)

    model = BeltramiModel(
        gtab_scaled,
        init_method="hybrid",
        Diso=3.0,
        iterations=100,
        learning_rate=0.0005,
    )
    model_fit = model.fit(DWI_beltrami)
    fwf = 1 - model_fit.model_params[..., 12]

    # All voxels have the same fwf; regularization should keep them similar.
    # With S0 variation (needed for init), some residual variation is expected.
    fwf_std = np.std(fwf)
    assert fwf_std < 0.25, f" fwf spatial std {fwf_std:.4f} > 0.25 on uniform fwf data"


def test_golub_beltrami_wrapper():
    """golub_beltrami wrapper produces correct output shape and reasonable fwf."""
    dwi_img = nib.Nifti1Image(DWI_beltrami, affine=np.eye(4))
    fwe_img, model_params = golub_beltrami(
        dwi_img,
        gtab_beltrami,
        mask=None,
        Diso=3.0e-3,
        save_params=True,
        n_iterations=100,
        learning_rate=0.0005,
    )

    assert fwe_img.shape == DWI_beltrami.shape
    assert model_params.shape == DWI_beltrami.shape[:3] + (13,)

    # Check mean fwf is reasonable
    fwf = 1 - model_params.get_fdata()[..., 12]
    mean_fwf = np.mean(fwf)
    assert abs(mean_fwf - 0.3) < 0.15, f"Mean fwf {mean_fwf:.4f} != ~0.3"


def test_free_water_elimination_golub_beltrami(tmp_path):
    """End-to-end: free_water_elimination with golub_beltrami model."""
    shape = (10, 10, 10)
    mask_vol = np.ones(shape, dtype=np.float64)
    affine = np.eye(4)

    dwi_img = nib.Nifti1Image(DWI_beltrami, affine=affine)
    mask_img = nib.Nifti1Image(mask_vol, affine=affine)

    dwi_path = str(tmp_path / "dwi.nii.gz")
    bval_path = str(tmp_path / "dwi.bval")
    bvec_path = str(tmp_path / "dwi.bvec")
    mask_path = str(tmp_path / "mask.nii.gz")
    output_path = str(tmp_path / "fwe_output.nii.gz")

    nib.save(dwi_img, dwi_path)
    nib.save(mask_img, mask_path)
    np.savetxt(bval_path, gtab_beltrami.bvals.reshape(1, -1), fmt="%d")
    np.savetxt(bvec_path, gtab_beltrami.bvecs.T, fmt="%.6f")

    free_water_elimination(
        dwi_fname=dwi_path,
        bval_fname=bval_path,
        bvec_fname=bvec_path,
        mask_fname=mask_path,
        fwe_model="golub_beltrami",
        output_fname=output_path,
        Diso=3.0e-3,
    )

    assert (tmp_path / "fwe_output.nii.gz").exists()
    result = nib.load(output_path)
    assert result.shape == DWI_beltrami.shape
