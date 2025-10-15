# Free Water Elimination (FWE) for Diffusion MRI

## Overview

Implements free water elimination (FWE) models for preprocessing diffusion MRI data.

---

## To install

```bash

$ git clone https://github.com/nrdg/fwe
$ cd fwe
$ pip install .
```

The software depends on [DIPY](https://dipy.org), which is installed
automatically as part of the installation process specified above.

---

## Available Models

* Free water DTI as implmented in `dipy` ([Hoy et al., 2014](https://doi.org/10.1016/j.neuroimage.2014.09.053)): Use `fwe_model="dipy_fwdti"`
* Beltrami regularized gradient descent free water DTI ([Golub et al., 2020](https://doi.org/10.1002/mrm.28599)): Use `fwe_model="golub_beltrami"`

---

## Inputs

The software expects as input data that has already been preprocessed (we use
[qsiprep](https://qsiprep.readthedocs.io/en/latest/)).

If the data has multiple non-zero b-values, it is preferable to use the
`'dipy_fwdti'` model. If the data has one non-zero b-value, only the
`'golub_beltrami'` model can be used.

---

## Example

The following is a complete example, using a subject from the [HBN POD2 dataset](https://www.nature.com/articles/s41597-022-01695-7).

To run this example, you will also need the `boto3` software library (`pip install boto3`), which will download the data.

```python
from dipy.data.fetcher import fetch_hbn, dipy_home
import os.path as op
from fwe import free_water_elimination

fetch_hbn(["NDARAA948VFH"])

dwi_folder = op.join(dipy_home, "HBN/derivatives/qsiprep/sub-NDARAA948VFH/ses-HBNsiteRU/dwi/")
data_root = op.join(dwi_folder, "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-preproc_dwi")

free_water_elimination(
  dwi_fname    = data_root + ".nii.gz",
  bval_fname   = data_root + ".bval",
  bvec_fname   = data_root + ".bvec",
  mask_fname   = op.join(dwi_folder, "sub-NDARAA948VFH_ses-HBNsiteRU_acq-64dir_space-T1w_desc-brain_mask.nii.gz"),
  fwe_model    = "dipy_fwdti",
  output_fname = "sub-01_fwe.nii.gz"
)
```
