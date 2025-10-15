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

```python
from fwe import free_water_elimination

free_water_elimination(
  dwi_fname    = "sub-01_dwi.nii.gz",
  bval_fname   = "sub-01.bval",
  bvec_fname   = "sub-01.bvec",
  mask_fname   = "sub-01_mask.nii.gz",
  fwe_model    = "dipy_fwdti",
  output_fname = "sub-01_fwe.nii.gz"
)
```
