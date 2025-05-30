# Free Water Elimination (FWE) for Diffusion MRI

## Overview 

Implements free water elimination (FWE) models for preprocessing diffusion MRI data. 

---

## Available Models

* Free water DTI as implmented in `dipy` ([Hoy et al., 2014](https://doi.org/10.1016/j.neuroimage.2014.09.053))
* Beltrami regularized gradient descent free water DTI ([Golub et al., 2020](https://doi.org/10.1002/mrm.28599))

---

## Example

```python
from fwe import free_water_elimination

free_water_elimination(
  dwi_fname    = "sub-01_dwi.nii.gz",
  bval_fname   = "sub-01.bval",
  bvec_fname   = "sub-01.bvec",
  mask_fname   = "sub-01_mask.nii.gz",
  fwe_model    = "dipy_fwdit",
  output_fname = "sub-01_fwe.nii.gz"
)
```
