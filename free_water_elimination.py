import argparse
import numpy as np
import nibabel as nib

import dipy.reconst.fwdti as fwdti
from dipy.core.gradients import gradient_table


def free_water_elimination(dwi_fname, bval_fname, bvec_fname, 
                           mask_fname, model_name, output_fname):

  # load diffusion, mask image and bvec/bval gradient table
  dwi  = nib.load(dwi_fname)
  mask = nib.load(mask_fname).get_fdata() 
  gtab = gradient_table(bval_fname, bvec_fname)

  # perform free water elimination
  match model_name.lower():
    case "dipy_fwdti":
      fwe_image = dipy_fwdti(dwi, gtab, mask)
    case _:
      print("Unrecognized free water elimination model")

  # save free water eliminated image
  nib.save(fwe_image, output_fname)
  print(f"Saved: {output_fname}")
  

def dipy_fwdti(dwi, gtab, mask = None, Diso = 3.0e-3):
  # fit free-water diffusion tensor imaging model
  model = fwdti.FreeWaterTensorModel(gtab)
  model_fit = model.fit(dwi.get_fdata(), mask = mask)  

  # extract free-water dti model parameters
  model_params = model_fit.model_params # all model parameters
  fwf = model_params[..., -1] # extract free-water fraction

  # extract b0 signal from dwi image
  S0 = dwi.get_fdata()[..., gtab.bvals == 0].mean(axis = -1)

  # compute free-water signal
  fw_decay  = np.exp(-gtab.bvals * Diso) # free-water exponential decay
  fw_signal = (S0 * fwf).reshape(-1, 1) * fw_decay
  fw_signal = fw_signal.reshape(fwf.shape + (dwi.shape[-1], ))

  # compute free-water eliminated signal
  fwe_signal = dwi.get_fdata() - fw_signal

  # return free-water eliminated signal 
  return nib.Nifti1Image(fwe_signal, affine = dwi.affine)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("dwi_fname", type = str)
  parser.add_argument("bval_fname", type = str)
  parser.add_argument("bvec_fname", type = str)
  parser.add_argument("mask_fname", type = str)
  parser.add_argument("model_name", type = str)
  parser.add_argument("output_dir", type = str)
  args = parser.parse_args()
  
  free_water_elimination(
    dwi_fname  = args.dwi_fname,
    bval_fname = args.bval_fname, 
    bvec_fname = args.bvec_fname, 
    mask_fname = args.mask_fname,
    model_name = args.model_name, 
    output_dir = args.output_dir
  )