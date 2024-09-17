import argparse
import numpy as np
import nibabel as nib
from beltrami import BeltramiModel
import dipy.reconst.fwdti as fwdti
from dipy.core.gradients import gradient_table


def free_water_elimination(dwi_fname, bval_fname, bvec_fname, 
                           mask_fname, fwe_model, output_fname):

  # load diffusion, mask image and bvec/bval gradient table
  dwi  = nib.load(dwi_fname)
  mask = nib.load(mask_fname).get_fdata() 
  gtab = gradient_table(bval_fname, bvec_fname)

  # perform free water elimination
  match fwe_model.lower():
    case "golub_beltrami":
      fwe_image = golub_beltrami(dwi, gtab, mask)
    case "dipy_fwdti":
      fwe_image = dipy_fwdti(dwi, gtab, mask)
    case _:
      print("Unrecognized free water elimination model")

  # save free water eliminated image
  nib.save(fwe_image, output_fname)
  print(f"Saved: {output_fname}")


def golub_beltrami(dwi, gtab, mask = None, init_method = "hybrid", 
                   Diso = 3.0e-3, Stissue = None, Swater = None,
                   n_iterations = 100, learning_rate = 0.0005):
   
  # adjust b-values for Beltrami model
  gtab.bvals = gtab.bvals * 1e-3

  # fit Beltrami regularized gradient descent free-water diffusion tensor model
  model = BeltramiModel(gtab, init_method = init_method, Diso = Diso * 1e3, 
                        Stissue = Stissue, Swater = Swater, 
                        iterations = n_iterations, 
                        learning_rate = learning_rate)
  model_fit = model.fit(dwi.get_fdata(), mask = mask)

  # extract free-water dti model parameters 
  fwf = model_fit.fw # free water fraction

  # undo b-value adjustment from Beltrami model
  gtab.bvals = gtab.bvals * 1e3

  # return free-water eliminated signal
  return remove_free_water(dwi, gtab, fwf, Diso)
  

def dipy_fwdti(dwi, gtab, mask = None, Diso = 3.0e-3):
  # fit free-water diffusion tensor imaging model
  model = fwdti.FreeWaterTensorModel(gtab)
  model_fit = model.fit(dwi.get_fdata(), mask = mask)  

  # extract free-water dti model parameters
  fwf = model_fit.model_params[..., -1]

  # return free-water eliminated signal
  return remove_free_water(dwi, gtab, fwf, Diso)


def remove_free_water(dwi, gtab, fwf, Diso):
  # extract b0 signal from dwi image
  S0 = dwi.get_fdata()[..., gtab.bvals == 0].mean(axis = -1)  

  # compute free-water signal
  fw_decay = np.exp(-gtab.bvals * Diso) # free-water exponential decay
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
  parser.add_argument("fwe_model", type = str)
  parser.add_argument("output_fname", type = str)
  args = parser.parse_args()
  
  free_water_elimination(
    dwi_fname    = args.dwi_fname,
    bval_fname   = args.bval_fname, 
    bvec_fname   = args.bvec_fname, 
    mask_fname   = args.mask_fname,
    fwe_model    = args.fwe_model, 
    output_fname = args.output_fname
  )