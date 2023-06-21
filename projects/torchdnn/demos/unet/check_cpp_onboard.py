import numpy as np
import os
from prepare_functions import check_matrix_equal

if __name__ == "__main__":
  dataroot = "data/unet/"
  gtdatapath = os.path.join(dataroot, "unet_checkstage2.npz")
  cppdatapath = os.path.join(dataroot, "unet_checkcppresults.npz")
  
  gtdata = np.load(gtdatapath)
  gt_datain, gt_dataout, gt_datapred = gtdata["datain"], gtdata["dataout"], gtdata["pred"]

  cppdata = np.load(cppdatapath)
  cpp_datain, cpp_dataout, cpp_datapred = cppdata["datain"], cppdata["dataout"], cppdata["pred"]

  print("start check preprocess")
  check_matrix_equal(cpp_datain, gt_datain, 1e-4, dataroot, "cpp_preprocess")

  print("start check postprocess")
  check_matrix_equal(cpp_datapred, gt_datapred, 1e-4, dataroot, "cpp_postprocess")

  print("start check infer")
  check_matrix_equal(cpp_dataout, gt_dataout, 1e-4, dataroot, "cpp_infer")
