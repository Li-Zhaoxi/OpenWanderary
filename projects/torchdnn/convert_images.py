import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
	dsroot = "D:/01 - datasets/008 - 2DMRACerebrovascular/"
	# gtpath = "D:/01 - datasets/008 - 2DMRACerebrovascular/gt_npy/mra_img_0.npy"
	
	npyimgroot = os.path.join(dsroot, "images_npy")
	jpgimgroot = os.path.join(dsroot, "images")
	npygtroot = os.path.join(dsroot, "gt_npy")
	pnggtroot = os.path.join(dsroot, "labels")

	filenames = []
	for filename in os.listdir(npyimgroot):
		imgpath = os.path.join(npyimgroot, filename)
		gtpath = os.path.join(npygtroot, filename)
		if not os.path.exists(gtpath):
			print(f"cannot find path: {gtpath}")
		else:
			filenames.append(filename)
	print(f"find {len(filenames)} files")

	csvpath = os.path.join(dsroot, "npyfilenames.csv")
	csvlist = list(zip(filenames, filenames))
	filenamefile = pd.DataFrame(data=csvlist, columns=['imagename', 'gtname'])
	filenamefile.to_csv(csvpath, index=False)

	imgnames, gtnames = [], []
	for filename in tqdm(filenames):
		imgpath = os.path.join(npyimgroot, filename)
		gtpath = os.path.join(npygtroot, filename)
		img = np.load(imgpath)
		img = (img * 255).astype(np.uint8)

		gt = np.load(gtpath)
		gt = (gt * 255).astype(np.uint8)
		
		basename = filename[:-4]
		saveimgpath = os.path.join(jpgimgroot, basename + ".jpg")
		cv2.imwrite(saveimgpath, img)
		imgnames.append(basename + ".jpg")

		savegtpath = os.path.join(pnggtroot, basename + ".png")
		cv2.imwrite(savegtpath, gt)
		gtnames.append(basename + ".png")
	
	csvpath = os.path.join(dsroot, "filenames.csv")
	csvlist = list(zip(imgnames, gtnames))
	filenamefile = pd.DataFrame(data=csvlist, columns=['imagename', 'gtname'])
	filenamefile.to_csv(csvpath, index=False)

	
		

		

		