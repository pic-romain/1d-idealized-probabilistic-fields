# ------------------------------
# Import the needed libraries
# ------------------------------
# Import numpy 
import numpy as np

# Import matplotlib 
import matplotlib as matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)
import os
import string

from utils import read_case

in_dir = "fields/"
subfigs_out_dir = "selected_comparisons/"
montage_out_dir = "figures/"
os.makedirs(subfigs_out_dir, exist_ok=True)
os.makedirs(montage_out_dir, exist_ok=True)


comparisons = []
comparisons.append(["D02","D00"])
comparisons.append(["D02","P00"])
comparisons.append(["D02","P01"])
comparisons.append(["D02","P17"])
comparisons.append(["D02","P04"])
comparisons.append(["D02","P04h"])
comparisons.append(["D02","P06"])

comparisons.append(["D02","P10"])
comparisons.append(["D02","P10_25"])
comparisons.append(["D02","P10_50"])
comparisons.append(["D02","P10_50v"])
comparisons.append(["D02","P05"])
comparisons.append(["D02","P05_25"])

comparisons.append(["D02","P18"])
comparisons.append(["D02","P19"])

comparisons.append(["D02","P16_25h"])

for i, comp in enumerate(comparisons):
	
	id1=comp[0]
	id2=comp[1]
	
	f1,x,value = read_case(id1, in_dir)
	f2,x,value = read_case(id2, in_dir)
	
	# visulaize comparison
	fig, ax = plt.subplots(figsize=(5, 5))
	vmax_temp = np.max(f2)
	if np.average(f2) > vmax_temp*0.5:
		vmax_temp=vmax_temp*3
	cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white","#1f6fb3"],512)
	cmap_r = matplotlib.colors.LinearSegmentedColormap.from_list('rb_cmap',["white",(1.0,0.3,0.3)],512)
	norm_b = matplotlib.colors.Normalize(vmin=0, vmax=vmax_temp)
	norm_r = matplotlib.colors.Normalize()
	# convert to rgb values using colormaps
	f1x = cmap_b(norm_b(f2))
	f2x = cmap_r(norm_r(f1))
	# get the combined colors for the image using the multiply effect
	fx = f1x*f2x
	img_extent = (-0.5, 100.5, -0.5, 100.5)
	img = ax.imshow(fx, interpolation='nearest', origin='lower', extent=img_extent)
	plt.xlim(left = -0.5, right = 100.5)
	plt.ylim(bottom = -0.5, top = 100.5)
	ax.set_title("(" + string.ascii_lowercase[i]+r") $\mathbf{"+ id1.replace("_",r"\_") + r" - " + id2.replace("_",r"\_") + r"}$", fontsize=16, pad=12)
	#ax.set_title(prefix + id1 + r" - " + id2, fontsize=20, pad=12)
	ax.tick_params(top=True, right=True, which='both')
	ax.xaxis.set_major_locator(MultipleLocator(20))
	ax.xaxis.set_minor_locator(MultipleLocator(10))
	ax.yaxis.set_major_locator(MultipleLocator(20))
	ax.yaxis.set_minor_locator(MultipleLocator(10))
	plt.ylabel("Variable value")
	plt.xlabel("$x$ (spatial coordinate)")
	#print(fname)
	fname = subfigs_out_dir + "C" + str(i).zfill(2) + ".png"
	plt.savefig(fname, dpi=300, bbox_inches='tight')
	#plt.show()
	plt.close()


os.system("montage "+subfigs_out_dir+"C*.png -tile 4x -geometry 600x "+montage_out_dir+"comparisons.png")