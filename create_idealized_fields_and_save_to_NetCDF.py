import os
import numpy as np
from netCDF4 import Dataset

def save_netcdf(fout = None, displacements = None, id= None, name = None, out_dir = None):
	
	fname_nc = out_dir + id + ".nc"
	
	displacements_np = np.asarray(displacements)
	
	rootgrp = Dataset(fname_nc, "w", format="NETCDF4")
	rootgrp.id = id
	rootgrp.description = name
	
	nc_d = rootgrp.createDimension("dim_displacement", fout.shape[0]) 
	nc_value = rootgrp.createDimension("dim_value", fout.shape[1])
	nc_x = rootgrp.createDimension("dim_x", fout.shape[2]) 
	
	nc_var_displacement = rootgrp.createVariable("displacement","i4",("dim_displacement",)) 
	nc_var_displacement[:] = displacements_np
	nc_var_displacement.description = "Displacement"
	
	nc_var_value = rootgrp.createVariable("value","f4",("dim_value",))
	nc_var_value[:] =  np.asarray(range(fout.shape[1]))
	nc_var_value.description = "Variable value"
	
	nc_var_x = rootgrp.createVariable("x","f4",("dim_x",)) 
	nc_var_x[:] = np.asarray(range(fout.shape[2]))
	nc_var_x.description = "Spatial coordinate"
	
	nc_var = rootgrp.createVariable("prob_density","f4",("dim_displacement", "dim_value", "dim_x"))
	nc_var[:] = fout
	nc_var.description = "Probability density"
	
	rootgrp.close()                                   # zapremo datoteko 


out_dir = "fields/"
os.makedirs(out_dir, exist_ok=True)

#---------------------------------------------------------------------------------
# Probabilistic fields
#---------------------------------------------------------------------------------

# -----------------------------------------------
name = "Constant Gaussian"
id = "P00"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Constant Gaussian - wide"
id = "P01"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	sigma=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Constant Gaussian - narrow"
id = "P02"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	sigma=3
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian gradient"
id = "P03"
displacements = [0]
fout = []
for d in displacements:
	y0=25
	y1=75
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + i/100*(y1-y0)
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Constant Gaussian - width transition"
id = "P04"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	sigma1=5
	sigma2=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		sigma = sigma1 + (sigma2-sigma1)/100*i
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
id = "P05"
name = "Localized uncertainty event"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25+d
	sigma=10
	sigmax=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		sigmay = sigma + sigma*np.exp(-(i-x0)**2/(2*sigmax**2))
		out = np.exp(-(y-y0)**2/(2*sigmay**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Binomal Gaussians"
id = "P06"
displacements = [0]
fout = []
for d in displacements:
	y0=25
	y1=75
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out += np.exp(-(y-y1)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Binomal Gaussians - narrow/wide"
id = "P07"
displacements = [0]
fout = []
for d in displacements:
	y0=25
	y1=75
	sigma=10
	sigma1=5
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.exp(-(y-y0)**2/(2*sigma**2))
		out += np.exp(-(y-y1)**2/(2*sigma1**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian split"
id = "P08"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	y1=75
	y2=25
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y1temp = y0 + i/100*(y1-y0)
		y2temp = y0 + i/100*(y2-y0)
		out = np.exp(-(y-y1temp)**2/(2*sigma**2))
		out += np.exp(-(y-y2temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian split - nonsymetric"
id = "P09"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	y1=75
	y2=25
	sigma=10
	sigma2=5
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y1temp = y0 + i/100*(y1-y0)
		y2temp = y0 + i/100*(y2-y0)
		out = np.exp(-(y-y1temp)**2/(2*sigma**2))
		out += np.exp(-(y-y2temp)**2/(2*sigma2**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian event"
id = "P10"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25+d
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Less intense Gaussian event"
id = "P11"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25+d
	sigma=10
	sigmax=10
	ampx=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Narrow Gaussian event"
id = "P12"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25 + d
	sigma=10
	sigmax=3
	ampx=20
	f = np.zeros((101,101))
	
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Wide Gaussian event"
id = "P13"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25+d
	sigma=10
	sigmax=20
	ampx=20
	f = np.zeros((101,101))
	
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Two Gaussian events"
id = "P14"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	x0=25
	x1=75
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) + ampx *np.exp(-(i-x1)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Two Gaussian events - different amplitude"
id = "P15"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	x0=25
	x1=75
	sigma=10
	sigmax=10
	ampx=20
	ampx1=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) + ampx1 *np.exp(-(i-x1)**2/(2*sigmax**2))
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian step"
id = "P16"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	x0=25 + d
	sigma=10
	f = np.zeros((101,101))
	y0 = 25
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		strech = 4
		y0temp = y0 + np.exp( (i-x0) / strech) / (1 + np.exp((i-x0)/strech)) * 50
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Constant probability"
id = "P17"
displacements = [0]
fout = []
for d in displacements:
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		out = np.ones(y.shape)
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian event with some noise"
id = "P18"
displacements = [0]
fout = []
for d in displacements:
	rng = np.random.default_rng(seed=7)
	y0=50
	x0=25
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	y0_nois = rng.integers(low=-20, high=20, size=f.shape[1], endpoint=True)
	y0_nois[:80] = 0
	rng.shuffle(y0_nois)
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + y0_nois[i] + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) 
		out = np.exp(-(y-y0temp)**2/(2*sigma**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Random-width Gaussian event"
id = "P19"
displacements = [0]
fout = []
for d in displacements:
	rng = np.random.default_rng(seed=3)
	f = np.zeros((101,101))
	sigma = 5 + rng.random(f.shape[1])*10
	sigmax=10
	ampx=20
	y0=50
	x0=25
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) 
		out = np.exp(-(y-y0temp)**2/(2*sigma[i]**2))
		out = out/np.sum(out)
		f[:,i] = out
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

#---------------------------------------------------------------------------------
# Deterministic fields
#---------------------------------------------------------------------------------


# -----------------------------------------------
name = "Constant"
id = "D00"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		f[y0,:] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gradient"
id = "D01"
displacements = [0]
fout = []
for d in displacements:
	y0=25
	y1=75
	sigma=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + i/100*(y1-y0)
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian Event"
id = "D02"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25+d
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Less intense Gaussian event"
id = "D03"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25 + d
	sigma=10
	sigmax=10
	ampx=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Narrow Gaussian event"
id = "D04"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25 + d
	sigma=10
	sigmax=3
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Wide Gaussian event"
id = "D05"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	y0=50
	x0=25 + d
	sigma=10
	sigmax=20
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Two Gaussian events"
id = "D06"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	x0=25
	x1=75
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) + ampx *np.exp(-(i-x1)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Two Gaussian events - different amplitude"
id = "D07"
displacements = [0]
fout = []
for d in displacements:
	y0=50
	x0=25
	x1=75
	sigma=10
	sigmax=10
	ampx=20
	ampx1=10
	f = np.zeros((101,101))
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + ampx *np.exp(-(i-x0)**2/(2*sigmax**2)) + ampx1 *np.exp(-(i-x1)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Step"
id = "D08"
displacements = list(range(0,51,1))
fout = []
for d in displacements:
	x0=25 + d
	sigma=10
	f = np.zeros((101,101))
	y0 = 25
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		strech = 4
		y0temp = y0 + np.exp( (i-x0) / strech) / (1 + np.exp((i-x0)/strech)) * 50
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Random noise"
id = "D09"
displacements = [0]
fout = []
for d in displacements:
	rng = np.random.default_rng(seed=1)
	f = np.zeros((101,101))
	y0= rng.integers(low=0,high=100,size=f.shape[1], endpoint=True)
	for i in range(f.shape[1]):
		f[y0[i],i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

# -----------------------------------------------
name = "Gaussian event with some noise"
id = "D10"
displacements = [0]
fout = []
for d in displacements:
	rng = np.random.default_rng(seed=14)
	y0=50
	x0=25
	sigma=10
	sigmax=10
	ampx=20
	f = np.zeros((101,101))
	y0_nois = rng.integers(low=-20, high=20, size=f.shape[1], endpoint=True)
	y0_nois[:80] = 0
	rng.shuffle(y0_nois)
	for i in range(f.shape[1]):
		y=np.asarray(list(range(f.shape[0])))
		y0temp = y0 + y0_nois[i] + ampx *np.exp(-(i-x0)**2/(2*sigmax**2))
		f[int(y0temp),i] = 1
	fout.append(f)

fout = np.asarray(fout)
save_netcdf(fout, displacements=displacements, id=id, name=name, out_dir = out_dir)

