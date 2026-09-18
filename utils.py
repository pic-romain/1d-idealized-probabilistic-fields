import numpy as np
import xarray as xr
from scipy import signal

from netCDF4 import Dataset

def read_case(id, in_dir):
	split = id.split("_")
	if len(split) == 1:
		d=0;
	else:
		d=int(split[1][:2])
	
	nc_file_id = Dataset(in_dir+"/"+id[:3]+".nc", 'r') # prebere natcdf datoteko
	x = nc_file_id.variables["x"][:].data
	value = nc_file_id.variables["value"][:].data
	fout = nc_file_id.variables["prob_density"][:].data
	f = fout[d,:,:]
	nc_file_id.close()
	
	# mirror fields if needed
	if "h" in id: f=np.flip(f,axis=1)
	if "v" in id: f=np.flip(f,axis=0)
	
	return([f,x,value])

def ensmean_fraction_skill_score(obs, forecast, threshold, window_size):
    """
    Calculate Fraction Skill Score (FSS) for xarray datasets.
    
    Parameters
    ----------
    obs : xr.Dataset or xr.DataArray
        Observed data
    forecast : xr.Dataset or xr.DataArray
        Forecast data (same shape as obs)
    threshold : float
        Threshold value for binary classification
    window_size : int
        Size of the moving window (must be odd)
    
    Returns
    -------
    float
        FSS value between 0 and 1
    """
    # Convert to numpy arrays
    obs_data = obs.values if isinstance(obs, xr.DataArray) else obs[list(obs.data_vars)[0]].values
    fcst_data = forecast.values if isinstance(forecast, xr.DataArray) else forecast[list(forecast.data_vars)[0]].values
    
    # Binary fields
    obs_binary = (obs_data >= threshold).astype(float)
    fcst_binary = (fcst_data >= threshold).astype(float)
    
    # Create moving window kernel
    kernel = np.ones((window_size,)) / (window_size)
    
    # Calculate fractions within windows
    # print(obs_binary.shape, fcst_binary.shape)
    # print(kernel.shape, kernel.shape)
    obs_frac = signal.convolve(obs_binary, kernel, mode='valid', method='direct')
    fcst_frac = signal.convolve(fcst_binary, kernel, mode='valid', method='direct')
    # print(obs_frac.shape, fcst_frac.shape)
    
    # FSS formula
    mse = np.mean((fcst_frac - obs_frac) ** 2)
    mse_worst = np.mean(obs_frac ** 2) + np.mean(fcst_frac ** 2)
    
    fss = 1 - (mse / mse_worst) if mse_worst > 0 else np.nan
    
    return float(fss)