#%%
import yaml
obs_info = {
    'dpix': 0.074, #pixel size in arcsec
    'bands': ['g', 'r', 'i', 'z'], #observation bands
    'zero_mag': [26.23, 26.04, 25.85, 25.27], #the instrumental zero magnitude
    'zero_exp_time': 1.0, #the exposure time which defined the zero_mag
    'sky_mag': [22.57, 22.10, 21.87, 21.86], #sky brightness, in mag/arcsec^2 unit
    'gains': [1.0, 1.0, 1.0, 1.0],
    #the psf kernel are represented with a gaussian function,
    #you can set the width of psf, either via the ree80, sigma,
    #or fwhm of the gauss.
    #ree80: the 2d aperture radius which encloses 80% total energy of the gauss
    'psfs_ree80': [None, None, None, None],
    #the sigma value of the 2d gaussian function 
    'psfs_sigma': [None, None, None, None],
    #the full-width-half-maximum of the gaussian function, it is defined in 1d
    'psfs_fwhm': [0.051, 0.064, 0.076, 0.089],
    'n_exp': 2, #number of exposure for y-band (not simulated here), csst have 4 exposures
    'exp_time': [300, 300, 300, 300], #total expsoure time
    'survey_area': 17500, #in deg^2 unit
    'readout': 5.0, #e-/pix
    'dark_current': 0.02, #e-/pix/s
    'rnd_obs': False,
    'rnd_obs_info': [None, None, None, None], 
}

with open('csst_setting.yaml', 'w') as f:
    yaml.dump(obs_info, f)

#%%

