import numpy as np
from SimCsstLens.SimLensImage import Util
import scipy.signal as signal
from astropy.convolution import convolve, convolve_fft


class SingleBandSimulator(object):
    """
    Input:
    Lens light + mass, + Src light, pars-dict
    number of sources
    psf-sigma or a kernel image,
    dpix,
    background,
    readout,
    exp time,
    mag zero point
    Output:
    image, noise,
    magnification,
    SN of arc,
    """
    def __init__(
        self,
        lens_light_models=None,
        lens_mass_models=None,
        source_light_models=None,
    ):
        self.lens_light_models = lens_light_models
        self.lens_mass_models = lens_mass_models
        self.source_light_models = source_light_models


    def generate_ideal_image(self, npix, dpix, nsub, if_eval_deflections=True):
        self.xgrid, self.ygrid = Util.make_grid_2d(npix, dpix, nsub)
        if if_eval_deflections:
            self.alphax, self.alphay = self.eval_mass_deflections()
        self.npix = npix
        self.dpix = dpix
        self.nsub = nsub
        self.extent = [-0.5*npix*dpix, 0.5*npix*dpix, -0.5*npix*dpix, 0.5*npix*dpix]

        lens_image_over_samp = self.eval_lens_image()
        self.ideal_lens_image = Util.bin_image(lens_image_over_samp, nsub) #unit: counts/s/arcsec^2

        source_image_over_samp, self.unlensed_source_flux = self.eval_source_image()
        self.ideal_source_image = Util.bin_image(source_image_over_samp, nsub) #unit: counts/s/arcsec^2

        self.ideal_image = self.ideal_lens_image + self.ideal_source_image #unit: counts/s/arcsec^2

        self.ideal_image_cps = self.ideal_image * dpix**2 #unit: counts/s or counts/s/pix
        self.ideal_source_image_cps = self.ideal_source_image * dpix**2 #unit: counts/s or counts/s/pix

        self.magnification = np.sum(self.ideal_source_image_cps) / self.unlensed_source_flux


    def eval_lens_image(self):
        lens_light_image = np.zeros_like(self.xgrid)
        if self.lens_light_models is not None:
            for item in self.lens_light_models:
                lens_light_image += item.brightness_at(self.xgrid, self.ygrid)
        return lens_light_image


    def eval_mass_deflections(self):
        alphax = np.zeros_like(self.xgrid)
        alphay = np.zeros_like(self.xgrid)
        if self.lens_mass_models is not None:
            for item in self.lens_mass_models:
                alphax_tmp, alphay_tmp = item.deflection_at(self.xgrid, self.ygrid)
                alphax += alphax_tmp
                alphay += alphay_tmp
        return alphax, alphay


    def eval_source_image(self):
        source_image = np.zeros_like(self.alphax)
        unlensed_source_flux = 0.0
        if self.source_light_models is not None:
            for item in self.source_light_models:
                source_image += item.brightness_at(
                    self.xgrid-self.alphax, 
                    self.ygrid-self.alphay,
                )
                unlensed_source_flux += item.total_flux
        return source_image, unlensed_source_flux


    def overlay_instrument_effect(
        self,
        psf=None, #a 2d psf kernel image
        skylevel=0.5, #unit: counts/s/pix
        dark_current=0.0, #unit: counts/s/pix
        readout_noise=0.0, #unit: counts/pix
        n_exposures=1, 
        exposure_time=300.0, #unit: seconds
        seed=None,
    ):
        if psf is None:
            self.blurred_image = np.copy(self.ideal_image)
            self.blurred_lens_image = np.copy(self.ideal_lens_image)
        else:
            self.blurred_image = convolve_fft(self.ideal_image, psf) #unit: counts/s/arcsec^2
            self.blurred_lens_image = convolve_fft(self.ideal_lens_image, psf) #unit: counts/s/arcsec^2
        self.blurred_image_cps = self.blurred_image * self.dpix**2  #unit: counts/s/pixel
        self.blurred_lens_image_cps = self.blurred_lens_image * self.dpix**2  #unit: counts/s/pixel

        self.image_map_cps, self.noise_map_cps = Util.add_noise_to_image_gaussian(
            ideal_image=self.blurred_image_cps, 
            skylevel=skylevel, 
            dark_current=dark_current, 
            readout_noise=readout_noise, 
            n_exposures=n_exposures, 
            total_exposure_time=exposure_time, 
            seed=seed,
        ) #units of self.image_map_cps and self.noise_map_cps are counts/s/pixel

        self.image_map = self.image_map_cps/self.dpix**2 #unit: counts/s/arcsec^2
        self.noise_map = self.noise_map_cps/self.dpix**2 #unit: counts/s/arcsec^2

    