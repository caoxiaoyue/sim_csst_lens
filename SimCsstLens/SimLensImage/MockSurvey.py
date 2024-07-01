import numpy as np
import pickle
import os
import h5py
from multiprocessing import Pool
from SimCsstLens.SimLensImage import MassLightModel as MLM
from SimCsstLens.SimLensImage import Simulator
from SimCsstLens.SimLensImage import Util as SSU
from SimCsstLens.LensPop import Util as SLU
import yaml
from matplotlib import pyplot as plt

class MockSurvey(object):
    def  __init__(self, config_path='./', config_file='csst.yaml'):
        if not os.path.exists(config_path):  #check if path exist
            abs_path = os.path.abspath(config_path)  #get absolute path
            os.makedirs(abs_path) #create new directory recursively
        dest_file = os.path.join(config_path, config_file)

        with open(dest_file, 'r') as f:
            survey_info = yaml.load(f, Loader=yaml.FullLoader)

        self.dpix = survey_info['dpix']
        self.bands = survey_info['bands'] #['g', 'r', 'i', 'z']
        self.bands.append('stack') 
        self.zero_mag = survey_info['zero_mag']
        self.zero_exp_time = survey_info['zero_exp_time']
        self.sky_mag = survey_info['sky_mag']
        self.gains = survey_info['gains']
        #the precedence order to set the psf info is: sigma->fwhm->ree80
        self.psfs_sigma = survey_info['psfs_sigma']
        self.psfs_fwhm = survey_info['psfs_fwhm']
        self.psfs_ree80 = survey_info['psfs_ree80']
        self.n_exp = survey_info['n_exp']
        self.exp_time = survey_info['exp_time']
        self.survey_area = survey_info['survey_area']
        self.readout = survey_info['readout']
        self.dark_current = survey_info['dark_current'] #a float number, in e-/s/pix
        self.rnd_obs = survey_info['rnd_obs']
        self.rnd_obs_info = survey_info['rnd_obs_info']

        self.whole_sky_area = np.pi*4 * (180/np.pi)**2 #in deg^2 unit
        self.survey_sky_frac = self.survey_area / self.whole_sky_area

        self.npix = int(10.0/self.dpix)
        if (self.npix%2) != 0:
            self.npix += 1

        self.psf_kernel_dict = {}
        self.psf_fwhm_dict = {}
        self.skyback_dict = {}
        for ii,band in enumerate(self.bands[0:-1]):
            self.psf_kernel_dict[band] = SSU.gauss_psf(
                npix=None, 
                dpix=self.dpix, 
                sigma=self.psfs_sigma[ii], 
                fwhm=self.psfs_fwhm[ii], 
                ree80=self.psfs_ree80[ii], 
                nsub=4
            )
            self.skyback_dict[band] = SSU.sky_cps_from_mag(self.sky_mag[ii], self.zero_mag[ii], self.dpix) #sky background in unit of counts/s/pix

            if self.psfs_ree80[ii] is not None:
                self.psf_fwhm_dict[band] = self.psfs_ree80[ii]/np.sqrt(np.log(1-0.8)*(-2.0)) * (2*np.sqrt(2*np.log(2)))
            if self.psfs_sigma[ii] is not None:
                self.psf_fwhm_dict[band] = self.psfs_sigma[ii] * (2*np.sqrt(2*np.log(2)))
            if self.psfs_fwhm[ii] is not None:
                self.psf_fwhm_dict[band] = self.psfs_fwhm[ii]
        # self.psf_fwhm_dict['stack'] = max([val for val in self.psf_fwhm_dict.values()])
        self.psf_fwhm_dict['stack'] = sum(self.psf_fwhm_dict.values()) / len(self.psf_fwhm_dict.values())


    def load_ideal_lens_samples(
        self, 
        file_path='./', 
        file_name='ideal_lens_samples.hdf5'
    ):
        dest_file = os.path.join(file_path, file_name)
        fn = h5py.File(dest_file, "r")

        self.src_z = fn['source']['z'][()] #shape: [n_src_per_lens, n_ideal_lenses]
        self.src_xs = fn['source']['xs'][()]
        self.src_ys = fn['source']['ys'][()]
        self.src_Re = fn['source']['Re'][()]
        self.src_q = fn['source']['q'][()]
        self.src_pa = fn['source']['pa'][()]
        self.src_thetaE = fn['source']['thetaE'][()]
        # self.src_bool = fn['source']['bool_arr'][()] #shape: [n_src_per_lens, n_ideal_lenses]

        self.dfl_Re = fn['deflector']['Re'][()] #shape: [n_ideal_lenses]
        self.dfl_z = fn['deflector']['z'][()]
        self.dfl_q = fn['deflector']['q'][()]
        self.dfl_vdisp = fn['deflector']['vdisp'][()]
        # self.dfl_bool = fn['deflector']['bool_arr'][()]

        for band in self.bands[0:-1]:
            self.__dict__[f'src_app_mag_{band}'] = fn['source'][f'app_mag_{band}'][()]
            self.__dict__[f'dfl_app_mag_{band}'] = fn['deflector'][f'app_mag_{band}'][()]

        fn.close()
        self.n_ideal_lenses = len(self.dfl_z)


    def sim_obj_from(self, index):
        """
        return the simulator of a lens.
        a lens typically includes multiple background sources, and multi-band observation
        """
        sim_obj = {}
        nsrc_per_lens = self.src_z.shape[0]
        for ii in range(nsrc_per_lens): #ii is the index of background sources 
            sim_obj[ii] = {}
            this_lens_mass = [
                MLM.SieMass(
                    xc=0.0, 
                    yc=0.0, 
                    q=self.dfl_q[index], 
                    PA=90.0, #manually requirment, consistent to the lenspop code, just used for code debugging 
                    thetaE=self.src_thetaE[ii, index],        
                ),
            ]

            for jj, band in enumerate(self.bands[0:-1]):
                this_lens_light = [
                    MLM.SersicLight(        
                        xc=0.0, 
                        yc=0.0, 
                        q=self.dfl_q[index], 
                        PA=90.0, #manually requirment, consistent to the lenspop code, just used for code debugging
                        Re=self.dfl_Re[index], 
                        Ie=None, 
                        n=4.0,
                        m=(self.__dict__[f'dfl_app_mag_{band}'])[index],
                        mag_zero=self.zero_mag[jj],
                    ),
                ]

                this_source_light = [
                    MLM.SersicLight(
                        xc=self.src_xs[ii, index], 
                        yc=self.src_ys[ii, index], 
                        q=self.src_q[ii, index], 
                        PA=self.src_pa[ii, index], 
                        Re=self.src_Re[ii, index], 
                        Ie=None, 
                        n=1.0,
                        m=(self.__dict__[f'src_app_mag_{band}'])[ii, index],
                        mag_zero=self.zero_mag[jj],      
                    )
                ]

                sim_obj[ii][band] = Simulator.SingleBandSimulator(
                    lens_light_models=this_lens_light,
                    lens_mass_models=this_lens_mass,
                    source_light_models=this_source_light,
                )
                
        return sim_obj


    def lensing_image_from(self, sim_obj, seed=None):
        """
        run the lensing image simulation for the sim_obj
        lensing image are saved in the sim_obj internally
        """
        nsrc_per_lens = self.src_z.shape[0]
        nband_no_stack = len(self.bands) - 1

        eff_exp_time = np.zeros(nband_no_stack)
        for jj in range(nband_no_stack):
            eff_exp_time[jj] = self.exp_time[jj]/self.zero_exp_time*self.gains[jj] #effective expsoure time

        #lensing image simulation of each band
        for ii in range(nsrc_per_lens):
            for jj,this_band in enumerate(self.bands[0:-1]):
                sim_obj[ii][this_band].generate_ideal_image(npix=self.npix, dpix=self.dpix, nsub=2)
                sim_obj[ii][this_band].overlay_instrument_effect(
                    psf=self.psf_kernel_dict[this_band], #a 2d psf kernel image
                    skylevel=self.skyback_dict[this_band], #unit: counts/s/pix
                    dark_current=self.dark_current/self.gains[jj], #unit: adu counts/s/pix
                    readout_noise=self.readout, #unit: counts/pix
                    n_exposures=self.n_exp, 
                    exposure_time=eff_exp_time[jj], #effective expsoure time
                    seed=seed,
                )

        #stack all individual band
        for ii in range(nsrc_per_lens):
            sim_obj[ii]['stack'] = {}
            sim_obj[ii]['stack']['eff_exp_time'] = np.sum(eff_exp_time)
            sim_obj[ii]['stack']['lens_image_cts'] = np.zeros((self.npix, self.npix))
            sim_obj[ii]['stack']['image_cts'] = np.zeros((self.npix, self.npix))
            sim_obj[ii]['stack']['unlensed_src_cts'] = 0.0
            # sim_obj[ii]['stack']['image_map_cps'] = np.zeros((self.npix, self.npix))
            # sim_obj[ii]['stack']['noise_map_cps'] = np.zeros((self.npix, self.npix))
            variance_counts = np.zeros((self.npix, self.npix))
            for jj,this_band in enumerate(self.bands[0:-1]):
                variance_counts += (sim_obj[ii][this_band].noise_map_cps * eff_exp_time[jj])**2
                sim_obj[ii]['stack']['lens_image_cts'] += sim_obj[ii][this_band].blurred_lens_image_cps * eff_exp_time[jj]
                sim_obj[ii]['stack']['image_cts'] += sim_obj[ii][this_band].blurred_image_cps * eff_exp_time[jj]
                sim_obj[ii]['stack']['unlensed_src_cts'] += sim_obj[ii][this_band].unlensed_source_flux * eff_exp_time[jj]
            sim_obj[ii]['stack']['noise_map_cps'] = np.sqrt(variance_counts)/sim_obj[ii]['stack']['eff_exp_time']
            sim_obj[ii]['stack']['lens_image_cps'] = sim_obj[ii]['stack']['lens_image_cts']/sim_obj[ii]['stack']['eff_exp_time']
            sim_obj[ii]['stack']['lensed_arc_cps'] = (sim_obj[ii]['stack']['image_cts'] - sim_obj[ii]['stack']['lens_image_cts'])/sim_obj[ii]['stack']['eff_exp_time']
            sim_obj[ii]['stack']['image_map_cps'] = sim_obj[ii]['stack']['image_cts'] / sim_obj[ii]['stack']['eff_exp_time']
            sim_obj[ii]['stack']['magnification'] = np.sum(sim_obj[ii]['stack']['image_cts'] - sim_obj[ii]['stack']['lens_image_cts'])/sim_obj[ii]['stack']['unlensed_src_cts']
            sim_obj[ii]['stack']['image_map_cps'] += np.random.normal(loc=0.0, scale=sim_obj[ii]['stack']['noise_map_cps'])
            sim_obj[ii]['stack']['extent'] = sim_obj[ii][this_band].extent


    def sn_and_mu_from(self, sim_obj):
        nsrc_per_lens = self.src_z.shape[0]
        nband = len(self.bands)
        sn = np.zeros((nsrc_per_lens, nband))
        mu = np.zeros((nsrc_per_lens, nband))

        for ii in range(nsrc_per_lens):
            for jj,band in enumerate(self.bands[0:-1]):
                lensed_arc_cps = sim_obj[ii][band].blurred_image_cps - sim_obj[ii][band].blurred_lens_image_cps
                sn[ii, jj] = SSU.SN_from_lensed_image(
                    lensed_arc_cps, 
                    sim_obj[ii][band].noise_map_cps,
                )
                mu[ii, jj] = sim_obj[ii][band].magnification

            # save the “stacked" band info 
            sn[ii, -1] = SSU.SN_from_lensed_image(
                    sim_obj[ii]['stack']['lensed_arc_cps'], 
                    sim_obj[ii]['stack']['noise_map_cps'],
            )
            mu[ii, -1] = sim_obj[ii]['stack']['magnification']
        
        return sn, mu


    def simulate_observed_lenses(
        self,
        output_image=False,
        output_path='./output/images',
    ):
        if output_image: #Note!!!, when using multiprocess, mkdir a new folder may lead to a sync bug
            #when a thread begin, the code may find the destination folder do not exsit,
            #however when this thread begin to create the folder, that folder may already be created by 
            #another thread.
            # if not os.path.exists(output_path):  #check if path exist
            #     abs_path = os.path.abspath(output_path)  #get absolute path
            #     os.makedirs(abs_path) #create new directory recursively
            try:
                abs_path = os.path.abspath(output_path)
                os.makedirs(abs_path)
            except IOError:
                pass

        self.sn_arr = np.zeros((self.src_z.shape[0], len(self.bands), self.n_ideal_lenses))
        self.mu_arr = np.zeros((self.src_z.shape[0], len(self.bands), self.n_ideal_lenses))
        self.obs_cond = np.zeros((self.src_z.shape[0], len(self.bands), self.n_ideal_lenses, 3), dtype='bool')
        self.ring_cond = np.zeros((self.src_z.shape[0], len(self.bands), self.n_ideal_lenses), dtype='bool') #NOte only works for SIE now

        for ii in range(self.n_ideal_lenses):
            sim_obj = self.sim_obj_from(ii)
            self.lensing_image_from(sim_obj, seed=ii)
            sn, mu = self.sn_and_mu_from(sim_obj)

            self.sn_arr[:, :, ii] = np.copy(sn)
            self.mu_arr[:, :, ii] = np.copy(mu)

            for jj in range(self.src_z.shape[0]):
                for kk, band in enumerate(self.bands):
                    self.obs_cond[jj, kk, ii, 0] = SSU.condition_resolved_image(
                        self.src_thetaE[jj, ii], 
                        self.src_Re[jj, ii], 
                        self.psf_fwhm_dict[band],
                    )

                    self.obs_cond[jj, kk, ii, 1] = SSU.condition_tangential_arc(
                        mu[jj, kk], 
                        self.src_Re[jj, ii], 
                        self.psf_fwhm_dict[band],
                    )

                    self.obs_cond[jj, kk, ii, 2] = SSU.condition_images_sn(sn[jj, kk])

                    if band != 'stack':
                        self.ring_cond[jj, kk, ii] = SSU.check_ring(
                            sim_obj[jj][band].lens_mass_models[0],
                            self.src_xs[jj, ii],
                            self.src_ys[jj, ii],
                            self.src_Re[jj, ii],
                            self.src_q[jj, ii],
                            self.src_pa[jj, ii],
                        ) #Note, only works for a single SIE lens
                    else:
                        self.ring_cond[jj, -1, ii] = np.sum((self.ring_cond[jj, 0:-1, ii]).astype('int'))>0

                    if output_image is True:
                        self.output_lens_image(sim_obj, ii, jj, kk, band, output_path=output_path)


    def append_obs_info(
        self, 
        output_path='./output', 
        output_file=f'ideal_lens_samples.hdf5'     
    ):
        dest_file = os.path.join(output_path, output_file)
        fn = h5py.File(dest_file, "r+")

        SLU.save_hdf5_overwrite(
            fn,
            "Obs/SNR", 
            data=self.sn_arr,
        )

        SLU.save_hdf5_overwrite(
            fn,
            "Obs/magnification", 
            data=self.mu_arr,
        )

        SLU.save_hdf5_overwrite(
            fn,
            "Obs/detect_cond", 
            data=self.obs_cond,
        )

        SLU.save_hdf5_overwrite(
            fn,
            "Obs/ring_cond", 
            data=self.ring_cond,
        )

        fn.close()

    
    def output_lens_image(
        self, 
        sim_obj,
        lens_id,
        src_id,
        band_id,
        band_name, 
        output_path='./output_lens_im'
    ):
        if band_name != 'stack':
            lensed_arc = sim_obj[src_id][band_name].blurred_image_cps - sim_obj[src_id][band_name].blurred_lens_image_cps
            plt.figure(figsize=(15,10))
            plt.subplot(231)
            plt.imshow(
                lensed_arc, 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id][band_name].extent,
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('ideal src')
            plt.subplot(232)
            plt.imshow(
                sim_obj[src_id][band_name].blurred_lens_image_cps, 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id][band_name].extent,
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('ideal lens')
            plt.subplot(233)
            plt.imshow(
                sim_obj[src_id][band_name].noise_map_cps, 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id][band_name].extent,
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('noise map')
            plt.subplot(234)
            plt.imshow(
                lensed_arc/sim_obj[src_id][band_name].noise_map_cps, 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id][band_name].extent,
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('ideal src SN')
            plt.subplot(235)
            plt.imshow(
                (sim_obj[src_id][band_name].image_map_cps - sim_obj[src_id][band_name].blurred_lens_image_cps)/sim_obj[src_id][band_name].noise_map_cps, 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id][band_name].extent,
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('SN-{:.2f}'.format(self.sn_arr[src_id, band_id, lens_id]))
            plt.tight_layout()
            plt.savefig(f'{output_path}/{lens_id}_{src_id}_{band_name}.pdf', bbox_inches='tight')
            plt.close()
        else:
            plt.figure(figsize=(15,5))
            plt.subplot(131)
            plt.imshow(
                sim_obj[src_id]['stack']['image_map_cps'], 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id]['stack']['extent'],
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('image')
            plt.subplot(132)
            plt.imshow(
                sim_obj[src_id]['stack']['noise_map_cps'], 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id]['stack']['extent'],
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('noise')
            plt.subplot(133)
            plt.imshow(
                (sim_obj[src_id]['stack']['image_map_cps'] - sim_obj[src_id]['stack']['lens_image_cps'])/sim_obj[src_id][band_name]['noise_map_cps'],
                # sim_obj[src_id]['stack']['lensed_arc_cps']/sim_obj[src_id]['stack']['noise_map_cps'], 
                origin='lower', 
                cmap='jet', 
                extent=sim_obj[src_id]['stack']['extent'],
            )
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title('SN-{:.2f}'.format(self.sn_arr[src_id, -1, lens_id]))
            plt.tight_layout()
            plt.savefig(f'{output_path}/{lens_id}_{src_id}_{band_name}.pdf', bbox_inches='tight')
            plt.close()