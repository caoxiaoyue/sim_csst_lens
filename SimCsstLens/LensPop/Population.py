import numpy as np
from SimCsstLens.DeflectorPop.Population import DeflectorPopulation
from SimCsstLens.SourcePop.Population import SourcePopulation
from SimCsstLens.LensPop import Util as SLU
import h5py
import os

class LensPopulation(object):
    def __init__(
        self,
        vdisp_floor=100, 
        zl_max=2.0, 
        cosmo_dist=None,
        src_catalog_type='lsst',
        bands=['g', 'r', 'i', 'z'],
        n_vdisp_bins=20,
    ) -> None:
        self.vdisp_floor = vdisp_floor
        self.zl_max = zl_max
        self.cosmo_dist = cosmo_dist
        self.src_catalog_type = src_catalog_type
        self.bands = bands
        self.n_vdisp_bins = n_vdisp_bins
        self.vdisp_bin_edges = np.linspace(vdisp_floor, 400.0, n_vdisp_bins+1)

        self.dfl_pop = DeflectorPopulation(
            vdisp_floor=self.vdisp_floor,
            zl_max=self.zl_max,
            cosmo_dist=self.cosmo_dist,
            bands=self.bands,
        )

        self.src_pop = SourcePopulation(
            catalog_type=self.src_catalog_type,
            bands=self.bands,
            cosmo_dist=self.cosmo_dist
        )


    def draw_lens_samples(
        self, 
        nsamples=100,
        nsrcs_per_sample=1,
        src_over_density=1,
        target_nsrc=1,
    ):
        self.nsamples = nsamples
        self.nsrcs_per_sample = nsrcs_per_sample
        self.src_over_density = src_over_density

        self.dfl_pop.draw_deflector_samples(self.nsamples)
        self.src_pop.draw_source_samples(
            nsamples=self.nsamples,
            nsrcs_per_sample=self.nsrcs_per_sample,
            over_density=self.src_over_density,
        )

        #count the number of ETGs in each vdisp bins
        self.n_etg_arr = np.histogram(self.dfl_pop.vdisp_arr, bins=self.vdisp_bin_edges)[0] #shape: (n_vdisp_bins,)

        #calculate the einstein radius (in arcsec) of each source
        zl_tmp_arr = np.zeros_like(self.src_pop.zs_arr)
        vdisp_tmp_arr = np.zeros_like(self.src_pop.zs_arr)
        self.thetaE_arr = np.zeros_like(self.src_pop.zs_arr)
        self.einstein_mass_arr = np.zeros_like(self.src_pop.zs_arr)
        for ii in range(self.nsrcs_per_sample):
            zl_tmp_arr[ii*self.nsamples:(ii+1)*self.nsamples] = self.dfl_pop.zl_arr[:]
            vdisp_tmp_arr[ii*self.nsamples:(ii+1)*self.nsamples] = self.dfl_pop.vdisp_arr[:]

        tmp_bool = (self.src_pop.zs_arr>zl_tmp_arr)
        self.thetaE_arr[tmp_bool] = SLU.vdisp_to_thetaE(
            zl_tmp_arr[tmp_bool], 
            self.src_pop.zs_arr[tmp_bool], 
            vdisp_tmp_arr[tmp_bool], 
            cosmos=self.cosmo_dist,
        )
        self.einstein_mass_arr[tmp_bool] = SLU.einstein_mass_from(
            self.thetaE_arr[tmp_bool],
            zl_tmp_arr[tmp_bool],
            self.src_pop.zs_arr[tmp_bool],
            astropy_cosmos=self.cosmo_dist.cosmos,
        )

        #check if a lens candiate is real
        self.ideal_lens_bool_src = self.check_if_lensing(
            self.src_pop.xs_arr,
            self.src_pop.ys_arr,
            self.thetaE_arr,
        )
        tmp_arr = self.ideal_lens_bool_src.reshape(self.nsrcs_per_sample, -1).astype('int')
        tmp_arr = np.sum(tmp_arr, axis=0)
        self.ideal_lens_bool_dfl = (tmp_arr == target_nsrc) #if tmp_arr > 1, multiple source lenses; shape: [n_samples,]

        #reshape the source info array, from [N_src_per_sample*n_lens,] to [N_src_per_sample, n_samples]
        self.thetaE_arr = self.thetaE_arr.reshape(self.nsrcs_per_sample,-1) #shape: [N_src_per_sample, n_samples]
        self.einstein_mass_arr = self.einstein_mass_arr.reshape(self.nsrcs_per_sample,-1)
        self.src_pop.zs_arr = self.src_pop.zs_arr.reshape(self.nsrcs_per_sample,-1) #shape: [N_src_per_sample, n_samples]
        self.src_pop.xs_arr = self.src_pop.xs_arr.reshape(self.nsrcs_per_sample,-1)
        self.src_pop.ys_arr = self.src_pop.ys_arr.reshape(self.nsrcs_per_sample,-1)
        self.src_pop.pa_arr = self.src_pop.pa_arr.reshape(self.nsrcs_per_sample,-1)
        self.src_pop.q_arr = self.src_pop.q_arr.reshape(self.nsrcs_per_sample,-1)
        self.src_pop.Re_arr = self.src_pop.Re_arr.reshape(self.nsrcs_per_sample,-1)
        for key in self.src_pop.app_mag_arr.keys():
            self.src_pop.app_mag_arr[key] = self.src_pop.app_mag_arr[key].reshape(self.nsrcs_per_sample,-1)
        self.ideal_lens_bool_src = self.ideal_lens_bool_src.reshape(self.nsrcs_per_sample,-1) #shape: [N_src_per_sample, n_samples]

        #remove the source that is not lensed by any deflector
        tmp_bool = self.ideal_lens_bool_src[:, self.ideal_lens_bool_dfl] #shape: [N_src_per_sample, n_lens]
        self.thetaE_arr = self.thetaE_arr[:, self.ideal_lens_bool_dfl] #shape: [N_src_per_sample, n_lens]
        self.thetaE_arr = self.thetaE_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T #shape: [target_nsrc, n_lens]
        self.einstein_mass_arr = self.einstein_mass_arr[:, self.ideal_lens_bool_dfl]
        self.einstein_mass_arr = self.einstein_mass_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.zs_arr = self.src_pop.zs_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.zs_arr = self.src_pop.zs_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.xs_arr = self.src_pop.xs_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.xs_arr = self.src_pop.xs_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.ys_arr = self.src_pop.ys_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.ys_arr = self.src_pop.ys_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.pa_arr = self.src_pop.pa_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.pa_arr = self.src_pop.pa_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.q_arr = self.src_pop.q_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.q_arr = self.src_pop.q_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.src_pop.Re_arr = self.src_pop.Re_arr[:, self.ideal_lens_bool_dfl]
        self.src_pop.Re_arr = self.src_pop.Re_arr.T[tmp_bool.T].reshape(-1, target_nsrc).T
        for key in self.src_pop.app_mag_arr.keys():
            self.src_pop.app_mag_arr[key] = self.src_pop.app_mag_arr[key][:, self.ideal_lens_bool_dfl]
            self.src_pop.app_mag_arr[key] = self.src_pop.app_mag_arr[key].T[tmp_bool.T].reshape(-1, target_nsrc).T
        self.ideal_lens_bool_src = self.ideal_lens_bool_src[:, self.ideal_lens_bool_dfl]
        self.ideal_lens_bool_src = self.ideal_lens_bool_src.T[tmp_bool.T].reshape(-1, target_nsrc).T #shape: [target_nsrc, n_lens]

        # count the number of ideal lens in each vdisp bins
        ideal_lens_vdisp_arr = self.dfl_pop.vdisp_arr[self.ideal_lens_bool_dfl] #shape: [n_lens,]
        self.n_ideal_lens_arr = np.histogram(ideal_lens_vdisp_arr, bins=self.vdisp_bin_edges)[0] #shape: (n_vdisp_bins,)
    

    def check_if_lensing(self, xs_arr, ys_arr, thetaE_arr):
        """
        Check if the lens candidate is real

        :param xs_arr: the x coordinate of source location
        :param ys_arr: the y coordinate of source location

        :return : a bool array shows the result of this checking
        """
        rs_arr = np.sqrt(xs_arr**2 + ys_arr**2)
        return (thetaE_arr>rs_arr)


    def save_ideal_lens_samples(
        self, 
        output_path='./',
        output_file='ideal_lens_samples.hdf5',
    ):
        """
        Save the lens sample parameters into a hdf5 file
        """
        try:
            abs_path = os.path.abspath(output_path)
            os.makedirs(abs_path)
        except IOError:
            pass
        
        dest_file = os.path.join(output_path, output_file)
        fn = h5py.File(dest_file, "w")

        #save deflector info
        SLU.save_hdf5_overwrite(
            fn,
            "deflector/z", 
            data=self.dfl_pop.zl_arr[self.ideal_lens_bool_dfl], #shape: [n_lens, ]
        )
        SLU.save_hdf5_overwrite(
            fn,
            "deflector/q", 
            data=self.dfl_pop.q_arr[self.ideal_lens_bool_dfl], #shape: [n_lens, ]
        )
        SLU.save_hdf5_overwrite(
            fn,
            "deflector/vdisp", 
            data=self.dfl_pop.vdisp_arr[self.ideal_lens_bool_dfl], #shape: [n_lens, ]
        )
        SLU.save_hdf5_overwrite(
            fn,
            "deflector/rband_abs_mag", 
            data=self.dfl_pop.abs_mag_arr[self.ideal_lens_bool_dfl], #shape: [n_lens, ]
        )          
        SLU.save_hdf5_overwrite(
            fn,
            "deflector/Re", 
            data=self.dfl_pop.Re_arr[self.ideal_lens_bool_dfl], #shape: [n_lens, ]
        )
        for key in self.dfl_pop.app_mag_arr.keys():
            SLU.save_hdf5_overwrite(
                fn,
                f"deflector/app_mag_{key}", 
                data=self.dfl_pop.app_mag_arr[key][self.ideal_lens_bool_dfl], #shape: [n_lens, ]
            )

        #save infomation related to the source
        SLU.save_hdf5_overwrite(
            fn,
            "source/thetaE", 
            data=self.thetaE_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/einstein_mass", 
            data=self.einstein_mass_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/z", 
            data=self.src_pop.zs_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/xs", 
            data=self.src_pop.xs_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/ys", 
            data=self.src_pop.ys_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/pa", 
            data=self.src_pop.pa_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/q", 
            data=self.src_pop.q_arr, #shape: [target_nsrc, n_lens]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "source/Re", 
            data=self.src_pop.Re_arr, #shape: [target_nsrc, n_lens]
        )

        for key in self.src_pop.app_mag_arr.keys():
            SLU.save_hdf5_overwrite(
                fn,
                f"source/app_mag_{key}", 
                data=self.src_pop.app_mag_arr[key], #shape: [target_nsrc, n_lens]
            )

        SLU.save_hdf5_overwrite(
            fn,
            "n_etg_arr", 
            data=self.n_etg_arr, #shape: [n_vdisp_bins, ]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "n_ideal_lens_arr", 
            data=self.n_ideal_lens_arr, #shape: [n_vdisp_bins, ]
        )

        SLU.save_hdf5_overwrite(
            fn,
            "vdisp_bin_edges", 
            data=self.vdisp_bin_edges, #shape: [n_vdisp_bins, ]
        )

        fn.close()



