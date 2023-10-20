import numpy as np
from SimCsstLens.SourcePop import Util as SSU
from SimCsstLens.DeflectorPop import Util as SDU

class SourcePopulation(object):
    def __init__(
        self, 
        catalog_type='lsst', 
        bands=['g', 'r', 'i', 'z'],
        cosmo_dist=None
    ):
        self.bands = bands
        self.cosmos_dist = cosmo_dist
        self.catalog_type = catalog_type

        self.load_catalog()


    def load_catalog(self):
        if self.catalog_type == 'lsst':
            self.zs_cat, self.app_mag_arr_cat, self.abs_mag_r_cat = SSU.load_lsst()
        elif self.catalog_type == 'CosmosDC2':
            self.zs_cat, self.app_mag_arr_cat, self.abs_mag_r_cat = SSU.load_CosmosDC2()
        else:
            print('no such source catalog exists')


    def draw_source_samples(self, nsamples=100, nsrcs_per_sample=1, over_density=1):        
        #radomly draw sources from the catalog
        nsources = nsamples * nsrcs_per_sample
        source_index=np.random.randint(0, len(self.zs_cat), nsources) #repetitive random index
        self.zs_arr = self.zs_cat[source_index] #randomly draw source redshift
        self.abs_mag_r_arr = self.abs_mag_r_cat[source_index] #r-band absolute AB magnitude

        self.app_mag_arr={} #{'band-1': ab_apprent_mag_arr1, 'band-2': ab_apprent_mag_arr2, ...}
        for band in self.bands:
            if band != "VIS":
                self.app_mag_arr[band]=self.app_mag_arr_cat[band][source_index]
            else:
                self.app_mag_arr[band]=(
                    self.app_mag_arr_cat["r"][source_index]+ \
                    self.app_mag_arr_cat["i"][source_index]+ \
                    self.app_mag_arr_cat["z"][source_index] \
                )/3.

        Re_phys_arr = SSU.source_size_from(self.abs_mag_r_arr, self.zs_arr, scatter=True) #source physial size in kpc unit
        self.Re_arr = Re_phys_arr / self.cosmos_dist.angular_diameter_distance(z1=0.0, z2=self.zs_arr) / 1e3 #source size in radian unit
        self.Re_arr *= SDU.RADIAN_TO_ARCSEC #source size in arcsec unit

        self.q_arr = SSU.draw_axis_ratio(nsources) #axis-ratio of source
        
        self.pa_arr = np.random.random_sample(nsources)*180.0 #source position angle in degree unit
        
        
        if self.catalog_type == "lsst": #lsst sim has a source number density of ~0.06 per square arcsecond
            src_number_density = 0.06 #per arcsec^2
            src_number_density *= over_density
            box_size = np.sqrt(nsrcs_per_sample/src_number_density)
            #box_size: the average angular size (in arcsec) for which we can detect `nsrcs_per_sample` source.
            #the `over_density`` represents a `factor``, 
            #that describe how a local region's source density, exceed the average value across the whole sky
        elif self.catalog_type == "cosmos": #cosmos has a source number density of ~0.015 per square arcsecond
            src_number_density = 0.015 #per arcsec^2
            src_number_density *= over_density
            box_size = np.sqrt(nsrcs_per_sample/src_number_density)
        else:
            print('no such source catalog exists')

        self.xs_arr = (np.random.random_sample(nsources)-0.5)*box_size #for "lsst", box_size=4.08 arcsec
        self.ys_arr = (np.random.random_sample(nsources)-0.5)*box_size 