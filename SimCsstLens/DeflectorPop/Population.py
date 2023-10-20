import numpy as np
from SimCsstLens.DeflectorPop import Util as SDU
from scipy import  interpolate

class DeflectorPopulation(object):

    def __init__(
        self, 
        vdisp_floor=100, 
        zl_max=2.0, 
        cosmo_dist=None, 
        bands=['g', 'r', 'i', 'z']
    ) -> None:
        """
        vdisp_floor: the lower limit of the velocity dispersion of ETGs, that could potentially acted as a lens.
        zl_max: the maximum redshift of the potential lens
        cosmos_dict: a ComologyDistance object, see CosmologyDistance.py
        """
        self.vdisp_floor = vdisp_floor
        self.zl_max = zl_max
        self.cosmos_dist = cosmo_dist
        self.bands = bands

        self.color_splines = SDU.load_ETGs_color_spline(
            redshift_bins=np.linspace(0, self.zl_max, int(self.zl_max/0.01)+1), 
            bands=self.bands,
        )


    def vdisp_pdf(self, vdisp):
        """
        The velocity dipsersion function of ETGs

        :param vdisp: the velocity dispersion of ETGs in km/s 
        """
        return SDU.ETGs_vdisp_pdf(vdisp, self.cosmos_dist.cosmos.h)


    def build_ETGs_cdf_splines(self, z_max=2.0):
        """
        Build the cumulative Density Function (CDF) splines from 
        a given probablity density function (PDF), 
        to draw new samples using the inverse cdf method.
        """
        vdisp_bins = np.linspace(self.vdisp_floor,400,401)
        Dphi_Dvdisp = self.vdisp_pdf(vdisp_bins)
        self.inv_cdf_vdisp_spline = interpolate.splrep(
            np.cumsum(Dphi_Dvdisp)/np.sum(Dphi_Dvdisp), 
            vdisp_bins,
        )
        self.Dphi_Dvdisp_spline = interpolate.splrep(
            vdisp_bins,
            Dphi_Dvdisp
        )

        #number density of ETGs, marginalized the velocity dispersion
        Dphi = interpolate.splint(
            self.vdisp_floor, 
            400, 
            self.Dphi_Dvdisp_spline,
        )
        z_bins = np.linspace(0, z_max, int(z_max/0.01)+1)
        Dn_lens_per_Dz_Dsr = Dphi*self.cosmos_dist.differential_comoving_volume(z_bins) #per solid angle (steradian)
        self.inv_cdf_z_spline = interpolate.splrep(
            np.cumsum(Dn_lens_per_Dz_Dsr)/np.sum(Dn_lens_per_Dz_Dsr),
            z_bins,
        )
        self.DnDzDsr_spline = interpolate.splrep(
            z_bins,
            Dn_lens_per_Dz_Dsr,
        )


    def draw_redshift(self, nsamples=100):
        return interpolate.splev(np.random.random(nsamples), self.inv_cdf_z_spline)


    def draw_velocity_dispersion(self, nsamples=100):
        return interpolate.splev(np.random.random(nsamples), self.inv_cdf_vdisp_spline)


    def r_band_mag_and_eff_radius_from(self, vdisp):
        return SDU.EarlyTypeRelations(vdisp, scatter=True)


    def draw_axis_ratio(self, vdisp):
        x=vdisp #velocity dispersion
        y=0.378-0.000572*x #s in eq-4 in collet15
        e=np.random.rayleigh(y)
        q=1-e
        #remove samples with q<0.2, keep sampleling until all q>0.2 (and <1)
        while len(q[q<0.2])>0 or len(q[q>1])>0:
            q[q<0.2]=1-np.random.rayleigh(y[q<0.2]) #re-draw the q sample, if q<0.2
            q[q>1]=1-np.random.rayleigh(y[q>1]) #re-draw the q sample, if q>1
        return q


    def draw_colors(self,z,band):
        return interpolate.splev(z, self.color_splines[band])


    def draw_apparent_magnitude(self, abs_mag_r, z, band='g'): #absolute mag to apparent mag
        if band is None:
            colors = np.zeros_like(z, dtype='float')
        else:
            colors = self.draw_colors(z, band)

        Dmods=self.cosmos_dist.distance_modulus(z) #distance module
        m = abs_mag_r - colors + Dmods #the colors terms include both the effects of `rest-frame color` and `k-correction`
        return m #m is the apprent magnitude given the `band`


    def draw_deflector_samples(
        self, 
        nsamples=100,
    ):
        if not hasattr(self, 'inv_cdf_vdisp_spline'):
            self.build_ETGs_cdf_splines(z_max=self.zl_max)

        self.zl_arr = self.draw_redshift(nsamples)
        self.vdisp_arr = self.draw_velocity_dispersion(nsamples)

        self.abs_mag_arr, Re_phys_arr = self.r_band_mag_and_eff_radius_from(self.vdisp_arr)
        self.Re_arr = Re_phys_arr / self.cosmos_dist.angular_diameter_distance(z1=0.0, z2=self.zl_arr) / 1e3
        self.Re_arr *= SDU.RADIAN_TO_ARCSEC

        self.app_mag_arr = {} #dict which save the apprent magnitude of each band
        for band in self.bands:
            self.app_mag_arr[band] = self.draw_apparent_magnitude(
                self.abs_mag_arr, 
                self.zl_arr,
                band=band,
            )

        self.q_arr = self.draw_axis_ratio(self.vdisp_arr)


    def number_of_etgs(self, sky_frac=1.0):
        """
        Calculate the number of the potential ETG deflectors, given the faction of sky that would be observed (sky_frac).
        """
        if not hasattr(self, 'DnDzDsr_spline'):
            self.build_ETGs_cdf_splines(z_max=self.zl_max)
        sky_sr = np.pi*4 * sky_frac #the observed sky area in sr
        n_dfl_per_sr = interpolate.splint(0.0, self.zl_max, self.DnDzDsr_spline) #number of deflectors between 0 and z
        return n_dfl_per_sr * sky_sr

