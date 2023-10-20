import numpy as np
from SimCsstLens.SimLensImage import Util
import scipy.special as sf


class SersicLight(object):
    def __init__(
        self,
        xc=None, 
        yc=None, 
        q=None, 
        PA=None, 
        Re=None, 
        Ie=None, 
        n=None,
        m=None,
        mag_zero=22,
    ):
        self.xc = xc
        self.yc = yc
        self.q = q
        self.PA = PA
        self.Re = Re
        self.Ie = Ie #Note, we set the unit of Ie to counts/s/arcsec^2, not the counts/s/pixel.
        self.n = n
        self.m = m #apprent magnitude of source
        self.m0 = mag_zero

        if self.Ie is None:
            if self.m is not None:
                self.Ie = self.Ie_from_magnitude(
                    self.m, 
                    self.m0,
                    Re=self.Re,
                    n=self.n, 
                )
                self.total_flux = Util.mag2cps(self.m, self.m0) #unit: counts/s
            else:
                raise Exception('please input either the intensity at effective radius, or magnitude')
        else:
            self.total_flux = self.total_flux_analytic_from(
                Re=self.Re, 
                Ie=self.Ie, 
                n=self.n,
            ) #unit: counts/s


    @staticmethod
    def sersic_light(xgrid, ygrid, xc=None, yc=None, q=None, PA=None, Re=None, Ie=None, n=None):
        xgrid_new, ygrid_new = Util.xy_transform(xgrid, ygrid, xc, yc, PA)
        k = sf.gammaincinv(2.0*n, 0.5) #use scipy built-in directly
        r = np.sqrt(q*xgrid_new**2.+ygrid_new**2./q)
        return Ie*np.exp(-k* ((r/Re)**(1./n)-1))


    @staticmethod
    def total_flux_from(
        xc=None,
        yc=None,
        q=None,
        PA=None,
        Re=None,
        Ie=None,
        n=None, 
        npix=100,
        nsub=4,
    ):
        dpix = 20.0*Re/npix #in arcsec unit
        xgrid, ygrid = Util.make_grid_2d(npix, dpix, nsub)
        xgrid += xc
        ygrid += yc
        
        image = SersicLight.sersic_light(
            xgrid, 
            ygrid,
            xc=xc, 
            yc=yc, 
            q=q, 
            PA=PA, 
            Re=Re, 
            Ie=Ie, 
            n=n,
        )
        image = Util.bin_image(image, nsub) #unit: count/s/arcsec^2
        return np.sum(image)*dpix**2 #brightness to flux, unit: counts/s;


    @staticmethod
    def total_flux_analytic_from(
        Re=None,
        Ie=None,
        n=None, 
    ):
        #result is almost the same as the `total_flux_from`, but faster speed
        k = sf.gammaincinv(2.0*n, 0.5)
        factor = k**(2.0*n) / (2*np.pi*Re**2*np.exp(k)*n*sf.gamma(2.0*n))
        return Ie/factor


    @staticmethod
    def Ie_from_magnitude(
        magnitude, 
        magnitude_zero_point=22.0,
        Re=None,
        n=None, 
    ):
        total_cps_tmp = SersicLight.total_flux_analytic_from(Re=Re, Ie=1.0, n=n)
        total_cps = Util.mag2cps(magnitude, magnitude_zero_point)
        rescale_factor = total_cps/total_cps_tmp
        return rescale_factor

        
    def brightness_at(self, xgrid, ygrid):
        return self.sersic_light(
            xgrid, 
            ygrid,
            self.xc,
            self.yc,
            self.q,
            self.PA,
            self.Re,
            self.Ie,
            self.n,
        )



class SieMass():
    def __init__(
        self,
        xc=None, 
        yc=None, 
        q=None, 
        PA=None, 
        thetaE=None,
    ):
        self.xc = xc
        self.yc = yc
        self.q = q
        self.PA = PA
        self.thetaE = thetaE


    @staticmethod
    def sie_deflection(xgrid, ygrid, xc=None, yc=None, q=None, PA=None, thetaE=None):
        PA = PA - 90.0 

        (xgrid_new, ygrid_new) = Util.xy_transform(xgrid, ygrid, xc, yc, PA)
        #deal with the numerical odd point, i.e, (0, 0)
        xgrid_new, ygrid_new, rgrid_new = Util.relocate_radii(xgrid_new, ygrid_new, eps=1e-8)

        qfact=np.sqrt(1.0/q-q)
        if np.abs(qfact) <= 1e-8: #degrade to SIS
            alpha_x = xgrid_new/rgrid_new * thetaE
            alpha_y = ygrid_new/rgrid_new * thetaE
        else:
            alpha_x = np.arcsinh(np.sqrt(1.0/q**2.0-1.0)*xgrid_new/rgrid_new)/qfact * thetaE
            alpha_y = np.arcsin(np.sqrt(1.0-q**2.0)*ygrid_new/rgrid_new)/qfact * thetaE
        return Util.xy_transform(alpha_x, alpha_y, 0.0, 0.0, -1.0*PA)


    def deflection_at(self, xgrid, ygrid):
        return self.sie_deflection(
            xgrid, 
            ygrid,
            self.xc,
            self.yc,
            self.q,
            self.PA,
            self.thetaE,
        )        



class EplMass():
    def __init__(
        self,
        xc=None, 
        yc=None, 
        q=None, 
        PA=None, 
        thetaE=None,
        slope=None,
    ):
        self.xc = xc
        self.yc = yc
        self.q = q
        self.PA = PA
        self.thetaE = thetaE
        self.slope = slope


    @staticmethod
    def epl_deflection(xgrid, ygrid, xc=None, yc=None, q=None, PA=None, thetaE=None, slope=None):
        """
        Calculating the deflection angle of an SPLE mass profile 
        following Tessore & Metcalf 2015 (arXiv:1507.01819)
        The convergence has the form of kappa(x, y)=0.5*(2-t)*(b/sqrt(q^2*x^2+y^2))^t
        In this form, b/sqrt(q) is the Einstein radius in the intermediate-axis convention
        """

        (xgrid_new, ygrid_new) = Util.xy_transform(xgrid, ygrid, xc, yc, PA)
        xgrid_new, ygrid_new, _ = Util.relocate_radii(xgrid_new, ygrid_new, eps=1e-8) #handle the numerical odd point

        slope = slope - 1 #3d -> 2d slope; we suppose the input slope is the 3d density slope.
        f=(1.0-q)/(1.0+q)
        thetaE_tessore = thetaE*np.sqrt(q) #Einstein radius defined in the intermediate-axis convention -> the definition of Tessore15 paper.

        phi=np.arctan2(ygrid_new, q*xgrid_new) #eq.4
        R=np.sqrt(q**2.*xgrid_new**2.+ygrid_new**2.) #eq.3
        z=np.cos(phi)*(1+0j)+np.sin(phi)*(0+1j) #Euler's formula
        tmp=2.0*thetaE_tessore/(1.0+q)*(thetaE_tessore/R)**(slope-1.)*z*sf.hyp2f1(1., 0.5*slope, 2.-0.5*slope, -f*z**2.) #eq.13
        alpha_x=tmp.real
        alpha_y=tmp.imag

        return Util.xy_transform(alpha_x, alpha_y, 0.0, 0.0, -PA)


    def deflection_at(self, xgrid, ygrid):
        return self.epl_deflection(
            xgrid, 
            ygrid,
            self.xc,
            self.yc,
            self.q,
            self.PA,
            self.thetaE,
            self.slope,
        )        



class ShearMass(object):
    def __init__(self, shear_strength, shear_angle):
        self.shear_strength = shear_strength
        self.shear_angle = shear_angle


    @staticmethod
    def shear_deflection(xgrid, ygrid, shear_strength=None, shear_angle=None):    
        shear_angle = np.deg2rad(shear_angle)

        phicoord=np.arctan2(xgrid, ygrid)  #Angle in radian unit
        rcoord=np.sqrt(xgrid**2.+ygrid**2.)
        # dpsi/dr
        dpsi_1=-rcoord*shear_strength*np.cos(2.*(phicoord-shear_angle))
        # 1/r * dpis/dphi
        dpsi_2=rcoord*shear_strength*np.sin(2.*(phicoord-shear_angle))

        alpha_x = dpsi_1*np.cos(phicoord)-dpsi_2*np.sin(phicoord)
        alpha_y = dpsi_1*np.sin(phicoord)+dpsi_2*np.cos(phicoord)
        return alpha_x, alpha_y

    
    def deflection_at(self, xgrid, ygrid):
        return self.shear_deflection(
            xgrid, 
            ygrid,
            self.shear_strength,
            self.shear_angle,
        )