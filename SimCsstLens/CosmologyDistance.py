import numpy as np
from astropy.cosmology import LambdaCDM

class CosmosDist(object):
    def __init__(self, Om0=0.3, Ode0=0.7, h=0.7):
        # https://docs.astropy.org/en/stable/api/astropy.cosmology.LambdaCDM.html#astropy.cosmology.LambdaCDM
        self.cosmos = LambdaCDM(H0=h*100, Om0=Om0, Ode0=Ode0)


    def comoving_distance(self, z1=0.0, z2=None):
        #in Mpc unit
        if z1 != 0.0:
            return self.cosmos.comoving_distance(z2).value - self.cosmos.comoving_distance(z1).value
        else:
            return self.cosmos.comoving_distance(z2).value


    def comoving_transverse_distance(self, z1=0.0, z2=None):
        if z1 != 0.0:
            return self.cosmos.comoving_transverse_distance(z2).value - self.cosmos.comoving_transverse_distance(z1).value
        else:
            return self.cosmos.comoving_transverse_distance(z2).value


    def angular_diameter_distance(self, z1=0.0, z2=None):
        #in Mpc unit
        return self.cosmos.angular_diameter_distance_z1z2(z1, z2).value


    def luminosity_distance(self, z):
        return self.cosmos.luminosity_distance(z).value


    def comoving_volume(self, z):
        #see eq-28 in Hogg1999, in Mpc^3 unit
        return self.cosmos.comoving_volume(z).value


    def differential_comoving_volume(self, z):
        ##Hogg, eq-27, in Mpc^3 unit
        return self.cosmos.differential_comoving_volume(z).value


    def distance_modulus(self, z):
        #Hogg, eq-24
        return  self.cosmos.distmod(z).value


    def age(self, z):
        #age of universe at redshit z, in Gyr unit
        return self.cosmos.age(z).value
    

    def lookback_time(self, z):
        #lookback time at redshit z, in Gyr unit
        return self.cosmos.lookback_time(z).value