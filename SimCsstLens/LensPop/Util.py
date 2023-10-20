import numpy as np 
from astropy import constants as const
from astropy import units 


def vdisp_to_thetaE(z_l, z_s, vdisp, cosmos=None):
    '''
    calculate einstein radius from velocity dispersion, assume SIE model
    vdisp is velocity dispersion in km/s
    return the einstein radius in arcsec unit
    '''
    #thetaE = 4*pi * sigma^2/c^2 * D_ls/D_s
    c = const.c.value / 1000 #to km/s 
    v_fac = (vdisp/c)**2
    d_fac = cosmos.angular_diameter_distance(z_l, z_s)/cosmos.angular_diameter_distance(0.0, z_s)
    radian_to_arcsec = 180.0/np.pi*60*60
    return 4.0*np.pi*v_fac*d_fac*radian_to_arcsec


def thetaE_to_vdisp(z_l, z_s, thetaE, cosmos=None):
    #thetaE = 4*pi * sigma^2/c^2 * D_ls/D_s
    arcsec_to_radian = np.pi/(180.0*60*60)
    thetaE *= arcsec_to_radian

    d_fac = cosmos.angular_diameter_distance(z_l, z_s)/cosmos.angular_diameter_distance(0.0, z_s)
    c = const.c.value / 1000 #to km/s 

    vdisp = np.sqrt(thetaE/d_fac/(4*np.pi))*c
    return vdisp


def return_critical_density(zl=0.2,zs=0.7, astropy_cosmos=None):
    v_light = const.c
    grav_const = const.G

    Ds = astropy_cosmos.angular_diameter_distance(zs)
    Dd = astropy_cosmos.angular_diameter_distance(zl)
    Dds = astropy_cosmos.angular_diameter_distance_z1z2(zl,zs)

    crit_surface_density = v_light**2 / \
                    (4*np.pi*grav_const) * \
                    Ds/Dd/Dds
    output_unit = units.kg / (units.m*units.m)
    return crit_surface_density.to(output_unit).value  #critical density in SI units


def einstein_mass_from(thetaE, zl=0.2,zs=0.7, astropy_cosmos=None):
    """
    thetaE is einstein radius in arcsec unit
    return einstein mass in solar mass
    """
    Mpc = 1e6*const.pc.value # 1 Mpc, in m
    Msun = const.M_sun.value # solar mass, in kg
    arcsec2rad = (np.pi/(180.*3600.))

    Dd = astropy_cosmos.angular_diameter_distance(zl).value

    factor = arcsec2rad*Dd*Mpc
    angular_area = np.pi * thetaE**2
    physical_area = angular_area * factor**2 

    crit_density = return_critical_density(zl,zs, astropy_cosmos=astropy_cosmos)

    return physical_area * crit_density / Msun  #in solar mass unit


def einstein_radius_from(einstein_mass, zl=0.2, zs=0.7, astropy_cosmos=None):
    """
    einstein mass in solar mass
    return thetaE, which is einstein radius in arcsec unit
    """
    Mpc = 1e6*const.pc.value # 1 Mpc, in m
    Msun = const.M_sun.value # solar mass, in kg
    arcsec2rad = (np.pi/(180.*3600.))

    Dd = astropy_cosmos.angular_diameter_distance(zl).value
    factor = arcsec2rad*Dd*Mpc

    einstein_mass *= Msun #in kg unit
    crit_density = return_critical_density(zl,zs, astropy_cosmos=astropy_cosmos)
    physical_area = einstein_mass/crit_density
    angular_area = physical_area/factor**2
    thetaE = np.sqrt(angular_area/np.pi)

    return thetaE


def save_hdf5_overwrite(
    hdf5_file=None, 
    data_path='dataset_full_path',
    data=np.array([1,2,3]), 
):
    if data_path in hdf5_file:
        del hdf5_file[data_path]
    hdf5_file.create_dataset(data_path, data=data)