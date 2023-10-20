import numpy as np
from scipy.special import gamma
import os
from scipy.interpolate import splev,splint,splrep

current_dir, current_file = os.path.split(os.path.abspath(__file__))

#define a few constant
C = 299792458 #speed of light in m/s, note seems LensPop use `299792452`?
RADIAN_TO_ARCSEC = 180.0/np.pi * 3600
ARCSEC_TO_RADIAN = 1.0/RADIAN_TO_ARCSEC


def ETGs_vdisp_pdf(vdisp, h=0.7):
    """
    The empirical velocity disperison probability function of early type galaxies, based on
    the SDSS observation. Choi et al. 2007; see also Collet 2015 eq3; 
    code from LensPop, https://github.com/tcollett/LensPop 

    :param vdisp: the velocity dispersion of early-type galaxies in km/s
    :param h: the hubble constant values (at now) in 100km/s/Mpc unit

    :retrun: the (comoving) number dneisty of ETGs.
    """
    vdisp[vdisp==0]+=1e-6 
    phi_star=(8*10**-3)*h**3
    alpha=2.32
    beta=2.67
    sigst=161
    phi=phi_star * \
        ((vdisp*1./sigst)**alpha)*\
        np.exp(-(vdisp*1./sigst)**beta)*beta/\
        gamma(alpha*1./beta)/\
        (1.*vdisp)
    #phi*=(1+z)**(-2.5) #no redshift dependance
    return phi


def load_ETGs_color_spline(redshift_bins=np.linspace(0, 2.0, 201), bands=['g', 'r', 'i', 'z']):
    """
    We suppose the color of ETGs follows the spectral energy distribution of a passive galaxy 
    whose star formation was a single burst 10 Gyrs ago. see Collet2015
    """
    #load sed table; 
    #sed_table[:,0] wavelength in Angstrom; sed_table[:,1] emission energy at that wavelengths
    sed_table = np.loadtxt(f'{current_dir}/data/sed/BC_Z=1.0_age=10.00gyr.sed')

    #load SDSS r-band filter as the base.
    #filter_r[:, 0]: wavelength in Angstrom
    #filter_r[:, 1]: troughout 
    #see https://mfouesneau.github.io/docs/pyphot/photometry.html?highlight=throughput
    filter_r = np.loadtxt(f'{current_dir}/data/filter/r_SDSS.res')
    filter_r = splrep(filter_r[:,0], filter_r[:,1], k=1,s=0)

    color_splines = {}
    for band in bands:
        if band == "VIS": continue  #ignore VIS band, since its photometry can be approximated as the mean of r,i,z bands
        filter_this = np.loadtxt(f'{current_dir}/data/filter/{band}_SDSS.res')
        filter_this = splrep(filter_this[:,0], filter_this[:,1], k=1,s=0)

        colors_this = np.zeros_like(redshift_bins)
        for i in range(len(redshift_bins)):
            #we built a color table (spline) here, this table allow us to
            #give the magnitude of ETGs given redshift and band, using the 
            #r-band magnitude at redshift 0 as the baseline.
            colors_this[i] = - (ABFilterMagnitude(filter_this, sed_table, redshift_bins[i]) - ABFilterMagnitude(filter_r, sed_table, 0)) 
        color_splines[band] = splrep(redshift_bins, colors_this)

    return color_splines
    

def ABFilterMagnitude(filter, spectrum, redshift):
    """
    Determines the AB magnitude (up to a constant) given an input filter, SED,
        and redshift.
    filter: a b-spline object represent the filter wavelength-troughout relation
    spectrum: the template SED; shape: [n_wavelength, 2]
    redshift: a number
    """
    sol = 299792452. #speed of light in m/s

    wave = spectrum[:, 0].copy() #wavelength! unit, Angstrom
    data = spectrum[:, 1].copy() #spectral energy, unit??????

    # Redshift the spectrum and determine the valid range of wavelengths
    wave *= (1.+redshift) #redshift the template SED wavelength
    wmin,wmax = filter[0][0],filter[0][-1] #unit: Angstrom
    cond = (wave>=wmin)&(wave<=wmax) #can do comparison, so the unit of wave is also Angstrom!!!
    wave = wave[cond]

    #redshift data
    data = data[cond]/(1.+redshift)

    # Evaluate the filter at the wavelengths of the spectrum
    response = splev(wave,filter) #evaluate the value of troughout (response) at given wavelengths (according to filter).

    #To understand the below code, please check https://mfouesneau.github.io/pyphot/photometry.html
    #determin the pivot wavelength
    spline_1 = splrep(wave,response*wave,s=0,k=1)
    factor_1 = splint(wave[0],wave[-1],spline_1)
    spline_2 = splrep(wave,response/wave,s=0,k=1)
    factor_2 = splint(wave[0],wave[-1],spline_2)
    lambda_p2 = factor_1/factor_2

    #determine mean f_lambda
    spline_1 = splrep(wave,response*wave*data,s=0,k=1)
    factor_1 = splint(wave[0],wave[-1],spline_1)
    spline_2 = splrep(wave,response*wave,s=0,k=1)
    factor_2 = splint(wave[0],wave[-1],spline_2)
    mean_f_lambda = factor_1/factor_2
    
    #get mean of f_mu
    mean_f_mu = lambda_p2/(sol*1e10) * mean_f_lambda

    return -2.5*np.log10(mean_f_mu) - 48.6


def EarlyTypeRelations(vdisp, scatter=True):#z dependence not encoded currently
    """
    We use this early-type galaxy relation to draw the r-band absolute magnitude
    and effective radius from the velocity dispersion. Below code are from LensPop code directly
    https://github.com/tcollett/LensPop; More information can refer to Collet15

    :param vdisp: the velocity dispersion of ETGs in km/s 
    :param scatter: option to control wheter add stochastics to the Early Type Relations

    :retrun Mr: the r band absolute magnitude of ETGs
    :retrun r_phys: the r band effetive radius of ETGs in kpc unit
    """
    #Hyde and Bernardi, Mr -> r band absolute magnitude.
    V=np.log10(vdisp)
    Mr=(-0.37+(0.37**2-(4*(0.006)*(2.97+V)))**0.5)/(2*0.006)
    R=2.46-2.79*V+0.84*V**2

    if scatter:
        Mr+=np.random.randn(len(Mr))*(0.15/2.4)
        R+=np.random.randn(len(R))*0.11

    #convert to observed r band size;
    r_phys = 10**R #lens effective radius in unit of kpc
    return Mr, r_phys