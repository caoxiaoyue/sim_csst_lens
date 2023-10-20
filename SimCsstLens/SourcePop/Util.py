import numpy as np
import os

current_dir, current_file = os.path.split(os.path.abspath(__file__))

def load_lsst():
    data = np.load(f'{current_dir}/data/lsst_1sqdegree_catalog.npy')
    app_mag_dict={} #a dict which save the magnitude info of the source galaxy
    app_mag_dict["i"] = data[:,5]
    mask = (app_mag_dict["i"] < 28)
    app_mag_dict["i"] = app_mag_dict["i"][mask]

    zs = data[:,2][mask] #source redshift
    abs_mag_r = data[:,7][mask] #absmag_r_total in `lsst.1sqdegree_catalog2.txt`; r-band absolute AB magnitude

    app_mag_dict["g"] = data[:,3][mask] #g-band apparent AB-magnitude
    app_mag_dict["r"] = data[:,4][mask]
    app_mag_dict["z"] = data[:,6][mask]
    
    app_mag_dict["VIS"] = (app_mag_dict["r"] + app_mag_dict["i"] + app_mag_dict["z"]) / 3.0

    return zs, app_mag_dict, abs_mag_r


def load_CosmosDC2():
    #the cosmosDC2 catalog cut the source redshift to 3.0
    #which may be problematic for lens prediction!
    #since a ton of lenses with source redshift greater than 3!
    pass


def source_size_from(abs_mag=None, z=None, scatter=True):
    #{mosleh et al}, {Huang, Ferguson et al.}, Newton SLACS XI.
    #abs_mag: r-band AB absolute magnitude; z: redshift
    R=-(abs_mag+18.)/4.
    r_phys=(10**R)*((1.+z)/1.6)**(-1.2)
    if scatter:
        r_phys = r_phys * 10**(np.random.randn(len(r_phys))*0.35) #0.35 dex scattering
    return r_phys #physical source size in unit of kpc


def draw_axis_ratio(nsample=100):
    y=np.ones(nsample)*0.3 #scale parameter s=0.3, see eq-4 in collet15
    q=1-np.random.rayleigh(y)
    #remove samples with q<0.2, keep sampleling until all q>0.2 (and <1)
    while len(q[q<0.2])>0 or len(q[q>1])>0:
        q[q<0.2]=1-np.random.rayleigh(y[q<0.2]) #re-draw the q sample, if q<0.2
        q[q>1]=1-np.random.rayleigh(y[q>1]) #re-draw the q sample, if q>1; 
        # TODO this one can be refactor, since rayleigh dsitribution always >0
    return q