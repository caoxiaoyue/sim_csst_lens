import numpy as np
from matplotlib import path


def make_grid_2d(npix, dpix, nsub=1):
    npix_sub = npix*nsub
    dpix_sub = dpix/float(nsub)

    grid = np.arange(npix_sub)*dpix_sub
    grid_center = np.mean(grid)
    grid -= grid_center

    xgrid, ygrid = np.meshgrid(grid, grid)
    return xgrid, ygrid


def bin_image(arr, sub_grid=None):
    new_shape = (arr.shape[0]//sub_grid,arr.shape[1]//sub_grid)
    shape = (new_shape[0], sub_grid,
             new_shape[1], sub_grid)
    return arr.reshape(shape).mean(axis=(-1,1)) #the mean operation happen at axis -1 and 1


def xy_transform(x, y, x_cen, y_cen, phi):
    """
    A clock-wise rotational matrix exert on vector (x, y)
    :param x: the x coordinate of the vector, which need to be rotated 
    :param y: the y coordinate of the vector, which need to be rotated  
    :param x_cen: the x coordinate of rotation center
    :param y_cen: the y coordinate of rotation center
    :param phi: the rotation angle in degree
    x,y can be a scalar or array. suppose x,y is a scalar, then this
    matrix rotate the vector (x,y) clockwisely with respect to the
    center (x_cen,y_cen).
    """
    xnew=(x-x_cen)*np.cos(np.pi*phi/180.0)+(y-y_cen)*np.sin(np.pi*phi/180.0)
    ynew=-(x-x_cen)*np.sin(np.pi*phi/180.0)+(y-y_cen)*np.cos(np.pi*phi/180.0)
    return (xnew, ynew)


def relocate_radii(x, y, eps=1e-8):
    """
    relocate the relative displacement x and y, to avoid 
    the `odd` points, where r=sqrt(x**2+y**2) -> 0, so that 
    calculations of `1/r` become unphysical.
    """
    r = np.sqrt(x**2 + y**2)

    if isinstance(r, np.ndarray):
        with np.errstate(all='ignore'):
            #coordinates relocation
            scale = np.where(r<eps, eps/r, 1.0)
            r *= scale
            x *= scale
            y *= scale
            #deal with (0, 0) point
            bool_nan = np.isnan(r)
            r[bool_nan] = eps
            x[bool_nan] = eps
            y[bool_nan] = 0.0
    else: #if x and y are numbers
        if r < eps:
            try:
                scale = eps/r
                r *= scale
                x *= scale
                y *= scale
            except: #deal with (0, 0) point
                r = eps
                x = eps
                y = 0.0

    return x, y, r


def add_noise_to_image_gaussian(ideal_image=None, skylevel=0.5, dark_current=0.0, readout_noise=0.0, n_exposures=1, total_exposure_time=300.0, seed=None):
    '''
    readout noise: e-/pixel, the standard deviation of readout gaussian noise (with zero mean) 
    dark current: e- / pixel / s, Note ensure dark current is in this unit!
    ideal_iamge,skylevel: e-/pixel/s
    '''
    background = skylevel + dark_current

    image_= ideal_image + background
    ideal_counts_ = image_ * total_exposure_time  #ideal mean counts

    noise_map = np.sqrt(ideal_counts_+n_exposures*readout_noise**2)
    noise_map = noise_map/total_exposure_time
    
    if seed is not None:
        np.random.seed(seed)
        
    noisy_image = ideal_image + np.random.normal(loc=0.0, scale=noise_map)
    return noisy_image, noise_map


def SN_from_lensed_image(image_map, noise_map, return_segment_map=False):
    data = np.copy(image_map)
    sigma = np.copy(noise_map)

    image_1d = data.flatten()
    noise_1d = sigma.flatten()

    sort_indices = np.argsort(-image_1d/noise_1d)
    sorted_image_1d = image_1d[sort_indices]
    sorted_noise_1d = noise_1d[sort_indices]
    sorted_image_1d_cum = np.cumsum(sorted_image_1d)
    sorted_noise_1d_cum = np.cumsum(sorted_noise_1d**2)**0.5
    SN_lensed_images = (sorted_image_1d_cum/sorted_noise_1d_cum).max()

    if return_segment_map:
        indices_1d_max_sn = np.argmax(sorted_image_1d_cum/sorted_noise_1d_cum)
        segment_indices_1d = sort_indices[0:indices_1d_max_sn+1]
        segment_indices_2d = np.unravel_index(segment_indices_1d, shape=data.shape)
        segment_map = np.zeros_like(data, dtype='int')
        segment_map[segment_indices_2d] = 1
        return SN_lensed_images, segment_map
    else:
        return SN_lensed_images


def gauss_psf(npix=None, dpix=None, sigma=None, fwhm=None, ree80=None, nsub=4):
    if sigma is None:
        if fwhm is not None:
            sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        elif ree80 is not None:
            sigma = ree80/np.sqrt(np.log(1-0.8)*(-2.0))
        else:
            raise Exception('Please input the sigma or FWHM of the gaussian psf')
    
    if npix is None:
        npix = int(10*sigma/dpix) #5 sigma width for the gauss psf
        if (npix%2) == 0:
            npix +=1

    xgrid, ygrid = make_grid_2d(npix, dpix, nsub)
    rgrid=np.sqrt(xgrid**2.+ygrid**2.)

    psf = np.exp(-0.5*(rgrid/sigma)**2)
    if nsub is not None:
        psf = bin_image(psf, nsub)

    psf /= psf.sum()
    return psf 


def mag2cps(magnitude, magnitude_zero_point):
    """
    From lenstronomy

    converts an apparent magnitude to counts per second

    The zero point of an instrument, by definition, is the magnitude of an object that produces one count
    (or data number, DN) per second. The magnitude of an arbitrary object producing DN counts in an observation of
    length EXPTIME is therefore:
    m = -2.5 x log10(DN / EXPTIME) + ZEROPOINT

    :param magnitude:
    :param magnitude_zero_point:
    :return: counts per second
    """
    delta_M = magnitude - magnitude_zero_point
    cps = 10**(-delta_M/2.5) #cps: counts per second
    return cps


def cps2mag(cps, magnitude_zero_point):
    delta_M = -2.5 * np.log10(cps) #cps: counts per second
    return delta_M + magnitude_zero_point


def sky_cps_from_mag(sky_mag_per_arc2, mag_zero, dpix):
    cps_per_arc2 = mag2cps(sky_mag_per_arc2, mag_zero)
    npix_in_arc2 = 1.0/dpix**2
    cps_per_pix = cps_per_arc2 / npix_in_arc2
    return cps_per_pix


def sky_mag_from_cps(sky_cps_per_pix, mag_zero, dpix):
    cps_per_arc2 = sky_cps_per_pix / dpix**2
    return cps2mag(cps_per_arc2, mag_zero)


def rescale_image_with_mag(image, dpix, mag, mag_zero):
    total_flux = image.sum()*dpix**2 
    factor = mag2cps(mag,mag_zero)/total_flux
    return image * factor


def condition_resolved_image(thetaE, Re_src, psf_fwhm):
    # return thetaE**2 > (Re_src**2 + (psf_fwhm/2.0)**2)
    return 2*thetaE**2 >= (2*Re_src)**2+psf_fwhm**2


def condition_tangential_arc(magnification, Re_src, psf_fwhm):
    cond1 = (magnification*Re_src > psf_fwhm)
    cond2 = (magnification>3)
    return (cond1 and cond2)


def condition_images_sn(total_sn_of_lensed_images=None):
    return total_sn_of_lensed_images>20.0


def cart2pol(x,y):
    x, y = np.asarray(x), np.asarray(y)

    r = np.sqrt(x**2. + y**2.)
    theta = np.arctan2(y,x)

    return r, theta


def pol2cart(r,theta):
    r, theta = np.asarray(r), np.asarray(theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x,y


def sie_caustics_analytic(sie_obj):
    """
    Follow the notation of korman93
    adapted from the Vislens code.
    """
    b = sie_obj.thetaE
    f = sie_obj.q
    xc = sie_obj.xc
    yc = sie_obj.yc
    PA = sie_obj.PA-90
    deg2rad = np.pi/180

    fprime = np.sqrt(1. - f**2.) 
    # Caustics calculated over a full 0,2pi angle range
    phi = np.linspace(0,2*np.pi,2000)
    # K+94, eq 21c; needed for diamond caustic
    Delta = np.sqrt(np.cos(phi)**2. + f**2. * np.sin(phi)**2.)
    
    if np.isclose(f,1.):
        xr,yr = -b*np.cos(phi)+xc, -b*np.sin(phi)+yc
        return [np.array([xr,yr]).T, None]
    else:
        # Calculate the radial caustic coordinates
        xr = (-b*np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f)
        yr = (-b*np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime)
        
        # Now rotate & shift the caustic to match the PA & loc of the lens
        r,th = cart2pol(xr,yr)
        xr,yr = pol2cart(r,th+PA*deg2rad)
        xr += xc
        yr += yc
        # Calculate the tangential caustic coordinates
        xt = b*(((np.sqrt(f)/Delta) * np.cos(phi)) - ((np.sqrt(f)/fprime)*np.arcsinh(fprime/f * np.cos(phi))))
        yt = b*(((np.sqrt(f)/Delta) * np.sin(phi)) - ((np.sqrt(f)/fprime)*np.arcsin(fprime * np.sin(phi))))
        # ... and rotate it to match the lens
        r,th = cart2pol(xt,yt)
        xt,yt = pol2cart(r,th+PA*deg2rad)
        xt += xc
        yt += yc
        return [np.array([xr,yr]).T,np.array([xt,yt]).T]


def draw_ellipse(xc, yc, radius, q, PA):
    phi = np.linspace(0, 2*np.pi, 361)

    #draw ellipse, x-direction is the major axis
    x_ellipse = radius*np.cos(phi)/np.sqrt(q)
    y_ellipse = radius*np.sin(phi)*np.sqrt(q)

    #rotate acoording to the PA
    r_ellipse, phi_ellipse = cart2pol(x_ellipse, y_ellipse)
    phi_ellipse += phi/180*np.pi
    x_ellipse, y_ellipse = pol2cart(r_ellipse, phi_ellipse)

    #translate the ellipse center
    x_ellipse += xc
    y_ellipse += yc

    return x_ellipse, y_ellipse


def check_ring(sie_obj, src_xcenter, src_ycenter, src_reff, src_q, src_PA):
    # https://stackoverflow.com/questions/31542843/inpolygon-examples-of-matplotlib-path-path-contains-points-method
    _, tangential_caustics = sie_caustics_analytic(sie_obj)

    x_ellipse, y_ellipse = draw_ellipse(src_xcenter, src_ycenter, src_reff, src_q, src_PA)
    points_ellipse = np.array([x_ellipse, y_ellipse]).T

    if tangential_caustics is not None:
        p = path.Path(tangential_caustics) 
        return np.any(p.contains_points(points_ellipse))
    else:
        return False