#%%
from SimCsstLens.LensPop.Population import LensPopulation
from SimCsstLens import CosmologyDistance as CD
import math
from multiprocessing import Pool
import time
import copy
import numpy as np

this_cosmos = CD.CosmosDist(Om0=0.3, Ode0=0.7, h=0.7)
lens_pop = LensPopulation(
    vdisp_floor=50, 
    zl_max=2.5, 
    cosmo_dist=this_cosmos,
    src_catalog_type='lsst',
    bands=['g', 'r', 'i', 'z'],
)

sky_frac = 17500.0/41252.96
N_etgs = lens_pop.dfl_pop.number_of_etgs(sky_frac=sky_frac) #ideal lenses
print(f'N ETGs in {sky_frac} fraction of sky area: {N_etgs:.2e}')

Nsamples_per_draw = 500000
Ndraw = math.ceil(N_etgs/Nsamples_per_draw)


#%%
lens_pop_tmp = {}
def run_simulation(count_draw):
    np.random.seed(count_draw)
    lens_pop_tmp[count_draw] = copy.deepcopy(lens_pop)
    lens_pop_tmp[count_draw].draw_lens_samples(
        nsamples=Nsamples_per_draw,
        nsrcs_per_sample=1,
        src_over_density=1,
    )
    lens_pop_tmp[count_draw].save_ideal_lens_samples(
        output_path='./samples',
        output_file=f'lenses_{count_draw}.hdf5'
    )
    lens_pop_tmp[count_draw] = None

t0 = time.time()
pool = Pool(processes=64)
pool.map(run_simulation, list(range(Ndraw)))
# run_simulation(0)
t1 = time.time()
print(f'{t1-t0:.0f} secs')


#%%
from SimCsstLens.SimLensImage.MockSurvey import MockSurvey
survey = MockSurvey(config_path='.', config_file='csst_setting.yaml')
survey_tmp = {}
def run_simulation(count_draw):
    np.random.seed(count_draw)
    t0 = time.time()
    survey_tmp[count_draw] = copy.deepcopy(survey)
    survey_tmp[count_draw].load_ideal_lens_samples(
        file_path='./samples', 
        file_name=f'lenses_{count_draw}.hdf5'
    )
    survey_tmp[count_draw].simulate_observed_lenses(output_image=False)
    survey_tmp[count_draw].append_obs_info(
        output_path='./samples', 
        output_file=f'lenses_{count_draw}.hdf5'      
    )
    survey_tmp[count_draw] = None
    t1 = time.time()
    print(f'Time comsumption of iteration-{count_draw}: {t1-t0:.0f} secs')
    

t0 = time.time()
pool = Pool(processes=64)
pool.map(run_simulation, list(range(Ndraw)))
t1 = time.time()
print(f'{t1-t0:.0f} secs')
