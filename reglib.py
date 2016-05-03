import io, os, copy
import numpy as np
from datetime import date, timedelta
from scipy import fftpack
import scipy.sparse.linalg as spli
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.dates import date2num, num2date
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cross_validation import KFold

from defaults import *

def monotonic(x):
    dx = np.diff(x)
    return np.all(dx <= 0) or np.all(dx >= 0)

def form_observation_matrices(dataset, ndays, ignore_weight=False):
    """
    dataset.keys = 'refl', 'weight', 'kernels', 'idoy', 'date', 'year', 'doy'
    Note: idoy is more like a day counter, that counts days since the
    baseline, i.e. day_0 and exceeds 365 (or 366) for multi-year datasets.
    """
    # get some counts of sorts...
    nk, nbands, nsamp = dataset['kernels'].shape
    steps = nbands * nk
    
    # create sparse matrices
    bigshape = tuple(np.array([ndays, ndays]) * nk * nbands)
    A_o = scipy.sparse.lil_matrix(bigshape)
    b_o = scipy.sparse.lil_matrix((bigshape[0],1))
    
    # alias some variables for legibility
    refl = dataset['refl']
    kernels = dataset['kernels']
    weight = dataset['weight']
    
    # load into A and b matrices, every day is a 9x9 block along
    # the diagonal of A and a series of 9 elements along b (=vector)
    for i in xrange(nsamp):
        R_i = np.matrix(refl[...,i]).T
        k33 = np.matrix(kernels[...,i]).T # k33 has shape (3 x 3), but we need (3 x 9)
        if ignore_weight:
            W_i = np.eye(3) # there might be an issue with the weight matrices, this ignore-option is for debugging...
        else:
            W_i = np.matrix(weight[...,i])
        thisdoy = dataset['idoy'][i]
        
        # load k33 into its 3 x 9 matrix
        K_i = np.matrix(np.zeros((nbands,nbands*nk)))
        for j in xrange(nbands): K_i[j,j*nk:(j+1)*nk] = k33[j]
        
        KR_i = K_i.T * W_i * R_i
        KK_i = K_i.T * W_i * K_i
        sli = slice(thisdoy*steps, (thisdoy+1)*steps)
        
        # add ie accumulate in case of multiple samples per day
        try:
            A_o[sli,sli] += KK_i
            b_o[sli,0] += KR_i
        except:
            print "ERROR in forming observations matrix", i, '>>-->', sli
    
    return A_o, b_o

def form_regularisation_matrices(gamma_day, gamma_year, ndays, nk, nbands):
    # form a D matrix for days -- should replace by sparse
    I = np.eye(ndays)
    D = np.matrix(I - np.roll(I,-1))
    D1 = D * D.T

    # form a D matrix for years -- should replace by sparse
    D = np.matrix(I - np.roll(I,-365))
    D365 = D * D.T
    
    bigshape = tuple(np.array([ndays, ndays]) * nk * nbands)
    A_D = scipy.sparse.lil_matrix(bigshape)
    A_D365 = scipy.sparse.lil_matrix(bigshape)
    
    steps = nbands * nk
    for i in xrange(steps):
        A_D[i::steps, i::steps] = D1
        A_D365[i::steps, i::steps] = D365
    
    A_d = gamma_day * (A_D.T * A_D)
    A_y = gamma_year * (A_D365.T * A_D365)

    return A_d, A_y

"""
def form_prior_matrices(prior, ndays, bands=['VIS', 'NIR', 'SW'], nk=3, nsamp=46, scale=1.0):

#     prior: e.g. select keys 'prior.v2.snow' or 'prior.v2.nosnow' in ncdata
#     day_0: baseline date
#     ndays: number of days from baseline (day_0) until last day of inferred state vector
#     nsamp: equals 46 if prior is sampled every 8 days over one year
#     scale: not sure why this was used... save to keep at one.

     nbands = len(bands)
    
     # mean
     mean = []
     for b in bands:
         m = np.array(prior['Mean_%s'%b]).squeeze().reshape(nk,nsamp)
         mean.append(m)
     mean = np.array(mean)
     print mean.shape

     # var -> weight
     kk = ['f%d'%i for i in xrange(nk)]
     var = [[]]*nbands
     for band in xrange(nbands):
         var[band] = []
         for p,f0 in enumerate(kk):
             cc = np.array(prior['Cov_%s_%s_%s_%s'%(bands[band],f0,bands[band],f0)]).squeeze()
             var[band].append(cc)  
     var = np.array(var)
     weight = 1./var
     weight = weight/scale
    
     # these are formed 
     xp = np.zeros((nbands*nk*ndays))
     Wp = np.zeros((nbands*nk*ndays))
    
     for d in xrange(ndays):
         thisday = day_0 + timedelta(d)
         thisprior = ((thisday - date(thisday.year,1,1)).days)/8
         for k in xrange(nk): 
             xp[d*nbands*nk+k*nbands:d*nbands*nk+(k+1)*nbands] = mean[:,k,thisprior]
             Wp[d*nbands*nk+k*nbands:d*nbands*nk+(k+1)*nbands] = weight[:,k,thisprior]
     A_p = scipy.sparse.lil_matrix(scipy.sparse.diags(Wp,0))
     b_p = A_p * scipy.sparse.lil_matrix(xp).T

     return A_p, b_p
"""

def compute_state_vector(dataset, gamma_day, gamma_year, ndays=None, \
                         ignore_weight=False, do_unc=False):
    if ndays == None:
        ndays = dataset['idoy'][-1]+1
    assert np.all( [i >= 0 and i <= ndays for i in dataset['idoy']] )
    
    nk, nbands, _ = dataset['kernels'].shape
    
    # form A & b matrices
    A_o, b_o = form_observation_matrices(dataset, ndays, ignore_weight=ignore_weight)
    A_d, A_y = form_regularisation_matrices(gamma_day, gamma_year, ndays, nk, nbands)
#    A_p, b_p = form_prior_matrices(priordata, ndays, \
#                                    bands=['VIS', 'NIR', 'SW'], nk=3, nsamp=46, scale=1.0)
    
    # solve for state vector
    A = (A_o + A_d + A_y).tocsc()
    b = b_o.tocsc()
    x = spli.spsolve(A, b, use_umfpack=True)
    if do_unc:
        assert np.all( A.diagonal() != 0 ) # check requirement for taking inverse of diagonal matrix...
        post_cov = 1.0 / A.diagonal()
        return x, post_cov
    else:
        return x # order f0(VIS), f1(VIS), f2(VIS), f0(NIR), f1(NIR), f2(NIR), f0(SW), f1(SW), f2(SW), ...repeated for ndays

def compute_y_hat(dataset, x):
    """
    Estimate BBDR reflectance in VIS, NIR and SW bands given a state vector
    """
    nk, nbands = dataset['kernels'].shape[:2]
    idoys = dataset['idoy']
    nobs = idoys.size

    # fetch states for idoy...
    idxs = np.zeros(nobs*nk*nbands, dtype=int)
    for i, idoy in enumerate(idoys):
        sli = slice(i*nk*nbands, (i+1)*nk*nbands)
        rng = range(idoy*nk*nbands, (idoy+1)*nk*nbands)
        idxs[sli] = rng # maps to: f0(VIS), f1(VIS), f2(VIS), f0(NIR), f1(NIR), f2(NIR), f0(SW), f1(SW), f2(SW)
    x_idoy = x[idxs] # state vector arranged by order (w/ duplication!) of observations
    
    # define K matrix that matches the order of idoy...
    K_idoy = np.zeros((nbands*nobs, nbands*nk*nobs), dtype=float)
    for i, idoy in enumerate(idoys):
        dim0 = i*nbands # top-left of this block of 3 rows x 9 columns
        dim1 = i*nbands*nk # top-left of this block of 3 rows x 9 columns
        for j in range(nbands):
            for jj in range(nk):
                # (ncdata[sensor]['kernels'] =) dataset['kernels'] has shape: [nk, nbands, ndoys]
                K_idoy[dim0+j, dim1+(j*3)+jj] = dataset['kernels'][jj,j,i]
    
    y_hat_vector = np.matrix(K_idoy) * np.matrix(x_idoy).T
    y_hat = y_hat_vector.reshape(nobs, nbands).T
    y_hat = np.asarray(y_hat)
    return y_hat

def plot_state_vector(dataset, x, ymin=-1.0, ymax=1.0, post_cov=None):
    idates = dataset['date']
    first_day = dataset['date'][0]
    last_day = dataset['date'][-1]
    ndays = dataset['idoy'][-1]+1
    nk, nbands, nobs = dataset['kernels'].shape
    step = nk * nbands
    
    day_range = np.arange(ndays)
    baseline = first_day - timedelta(days=dataset['idoy'][0])
    date_range = np.array([baseline + timedelta(days=i) for i in day_range])
    
    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    
    idoys = dataset['idoy']
    
    
    if post_cov != None:
        s = 1.96
        ax1.fill_between(date_range,  x[0::step]-s*np.sqrt(post_cov[0::step]), x[0::step]+s*np.sqrt(post_cov[0::step]), color="0.8")
        ax1.fill_between(date_range,  x[3::step]-s*np.sqrt(post_cov[3::step]), x[3::step]+s*np.sqrt(post_cov[3::step]), color="0.8")
        ax1.fill_between(date_range,  x[6::step]-s*np.sqrt(post_cov[6::step]), x[6::step]+s*np.sqrt(post_cov[6::step]), color="0.8")
        
        ax2.fill_between(date_range,  x[1::step]-s*np.sqrt(post_cov[1::step]), x[1::step]+s*np.sqrt(post_cov[1::step]), color="0.8")
        ax2.fill_between(date_range,  x[4::step]-s*np.sqrt(post_cov[4::step]), x[4::step]+s*np.sqrt(post_cov[4::step]), color="0.8")
        ax2.fill_between(date_range,  x[7::step]-s*np.sqrt(post_cov[7::step]), x[7::step]+s*np.sqrt(post_cov[7::step]), color="0.8")
        
        ax3.fill_between(date_range,  x[2::step]-s*np.sqrt(post_cov[2::step]), x[2::step]+s*np.sqrt(post_cov[2::step]), color="0.8")
        ax3.fill_between(date_range,  x[5::step]-s*np.sqrt(post_cov[5::step]), x[5::step]+s*np.sqrt(post_cov[5::step]), color="0.8")
        ax3.fill_between(date_range,  x[8::step]-s*np.sqrt(post_cov[8::step]), x[8::step]+s*np.sqrt(post_cov[8::step]), color="0.8")
    
    # order of elements in x: f0(VIS), f1(VIS, f2(VIS), f0(NIR), ..., f2(SW) repeated for ndays
    ax1.plot(date_range, x[0::step], label='f0(VIS)', alpha=0.8)
    ax1.plot(date_range, x[3::step], label='f0(NIR)', alpha=0.8)
    ax1.plot(date_range, x[6::step], label='f0(SW)', alpha=0.8)
    ax1.plot(idates, np.ones_like(idates)*ymax, 'k+', ms=45, alpha=0.3)
    ax1.legend(fontsize=22)
    ax1.set_ylabel('isotropic', fontsize=22)
    ax1.set_xlim((first_day, last_day))
    ax1.set_ylim((ymin, ymax))
    
    ax2.plot(date_range, x[1::step], label='f1(VIS)', alpha=0.8)
    ax2.plot(date_range, x[4::step], label='f1(NIR)', alpha=0.8)
    ax2.plot(date_range, x[7::step], label='f1(SW)', alpha=0.8)
    ax2.plot(idates, np.ones_like(idates)*ymax, 'k+', ms=45, alpha=0.3)
    ax2.set_xlim((first_day, last_day))
    ax2.legend(fontsize=22)
    ax2.set_ylabel('volumetric', fontsize=22)
    ax2.set_ylim((ymin, ymax))
    
    ax3.plot(date_range, x[2::step], label='f2(VIS)', alpha=0.8)
    ax3.plot(date_range, x[5::step], label='f2(NIR)', alpha=0.8)
    ax3.plot(date_range, x[8::step], label='f2(SW)', alpha=0.8)
    ax3.plot(idates, np.ones_like(idates)*ymax, 'k+', ms=45, alpha=0.3)
    ax3.set_ylabel('geometric', fontsize=22)
    ax3.set_xlim((first_day, last_day))
    ax3.legend(fontsize=22)
    ax3.set_ylim((ymin, ymax))
    ax3.set_xlabel("Date", fontsize=22)
    
    # make x-axis ticks bigger...
    for ax in [ax1, ax2, ax3]:
        for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(22) 
        for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(22) 
    return


def load_the_data():
	""" returns ncdata, abdata, nbands, nk, nskip """

	datakeys = np.array(['bbdr.meris', 'bbdr.vgt', 'ga.brdf.merge',\
	       'ga.brdf.nosnow', 'ga.brdf.snow', 'mod09', 'myd09',\
	       'prior.v2.nosnow', 'prior.v2.snow', 'prior.v2.snownosnow'])

	# load the datasets into a dictionary in ncdata
	ncdata = {}
	abdata = {}
	for k in datakeys:
	    try:
		abdata[k] = load_obj('obj/' + k + '_Ab_' )
		print '\nAb',k,
	    except:
		pass
	    try:
		ncdata[k] = load_obj('obj/' + k + '_s3.0_' )
		print '\n\t3',k,
	    except:
		ncdata[k] = load_obj('obj/' + k + '_s2.0_' )
		print '\n\t\t2',k,
	
	
	nbands, nk = ncdata['mod09']['kernels'].shape[0:2]
	nskip = nbands *nk
	return ncdata, abdata, nbands, nk, nskip

def sort_idoy(dataset):
    e = np.argsort(dataset['idoy'])
    dataset['idoy'] = dataset['idoy'][e]
    dataset['date'] = dataset['date'][e]
    dataset['kernels'] = dataset['kernels'][...,e]
    dataset['weight'] = dataset['weight'][...,e]
    dataset['refl'] = dataset['refl'][...,e]
    if 'source' in dataset.keys():
        dataset['source'] = dataset['source'][e]
    return dataset

def get_subset(dataset, IN_IDX):
    subset = {}
    subset['idoy'] = dataset['idoy'][IN_IDX]
    subset['date'] = dataset['date'][IN_IDX]
    subset['kernels'] = dataset['kernels'][...,IN_IDX]
    subset['weight'] = dataset['weight'][...,IN_IDX]
    subset['refl'] = dataset['refl'][...,IN_IDX]
    if 'source' in dataset.keys():
        subset['source'] = dataset['source'][IN_IDX]
    return subset

def pool_datasets(datasets):
    " datasets is a list of dictionaries (version S2.0 for prior and GA, and S3.0 for BBDR) "
    superset = {}
    for i, dataset in enumerate(datasets):
        n_obs = dataset['idoy'].size
        if i == 0:
            superset = copy.copy(dataset)
            superset['source'] = np.ones(n_obs, dtype=int) # points to elements in datasets
        else:
            superset['idoy'] = np.hstack(( superset['idoy'], dataset['idoy'] ))
            superset['date'] = np.hstack(( superset['date'], dataset['date'] ))
            superset['kernels'] = np.dstack(( superset['kernels'], dataset['kernels'] ))
            superset['weight'] = np.dstack(( superset['weight'], dataset['weight'] ))
            superset['refl'] = np.hstack(( superset['refl'], dataset['refl'] ))
            superset['source'] = np.hstack(( superset['source'], np.ones(n_obs, dtype=int)*(i+1) ))  
            # just be aware that following two objects have different/full shape (> n_obs)
            # not using them for the moment!!!
            #superset['doy'] = np.hstack(( superset['doy'], dataset['doy'] ))
            #superset['year'] = np.hstack(( superset['year'], dataset['year'] ))

    # sort w.r.t. idoy/dates
    superset = sort_idoy(superset)
    
    return superset

def subset_to_year_and_reset_idoy(dataset, year):
    bln = np.array([date.year == year for date in dataset['date']], dtype=bool)
    subset = get_subset(dataset, bln)
    subset['idoy'] -= subset['idoy'].min()
    return subset

def compute_rmse_with_LOO_cv(dataset, gamma_day, gamma_year):
    " Compute state vector with cross validation and return also RMSE "
    nk = dataset['kernels'].shape[0]
    SSE = np.zeros((nk, 1), dtype=float)
    n_obs = dataset['idoy'].size
    n_days = dataset['idoy'][-1]+1
    kf = KFold(n_obs, n_folds=n_obs) # leaf one out
    for train, test in kf:
        trainset = get_subset(dataset, train)
        testset  = get_subset(dataset, test)
        x_train = compute_state_vector(trainset, gamma_day, gamma_year, ndays=n_days)
        y_hat_test = compute_y_hat(testset, x_train)
        SSE += (testset['refl'] - y_hat_test)**2
    MSE = SSE / n_obs
    RMSE = np.sqrt(MSE)
    RMSE = RMSE.reshape((3,)) # reshape to 1d array (instead of (3,1)
    return RMSE

def compute_rmse_per_sensor_cv(dataset, gamma_day, gamma_year):
    assert 'source' in dataset.keys()
    sources = np.unique(dataset['source']) # e.g. references to different sensors or products
    ndays = dataset['idoy'][-1]+1
    SSE = np.zeros(3, dtype=float)
    for source in sources:
        bln = dataset['source'] == source
        testset = get_subset(dataset, bln)
        trainset = get_subset(dataset, ~bln)
        x = compute_state_vector(trainset, gamma_day, gamma_year, ndays=ndays)
        y_hat = compute_y_hat(testset, x)
        residual = testset['refl'] - y_hat
        SSE += (residual**2).sum(axis=-1)
    MSE = SSE/float(len(sources))
    RMSE = np.sqrt(MSE)
    return RMSE

def shift_gamma_with_LOO_cv(dataset, gammas_day=np.logspace(5, 11, 15), \
                gammas_year=np.logspace(1, 8, 15)):
    " Compute the state vector and y_hat for a range of different gamma's "
    surface = []
    for gamma_day in gammas_day:
        for gamma_year in gammas_year:
            rmse = compute_rmse_with_LOO_cv(dataset, gamma_day, gamma_year)
            result = (gamma_day, gamma_year, rmse[0], rmse[1], rmse[2])
            print "%6.2E %6.2E %6.2E %6.2E %6.2E" % result
            surface.append(result)
        print "_" * 45
    surface = np.array(surface)
    return surface

def shift_gamma_per_sensor_cv(dataset, gammas_day=np.logspace(5, 11, 15), \
                gammas_year=[1]): #np.logspace(1, 8, 15)):
    " Compute the state vector and y_hat for a range of different gamma's "
    surface = []
    for gamma_day in gammas_day:
        for gamma_year in gammas_year:
            rmse = compute_rmse_per_sensor_cv(dataset, gamma_day, gamma_year)
            result = (gamma_day, gamma_year, rmse[0], rmse[1], rmse[2])
            print "%6.2E %6.2E %6.2E %6.2E %6.2E" % result            
            surface.append(result)
        print "_" * 45
    surface = np.array(surface)
    return surface



