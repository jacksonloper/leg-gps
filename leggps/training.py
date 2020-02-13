import scipy as sp
import scipy.optimize
from . import legops
import tensorflow as tf
import numpy as np
import numpy.random as npr
from . import constructions

def fit_problem(problem,use_tqdm_notebook=False,maxiter=100):
    # do optimization
    if use_tqdm_notebook:
        import tqdm.notebook
        with tqdm.notebook.tqdm() as t:
            def callback(*args,**kwargs):
                t.update(len(problem['losses']))
                t.set_description(f"nats={problem['losses'][-1]:.2f}")
            rez=sp.optimize.minimize(problem['func'],problem['init'],jac=problem['jac'],options=
                dict(maxiter=maxiter),callback=callback)
    else:
        rez=sp.optimize.minimize(problem['func'],problem['init'],jac=problem['jac'],options=
            dict(maxiter=maxiter))

    N,R,B,Lambda=problem['unfl'](rez['x'],problem['lengths'])
    rez['params']=dict(N=N,R=R,B=B,Lambda=Lambda)
    rez['losses']=problem['losses']

    return rez

def fit(ts,xs,ell,N=None,R=None,B=None,Lambda=None,maxiter=100,use_tqdm_notebook=False,
                            diag_Lambda=False):
    '''
    Input:
    - ts: list of timestamp-vectors: nsamp x [ragged]
    - xs: list of observations:      nsamp x [ragged] x n 
    - ell: scalar

    Output: dictionary with the keys
    - message (result of optimization)
    - params (a dictionary with keys for each parameter of a LEG model)
    - nats (the negative log likelihood divided by the number of observations)
    '''

    time_info=[constructions.dedup_ts(tf.convert_to_tensor(x,dtype=tf.float64)) for x in ts]
    xs=[tf.convert_to_tensor(x,dtype=tf.float64) for x in xs]
    n=xs[0].shape[1]
    nobs = np.sum([np.prod(x.shape) for x in xs])
    N,R,B,Lambda = initialize(ell,n,diag_Lambda=diag_Lambda,N=N,R=R,B=B,Lambda=Lambda)
    lengths=[N.shape,R.shape,B.shape,Lambda.shape]
    
    # functions for scipy.optimize
    losses=[]
    def func(p):
        N,R,B,Lambda=[tf.convert_to_tensor(x,dtype=tf.float64) for x in unfl(p,lengths)]
        Ls=0
        for x,(sub_ts,sub_idxs) in zip(xs,time_info):
            Ls+= funcfast_irregular(sub_ts,x,sub_idxs,N,R,B,Lambda)
        loss=-Ls.numpy()/nobs
        losses.append(loss)
        return loss
    def jac(p):
        N,R,B,Lambda=[tf.convert_to_tensor(x,dtype=tf.float64) for x in unfl(p,lengths)]
        gs=0
        for x,(sub_ts,sub_idxs) in zip(xs,time_info):
            gs+= fl(*[x.numpy() for x in jacfast_irregular(sub_ts,x,sub_idxs,N,R,B,Lambda)])
        return -gs/nobs

    # get the initial parameters in flattened form
    init=fl(N,R,B,Lambda)

    # get an initial loss
    func(init)

    # collect it 
    problem =dict(
        func=func,
        jac=jac,
        fl=fl,
        unfl=unfl,
        init=init,
        losses=losses,
        lengths=lengths,
    )

    # fit it
    return fit_problem(problem,maxiter=maxiter,use_tqdm_notebook=use_tqdm_notebook)


def fit_regular(dts,xs,ell,N=None,R=None,B=None,Lambda=None,maxiter=100,use_tqdm_notebook=False,
                        diag_Lambda=False):
    '''
    Input:
    - dts: vector with same length as xs
    - xs: list of samples of the form [nchain x n]
    - ell: scalar

    Output: dictionary with the keys
    - message (result of optimization)
    - params (a dictionary with keys for each parameter of a LEG model)
    - nats (the negative log likelihood divided by the number of observations)
    '''

    timeinfo=tf.convert_to_tensor(dts,dtype=tf.float64)
    xs=[tf.convert_to_tensor(x,dtype=tf.float64) for x in xs]
    n=xs[0].shape[1]
    nobs = np.sum([np.prod(x.shape) for x in xs])
    N,R,B,Lambda = initialize(ell,n,diag_Lambda=diag_Lambda,N=N,R=R,B=B,Lambda=Lambda)
    lengths=[N.shape,R.shape,B.shape,Lambda.shape]

    # functions for scipy.optimize
    losses=[]
    def func(p):
        N,R,B,Lambda=[tf.convert_to_tensor(x,dtype=tf.float64) for x in unfl(p,lengths)]
        Ls=0
        for d,x in zip(timeinfo,xs):
            Ls+= funcfast_regular(d,x,N,R,B,Lambda)
        loss=-Ls.numpy()/nobs
        losses.append(loss)
        return loss
    def jac(p):
        N,R,B,Lambda=[tf.convert_to_tensor(x,dtype=tf.float64) for x in unfl(p)]
        gs=0
        for d,x in zip(timeinfo,xs):
            gs+= fl(*[x.numpy() for x in jacfast_regular(d,x,N,R,B,Lambda)])
        return -gs/nobs

    # get the initial parameters in flattened form
    init=fl(N,R,B,Lambda)

    # get an initial loss
    func(init)

    # collect it together
    problem=dict(
        func=func,
        jac=jac,
        fl=fl,
        unfl=unfl,
        init=init,
        losses=losses,
        lengths=lengths,
    )

    # fit it
    return fit_problem(problem,maxiter=maxiter,use_tqdm_notebook=True)

@tf.function(autograph=False)
def jacfast_regular(d,x,N,R,B,Lambda):
    with tf.GradientTape() as g:
        for v in [N,R,B,Lambda]:
            g.watch(v)
        nats = legops.log_likelihood_regular(d,x,N,R,B,Lambda)
    return g.gradient(nats,[N,R,B,Lambda])
@tf.function(autograph=False)
def funcfast_regular(d,x,N,R,B,Lambda):
    return legops.log_likelihood_regular(d,x,N,R,B,Lambda)

@tf.function(autograph=False)
def jacfast_irregular(d,x,idxs,N,R,B,Lambda):
    with tf.GradientTape() as g:
        for v in [N,R,B,Lambda]:
            g.watch(v)
        nats = legops.log_likelihood_irregular(d,x,idxs,N,R,B,Lambda)
    return g.gradient(nats,[N,R,B,Lambda])
@tf.function(autograph=False)
def funcfast_irregular(d,x,idxs,N,R,B,Lambda):
    return legops.log_likelihood_irregular(d,x,idxs,N,R,B,Lambda)

def initialize(ell,n,diag_Lambda=False,N=None,R=None,B=None,Lambda=None):
    # initialize
    if N is None:
        N=np.eye(ell)
    if R is None:
        R=npr.randn(ell,ell)*.2
        R=.5*(R-R.T)
    if B is None:
        B=np.ones((n,ell))
        B=.5*B/np.sqrt(np.sum(B**2,axis=1,keepdims=True))
    if Lambda is None:
        if diag_Lambda:
            Lambda = np.ones(n)*.1
        else:
            Lambda = .1*np.eye(n)
    return N,R,B,Lambda



def fl(*args):
    return np.concatenate([x.ravel() for x in args])
def unfl(p,lengths):
    vs=[]
    i=0
    for shp in lengths:
        cursize=np.prod(shp)
        vs.append(p[i:i+cursize].reshape(shp))
        i=i+cursize
    return vs