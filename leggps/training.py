import scipy as sp
import scipy.optimize
from . import legops
import tensorflow as tf
import numpy as np
import numpy.random as npr
from . import constructions


def fit_model_family(ts,xs,model_family,p_init,maxiter=100,use_tqdm_notebook=False):
    '''
    Fits a custom LEG model

    Input:
    - ts: list of timestamp-vectors: nsamp x [ragged]
    - xs: list of observations:      nsamp x [ragged] x n 
    - model_family: model family to fit
    - p_init: -- initial conditions for the parameter vector of the model family
    - [optional] maxiter -- max number of iters to use in BFGS
    - [optional] use_tqdm_notebook -- whether to make an update bar with tqdm

    Output: dictionary with lots of keys.  See supplementary.pdf for details.  Important keys are:
    - message (result of optimization)
    - params (a dictionary with keys for each parameter of a LEG model)
    - nats (the negative log likelihood divided by the number of observations)    
    '''

    # store initial values
    N,R,B,Lambda=model_family.p2NRBL(p_init)
    initial_params=dict(N=N.numpy(),R=R.numpy(),B=B.numpy(),Lambda=Lambda.numpy())

    # process dedups
    time_info=[constructions.dedup_ts(tf.convert_to_tensor(x,dtype=tf.float64)) for x in ts]
    xs=[tf.convert_to_tensor(x,dtype=tf.float64) for x in xs]
    n=xs[0].shape[1]
    nobs = np.sum([np.prod(x.shape) for x in xs])
    
    # functions for scipy.optimize
    nats=[]
    def func(p):
        Ls=0
        for x,(sub_ts,sub_idxs) in zip(xs,time_info):
            Ls+= model_family.log_likelihood(sub_ts,x,sub_idxs,p)
        loss=-Ls.numpy()/nobs
        nats.append(loss)
        return loss
    def jac(p):
        gs=0
        for x,(sub_ts,sub_idxs) in zip(xs,time_info):
            gs+= model_family.informant(sub_ts,x,sub_idxs,p)
        return -gs/nobs

    # get an initial loss
    func(p_init)

    # fit it
    if use_tqdm_notebook:
        import tqdm.notebook
        with tqdm.notebook.tqdm() as t:
            def callback(*args,**kwargs):
                t.update(len(nats))
                t.set_description(f"nats={nats[-1]:.2f}")
            result=sp.optimize.minimize(func,p_init,jac=jac,options=dict(maxiter=maxiter),callback=callback)
    else:
        result=sp.optimize.minimize(func,p_init,jac=jac,options=dict(maxiter=maxiter))

    # supplement loss dictionary with some stuff of interest
    result['nats']=nats

    # store initial params
    result['initial_params']=initial_params

    # store final params:
    N,R,B,Lambda=model_family.p2NRBL(result['x'])
    result['params']=dict(N=N.numpy(),R=R.numpy(),B=B.numpy(),Lambda=Lambda.numpy())

    # we call the parameters "p" not "x"
    result['p']=result['x']
    del result['x']

    # done
    return result

def fit(ts,xs,ell=None,N=None,R=None,B=None,Lambda=None,maxiter=100,use_tqdm_notebook=False):
    '''
    fit the LEG model with rank ell

    Input:
    - ts: list of timestamp-vectors: nsamp x [ragged]
    - xs: list of observations:      nsamp x [ragged] x n 
    - ell: order of the LEG model to fit
    - [optional] N,R,B,Lambda -- initial conditions 
    - [optional] maxiter -- max number of iters to use in BFGS
    - [optional] use_tqdm_notebook -- whether to make an update bar with tqdm

    Output: dictionary with lots of keys.  See supplementary.pdf for details.  Important keys are:
    - message (result of optimization)
    - params (a dictionary with keys for each parameter of a LEG model)
    - nats (the negative log likelihood divided by the number of observations)
    '''

    mf =LEGFamily(ell,xs[0].shape[1])
    p_init=mf.get_initial_guess(ts,xs,N=N,R=R,B=B,Lambda=Lambda)
    return fit_model_family(ts,xs,mf,p_init,use_tqdm_notebook=use_tqdm_notebook)


r'''
                     _      _    __                 _ _ _           
 _ __ ___   ___   __| | ___| |  / _| __ _ _ __ ___ (_) (_) ___  ___ 
| '_ ` _ \ / _ \ / _` |/ _ \ | | |_ / _` | '_ ` _ \| | | |/ _ \/ __|
| | | | | | (_) | (_| |  __/ | |  _| (_| | | | | | | | | |  __/\__ \
|_| |_| |_|\___/ \__,_|\___|_| |_|  \__,_|_| |_| |_|_|_|_|\___||___/
                                                                    

'''

class LEGFamily:
    def __init__(self,ell,n):
        self.ell=ell
        self.n=n

        msk=np.tril(np.ones((self.ell,self.ell)))
        self.N_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk=np.tril(np.ones((self.ell,self.ell)),k=-1)
        self.R_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk=np.tril(np.ones((self.n,self.n)))
        self.Lambda_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])
    
        self.psize = self.N_idxs.shape[0]+self.R_idxs.shape[0]+self.ell*self.n+self.Lambda_idxs.shape[0]

    def p2NRBL(self,p):
        i=0
        
        # N!
        sz=self.N_idxs.shape[0]
        N=tf.scatter_nd(self.N_idxs,p[i:i+sz],(self.ell,self.ell))
        i+=sz
        
        # R!
        sz=self.R_idxs.shape[0]
        R=tf.scatter_nd(self.R_idxs,p[i:i+sz],(self.ell,self.ell))
        i+=sz

        # B!
        sz=self.ell*self.n; B = tf.reshape(p[i:i+sz],(self.n,self.ell)); i+=sz
        
        # Lambda!
        sz=self.Lambda_idxs.shape[0]
        Lambda=tf.scatter_nd(self.Lambda_idxs,p[i:i+sz],(self.n,self.n))
        i+=sz
        
        return N,R,B,Lambda

    @tf.function(autograph=False)
    def informant(self,ts,x,idxs,p):
        '''
        gradient of log likelihood w.r.t. p
        '''
        with tf.GradientTape() as g:
            g.watch(p)
            N,R,B,Lambda = self.p2NRBL(p)
            nats = legops.leg_log_likelihood_tensorflow(ts,x,idxs,N,R,B,Lambda)
        return g.gradient(nats,p)

    @tf.function(autograph=False)
    def log_likelihood(self,ts,x,idxs,p):
        '''
        log likelihood
        '''
        N,R,B,Lambda = self.p2NRBL(p)
        return legops.leg_log_likelihood_tensorflow(ts,x,idxs,N,R,B,Lambda)

    def get_initial_guess(self,ts,xs,N=None,R=None,B=None,Lambda=None):
        # make up values when nothing is provided
        if N is None:
            N=np.eye(self.ell)
        if R is None:
            R=npr.randn(self.ell,self.ell)*.2
            R=.5*(R-R.T)
        if B is None:
            B=np.ones((self.n,self.ell))
            B=.5*B/np.sqrt(np.sum(B**2,axis=1,keepdims=True))
        if Lambda is None:
            Lambda = .1*np.eye(self.n)

        # make 'em nice for us
        N = tf.linalg.cholesky(N@tf.transpose(N))
        R = (R-tf.transpose(R))
        Lambda = tf.linalg.cholesky(Lambda@tf.transpose(Lambda))

        # put it all together
        pN=tf.gather_nd(N,self.N_idxs)
        pR=tf.gather_nd(R,self.R_idxs)
        pB=tf.reshape(B,(self.n*self.ell,))
        pL=tf.gather_nd(Lambda,self.Lambda_idxs)
        return tf.concat([pN,pR,pB,pL],axis=0)

class CeleriteFamily(LEGFamily):
    def __init__(self,nblocks,n):
        self.nblocks=nblocks
        self.ell=nblocks*2
        self.n=n

        msk=np.eye(self.ell,dtype=np.bool) + np.diag(np.tile([True,False],self.nblocks)[:-1],-1)
        self.N_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk = np.diag(np.tile([True,False],self.nblocks)[:-1],-1)
        self.R_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])

        msk=np.tril(np.ones((self.n,self.n)))
        self.Lambda_idxs = tf.convert_to_tensor(np.c_[np.where(msk)])
    
        self.psize = self.N_idxs.shape[0]+self.R_idxs.shape[0]+self.ell*self.n+self.Lambda_idxs.shape[0]

    def get_initial_guess(self,ts,xs):
        N=np.eye(self.ell)
        R=npr.randn(self.ell,self.ell)*.2
        B=np.ones((self.n,self.ell))
        B=.5*B/np.sqrt(np.sum(B**2,axis=1,keepdims=True))
        Lambda = .1*np.eye(self.n)
        N = tf.linalg.cholesky(N@tf.transpose(N))
        R = (R-tf.transpose(R))
        Lambda = tf.linalg.cholesky(Lambda@tf.transpose(Lambda))

        # put it all together
        pN=tf.gather_nd(N,self.N_idxs)
        pR=tf.gather_nd(R,self.R_idxs)
        pB=tf.reshape(B,(self.n*self.ell,))
        pL=tf.gather_nd(Lambda,self.Lambda_idxs)
        return tf.concat([pN,pR,pB,pL],axis=0)