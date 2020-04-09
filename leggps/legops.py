import tensorflow as tf
from . import constructions
from . import cr
import numpy as np
import dataclasses
import functools

def leg_log_likelihood_tensorflow(ts,x,idxs,N,R,B,Lambda):
    '''
    Input:
    - ts: list of times
    - x: nobs x n, representing X(ts[idxs[0]]),X(ts[idxs[1]]),X(ts[idxs[2]])...
    - idxs: nobs x n
    - N: ell x ell
    - R: ell x ell
    - B: n x ell
    - Lambda: ell x ell

    Output: the marginal log likelihood
        log p(
           X0 = x[0],
           X1 = x[1],
           X2 = x[2],
           ...
        )
        under the model
          Z~PEGGP(N,R)
          Xi ~ N(B Z(ts[idxs[i]), Lambda Lambda^T)
    '''

    distrinfo = txigbs2distrs(
        ts,x,idxs,
        constructions.calc_G(N,R),B,constructions.calc_LambdaLambdat(Lambda)
    )
    return distrinfo.log_likelihood()

def preprocess_numpy(ts,x,N,R,B,Lambda):
    ts = tf.convert_to_tensor(ts,dtype=tf.float64)
    x = tf.convert_to_tensor(x,dtype=tf.float64)
    N = tf.convert_to_tensor(N,dtype=tf.float64)
    R = tf.convert_to_tensor(R,dtype=tf.float64)
    B = tf.convert_to_tensor(B,dtype=tf.float64)
    Lambda = tf.convert_to_tensor(Lambda,dtype=tf.float64)

    # check sizes
    assert len(ts.shape)==1
    assert len(x.shape)==2
    assert len(N.shape)==2
    assert len(R.shape)==2
    assert len(B.shape)==2
    assert len(Lambda.shape)==2
    m=ts.shape[0]
    assert x.shape[0]==m
    n=x.shape[1]
    assert B.shape[0]==n
    ell = N.shape[0]
    assert N.shape[1]==ell
    assert R.shape[0]==ell
    assert R.shape[1]==ell
    assert Lambda.shape[0]==n
    assert Lambda.shape[1]==n

    # get info
    ts2,idxs=constructions.dedup_ts(ts)
    G=constructions.calc_G(N,R)
    Sig=constructions.calc_LambdaLambdat(Lambda)
    return ts2,x,idxs,G,B,Sig

def leg_log_likelihood(ts,x,N,R,B,Lambda):
    '''
    Input:
    - ts: list of times
    - x: nobs x n, representing X(ts[0]),X(ts[1]),X(ts[2])...
    - N: ell x ell
    - H: ell x ell
    - B: n x ell
    - Lambda: ell x ell

    Output: the marginal log likelihood
        log p(
           X0 = x[0],
           X1 = x[1],
           X2 = x[2],
           ...
        )
        under the model
          Z~PEGGP(N,R)
          Xi ~ N(B Z(ts[idxs[i]), Lambda Lambda^T)
    '''

    ts,x,idxs,G,B,Sig = preprocess_numpy(ts,x,N,R,Lambda)
    distrinfo = txigbs2distrs(ts,x,idxs,G,B,Sig)
    return distrinfo.log_likelihood().numpy()

def posterior_predictive(ts,x,targets,N,R,B,Lambda):
    '''
    Input:
    - ts: a vector of times
    - x: nchain x n, representing X(ts[0]),X(ts[1]),X(ts[2])...
    - targets: a vector of times
    - N: ell x ell
    - H: ell x ell
    - B: n x ell
    - Lambda: ell x ell
    - ts: a series of positions

    Output:
    - mean:  E[Y[i] |X] for each i
    - variance: Cov(Y[i]) |X) for each i

    under the model
      Z~PEG(N,R)
      X[i]|Z ~ N(B Z(t[i]), Lambda Lambda^T)
      Y[i] = BZ(targets[i])
    '''
    ts,x,idxs,G,B,Sig = preprocess_numpy(ts,x,N,R,B,Lambda)
    targets=tf.convert_to_tensor(targets,dtype=tf.float64)
    assert len(targets.shape)==1

    distrinfo = txigbs2distrs(ts,x,idxs,G,B,Sig)
    mean,Sig_diag,Sig_off=distrinfo.insample_posterior()
    means,variances=PEG_intercast(mean,Sig_diag,Sig_off,G,ts,targets)
    means2=tf.einsum('kl,il->ik',B,means)
    variances2=tf.einsum('kl,ilm,nm -> ikn',B,variances,B)
    return means2.numpy(),variances2.numpy()

def posterior(ts,x,idxs,targets,N,R,B,Lambda):
    '''
    Input:
    - ts, a set of times
    - x: nchain x n, representing X(ts[0]),X(ts[1]),X(ts[2])...
    - targets: a set of times
    - N: ell x ell
    - H: ell x ell
    - B: n x ell
    - Lambda: ell x ell

    Output:
    - mean:  E[Z(targets[i]) |X] for each i
    - variance: Cov(Z(targets[i]) |X) for each i
    - covariance: Cov(Z(targets[i]) |X)) for each i

    under the model
      Z~PEG(N,R)
      X[i]|Z ~ N(B Z(t[i]), Lambda Lambda^T)
    '''
    ts,x,idxs,G,Sig = preprocess_numpy(ts,x,N,R,B,Lambda)
    targets=tf.convert_to_tensor(targets,dtype=tf.float64)
    assert len(targets.shape)==1

    distrinfo = txigbs2distrs(ts,x,idxs,G,B,Sig)
    mean,Sig_diag,Sig_off=distrinfo.insample_posterior()
    means,variances=PEG_intercast(mean,Sig_diag,Sig_off,G,ts,targets)
    return means.numpy(),variances.numpy()


r'''
 _          _
| |__   ___| |_ __   ___ _ __ ___
| '_ \ / _ \ | '_ \ / _ \ '__/ __|
| | | |  __/ | |_) |  __/ |  \__ \
|_| |_|\___|_| .__/ \___|_|  |___/
             |_|
'''

def gaussian_stitch(mean1,cov1,mean2,cov2):
    '''
    Input:
    - mean1: ... x n
    - cov1:  ... x n x n
    - mean2: ... x m
    - cov2:  ... x m x m

    with m<n

    Output:
      E_q[Y], cov_q(Y)

    where

      q(x,y) = p2(x) p1(y|x)
      p1(x,y)=N(mean1,cov1)
      p2(x)=N(mean2,cov2)
      x in R^m
      y in R^(m-n)
    '''

    n=cov1.shape[-1]
    m=cov2.shape[-1]
    SLX=slice(0,m)
    SLY=slice(m,None)

    # E_p1[Y|X] = mean1 + mean_transformer @ x
    # mean_transformer = Cov_p1(Y,X) @ (Cov_p1(X,X))^-1
    mean_transformer = tf.matmul(cov1[...,SLY,SLX],tf.linalg.inv(cov1[...,SLX,SLX]))

    # E_q[Y] = mean1 + mean_transformer @ mean2
    mean = mean1[...,SLY] + (mean_transformer@mean2[...,None])[...,0]

    # Cov_p1[Y|X] = Var_p1(Y) - Cov_p1(Y,X) @ (Cov_p1(X,X))^-1 @ Cov_p1(X,Y)
    condcov = cov1[...,SLY,SLY] - tf.matmul(cov1[...,SLY,SLX],tf.linalg.solve(cov1[...,SLX,SLX],cov1[...,SLX,SLY]))

    # Cov_q[Y] = Cov_p1[Y|X] + mean_transformer @ Cov_p2(X) @ mean_transformer.T
    fullcov = condcov + tf.matmul(tf.matmul(mean_transformer,cov2),mean_transformer,transpose_b=True)

    return mean,fullcov

def PEG_forecast(eG,mean2,cov2):
    '''
    Input:
    - eG:    ... x n x n
    - mean2: ... x n
    - cov2:  ... x n

    Output:
      E_q[y], cov_q(y)

    under the model

      q(x,y,z) = p2(x,z)p1(y|x,z)
      p1(z,y) = N(0,[[I,eG^T],[eG,I]])
      p2(z) = N(mean2,cov2)


    '''

    n=eG.shape[-1]
    nones=(None,)*(len(eG.shape)-2)
    I = tf.linalg.eye(n,dtype=eG.dtype)[nones]

    p1_Sig=blockup4(
        I,tf.transpose(eG),
        eG,I
    )
    p1_mean = tf.zeros(n*2,dtype=eG.dtype)[nones]

    return gaussian_stitch(p1_mean,p1_Sig,mean2,cov2)

def PEG_interpolate(eG1,eG2,mean2a,cov2a,mean2b,cov2b,crosscov):
    '''
    Input:
    - d:     ...
    - eG1:   ... x n x n
    - eG2:   ... x n x n
    - mean2a: ... x n
    - cov2a:  ... x n x n
    - mean2b: ... x n
    - cov2b:  ... x n x n
    - crosscov:  ... x n x n

    Output:
      E_q[y], cov_q(y)

    under the model

      q(x,y,z) = p2(x,z)p1(y|x,z)
      p1(x,y,z) = N(0,[[I,eG1^T,(eG2@eG2)^T],[eG1,I,eG2^T],[eG1@eG2,eG2,I])
      p2(x,z) = N([mean2a,mean2b],[[cov2a,crosscov.T],[crosscov,cov2b]])

    '''

    n=eG1.shape[-1]
    nones=(None,)*(len(eG1.shape)-2)
    I = tf.linalg.eye(n,dtype=eG1.dtype)[nones]

    eG3 = eG1@eG2
    p1_Sig=blockup9(
        I,tf.transpose(eG3),tf.transpose(eG1),
        eG3,I,eG2,
        eG1,tf.transpose(eG2),I,
    )
    p1_mean = tf.zeros(n*3,dtype=eG1.dtype)[nones]

    p2_cov = blockup4(
        cov2a,tf.transpose(crosscov),
        crosscov,cov2b
    )
    p2_mean = tf.concat([mean2a,mean2b],axis=0)

    m,v=gaussian_stitch(p1_mean,p1_Sig,p2_mean,p2_cov)

    return m,v

def blockup4(a,b,c,d):
    '''
    Input:
    a: ... x n1 x n2
    b: ... x n1 x n3
    c: ... x n4 x n2
    d: ... x n4 x n3

    Output:
    [
        [a b],
        [c,d]
    ]
    suitably batched
    '''

    ab = tf.concat([a,b],axis=-1)
    cd = tf.concat([c,d],axis=-1)

    abcd = tf.concat([ab,cd],axis=-2)

    return abcd

def blockup9(a,b,c,d,e,f,g,h,i):
    '''
    Output:
    [
        [a b c],
        [d e f],
        [g h i]
    ]
    suitably batched
    '''

    abc = tf.concat([a,b,c],axis=-1)
    cde = tf.concat([d,e,f],axis=-1)
    fgh = tf.concat([g,h,i],axis=-1)

    rez = tf.concat([abc,cde,fgh],axis=-2)

    return rez

def allclose(x,y,eps=1e-10):
    return tf.reduce_all(tf.abs(x-y)<eps)

def PEG_intercast(mean,Sig_diag,Sig_off,G,ots,ts,thresh=1e-10):
    '''
    interpolation and forecasting for PEGGPs

    Input:
    - mean:        nchain x n
    - Sig_diag:    nchain x n x n
    - Sig_off:     nchain-1 x n x n
    - G:           n x n
    - ots:         nchain  -- an ordered vector of times
    - ts:          m -- a vector of times

    Output:
    - mean:  E_q[Z(ts[i])] for each i
    - variance: Cov_q(Z(ts[i]) for each i

    under the model
        Y=Z(ots[0]),Z(ots[1]),Z(ots[2])...
        q(Y,Z) = p2(Y) p1(Z|Y)
        p2(Y) is a Markov process with
            - Y[i] ~ N(mean[i],Sig_diag[i])
            - Cov(Y[i],Y[i+1]) = Sig_off[i]
        p1(Z) = PEGGP(Z;G)
    '''

    # collect stuff we'll need
    Gval,Gvec=tf.linalg.eig(G)
    Gveci = tf.linalg.inv(Gvec)
    blocksize=Sig_diag.shape[1]
    I = tf.linalg.eye(blocksize,dtype=mean.dtype)
    assert tf.reduce_all(ots[1:]-ots[:-1]>0) # make sure ots is in order

    # how do the new positions fall relative to the old locations?
    new_loc_idxs = tf.searchsorted(ots,ts)  # ts[i] is close to ots[new_loc_idxs[i]]

    # calculate...
    means=[]
    variances=[]
    import tqdm
    # for i,idx in enumerate(tqdm.notebook.tqdm(new_loc_idxs)):
    for i,idx in enumerate(new_loc_idxs):
        if idx==0:
            # forecasting backwards
            if allclose(ts[i],ots[0]): # actually we just want the first value:
                m,v=mean[0],Sig_diag[0]
            else:
                df=(ots[0]-ts[i])
                eG=tf.transpose(constructions.calc_eG1(Gval,Gvec,Gveci,df))
                m,v=PEG_forecast(eG,mean[0],Sig_diag[0])
        elif idx==ots.shape[0]:
            # forecasting forward
            if allclose(ts[i],ots[-1]): # actually we just want the last value
                m,v=mean[-1],Sig_diag[-1]
            else:
                df=(ts[i]-ots[-1])
                eG=constructions.calc_eG1(Gval,Gvec,Gveci,df)
                m,v=PEG_forecast(eG,mean[-1],Sig_diag[-1])
        else:
            # interpolating
            if allclose(ts[i],ots[-1]): # we just want that value
                m,v=mean[idx],Sig_diag[idx]
            else:
                df1=(ts[i]-ots[idx-1])
                df2=(ots[idx]-ts[i])
                eG1=constructions.calc_eG1(Gval,Gvec,Gveci,df1)
                eG2=constructions.calc_eG1(Gval,Gvec,Gveci,df2)
                m,v=PEG_interpolate(eG1,eG2,
                    mean[idx-1],Sig_diag[idx-1],
                    mean[idx],Sig_diag[idx],
                    Sig_off[idx-1])

        means.append(m)
        variances.append(v)

    return tf.stack(means,axis=0),tf.stack(variances,axis=0)

@dataclasses.dataclass
class DistrInfo:
    '''
    Collates some useful information about a model of the form

    Z ~ PEG(G)
    X[i] ~ N(B[i] Z[ts[idxs[i]],Sig[i])

    after observing X
    '''

    xs:           'observation values'
    J_dblocks:    'prior precision diagonal blocks'
    J_offblocks:  'prior precision offdiagonal blocks'
    JT_dblocks:   'posterior precision diagonal blocks'
    JT_offblocks: 'posterior precision off-diagonal blocks'
    JT_offset:    'posterior offset'
    Sig:          'Sig'
    Sigix:        'Sig multiplied by x'

    def insample_posterior(self):
        JT_decomp=cr.decompose(self.JT_dblocks,self.JT_offblocks)
        posterior_mean=cr.solve(JT_decomp,self.JT_offset)
        cov_dblocks,cov_offblocks=cr.inverse_blocks(JT_decomp)
        return posterior_mean,cov_dblocks,cov_offblocks

    def log_likelihood(self):
        # get marginal likelihood of x from prior and posterior distribution info
        J_decomp= cr.decompose(self.J_dblocks,self.J_offblocks)
        Jdet = cr.det(J_decomp)
        postmahal,JTdet = cr.mahal_and_det(self.JT_dblocks,self.JT_offblocks,self.JT_offset)
        ldets = tf.reduce_mean(tf.linalg.slogdet((2*np.pi)*self.Sig)[1])*tf.cast(self.xs.shape[0],tf.float64)
        fwdmahal = tf.reduce_sum(self.Sigix*self.xs)
        return .5*(Jdet-JTdet - ldets - fwdmahal +postmahal)

def txigbs2distrs(ts,xs,idxs,G,B,Sig):
    '''
    Input:
    - ts   -- m'         -- observations are at ts
    - xs   -- m x n      -- observations
    - idxs -- m          -- xs[i] is observed at ts[idxs[i]]
    - G    -- ell x ell  -- peg generator
    - Bs   -- n x ell
    - Sig -- n x n

    Output: a DistrInfo object
    '''

    nobs=xs.shape[0]
    nchain=ts.shape[0]
    n=B.shape[0]
    ell=B.shape[1]

    # get Jd
    J_dblocks,J_offblocks = constructions.exponentiate_generator(ts,G)

    # get JT
    Sigix = tf.transpose(tf.linalg.solve(Sig,tf.transpose(xs))) # <-- m x n
    Sigib = tf.linalg.solve(Sig,B) # <-- n x ell
    offset_adder = tf.einsum('nl,mn->ml',B,Sigix) # <-- m x ell
    JT_offset = tf.scatter_nd(idxs[:,None],offset_adder,shape=(ts.shape[0],G.shape[0])) # m' x ell
    prec_adder = tf.einsum('nk,nl->kl',B,Sigib)# <-- ell x ell
    weights=tf.scatter_nd(idxs[:,None],tf.ones(nobs,dtype=J_dblocks.dtype),(nchain,))
    JT_offblocks = J_offblocks
    JT_dblocks = J_dblocks + prec_adder[None,:,:]*weights[:,None,None]

    return DistrInfo(xs,J_dblocks,J_offblocks,JT_dblocks,JT_offblocks,JT_offset,Sig,Sigix)
