import tensorflow as tf
from . import constructions
from . import cr
import numpy as np

def log_likelihood_regular(delta,x,N,R,B,Lambda):
    '''
    Input:
    - delta: scalar
    - x: nchain x n, representing X(0),X(delta),X(2*delta)...
    - N: ell x ell
    - H: ell x ell 
    - B: n x ell
    - Lambda: ell x ell

    Output: the marginal log likelihood
        log p(
           X(0) = x[0],
           X(delta) = x[1],
           X(2*delta) = x[2],
           ...
        )
        under the model (Z,X)~LEGGP(N,R,B,Lambda).
    '''

    # construct the prior
    dblocks,offblocks = constructions.PEGSigi_regular(x.shape[0],delta,N,R)

    # get the likelihood 
    return log_leg_likelihood_fromblocks(x,dblocks,offblocks,B,Lambda)


def log_likelihood_irregular(ts,x,idxs,N,R,B,Lambda):
    '''
    Input:
    - ts: list of times
    - x: nobs x n, representing X(ts[idxs[0]]),X(ts[idxs[1]]),X(ts[idxs[2]])...
    - idxs: nobs x n
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

    # construct the prior
    dblocks,offblocks = constructions.PEGSigi_irregular(ts,N,R)

    # get the likelihood 
    return log_leg_likelihood_fromblocks_irregular(
        x,idxs,dblocks,offblocks,B,Lambda)

def leg_log_likelihood(ts,x,N,R,B,Lambda):
    '''
    Input:
    - ts: list of times
    - x: nobs x n, representing X(ts[0]),X(ts[1]),X(ts[2])...
    - idxs: nobs x n
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

    ts2,idxs=constructions.dedup_ts(ts)

    # construct the prior
    dblocks,offblocks = constructions.PEGSigi_irregular(ts2,N,R)

    # get the likelihood 
    return log_leg_likelihood_fromblocks_irregular(
        x,idxs,dblocks,offblocks,B,Lambda).numpy()


def insample_posterior_regular(delta,x,N,R,B,Lambda):
    '''
    Input:
    - ts: list of times
    - x: nchain x n, representing X(0),X(delta),X(2*delta)...
    - N: ell x ell
    - H: ell x ell 
    - B: n x ell
    - Lambda: ell x ell

    Output:
    - mean:  E[Z(delta*i) |X] for each i
    - variance: Cov(Z(delta*i) |X) for each i
    - covariance: Cov(Z(delta*i),Z(delta*(i+1)) |X) for each i

    under the model (Z,X)~LEGGP(N,R,B,Lambda).
    '''

    # construct the prior precision
    dblocks,offblocks = constructions.PEGSigi_regular(x.shape[0],delta,N,R)

    # get the posterior
    mean,Sig_diag,Sig_off= insample_posterior_fromblocks(x,dblocks,offblocks,B,Lambda)

    return mean,Sig_diag,Sig_off

def insample_posterior_irregular(ts,x,N,R,B,Lambda):
    '''
    Input:
    - delta: scalar
    - x: nchain x n, representing X(ts[0]),X(ts[1]),X(ts[2])...
    - N: ell x ell
    - H: ell x ell 
    - B: n x ell
    - Lambda: ell x ell

    Output:
    - mean:  E[Z(delta*i) |X] for each i
    - variance: Cov(Z(delta*i) |X) for each i
    - covariance: Cov(Z(delta*i),Z(delta*(i+1)) |X) for each i

    under the model (Z,X)~LEGGP(N,R,B,Lambda).
    '''

    ts2,idxs=constructions.dedup_ts(ts)

    # construct the prior precision
    dblocks,offblocks = constructions.PEGSigi_irregular(ts2,N,R)

    # get the posterior
    mean,Sig_diag,Sig_off= insample_posterior_fromblocks_irregular(x,idxs,dblocks,offblocks,B,Lambda)

    return mean,Sig_diag,Sig_off,ts2,idxs

def outsample_posterior_regular(delta,x,N,R,B,Lambda,ts):
    '''
    Input:
    - delta: scalar
    - x: nchain x n, representing X(0),X(delta),X(2*delta)...
    - N: ell x ell
    - H: ell x ell 
    - B: n x ell
    - Lambda: ell x ell
    - ts: a series of positions

    Output: 
    - mean:  E[Z(ts[i]) |X] for each i
    - variance: Cov(Z(ts[i]) |X) for each i

    under the model (Z,X)~LEGGP(N,R,B,Lambda)
    '''

    # get in-sample distribution of Z|X
    dblocks,offblocks = constructions.PEGSigi_regular(x.shape[0],d,N,R)
    mean,Sig_diag,Sig_off=insample_posterior_fromblocks(x,dblocks,offblocks,B,Lambda)

    # extrapolate to out-of-sample distribution of Z|X
    ots = tf.range(0,x.shape[0],delta)
    return PEG_intercast(mean,Sig_diag,Sig_off,N,R,ots,ts)

def posterior_predictive_distributions_regular(delta,x,N,R,B,Lambda,ts):
    '''
    Input:
    - delta: scalar
    - x: nchain x n, representing X(0),X(delta),X(2*delta)...
    - N: ell x ell
    - H: ell x ell 
    - B: n x ell
    - Lambda: ell x ell
    - ts: a series of positions

    Output: 
    - mean:  E[Y(ts[i]) |X] for each i
    - variance: Cov(Y(ts[i]) |X) for each i

    under the posterior predictive model
    - (Z,X)~LEGGP(N,R,B,Lambda)
    - Y(t) ~ N(BZ(t),Lambda Lambda^T)
    '''

    # get distribution of Z|X
    means,variances=outsample_posterior_regular(delta,x,N,R,B,Lambda,ts)

    # get distribution of X|Z
    means2=tf.einsum('kl,il->ik',B,means)
    variances2=tf.einsum('kl,ilm,nm -> ikn',B,variances,B) + (Lambda@tf.transpose(Lambda))[None]

    return means2,variances2

def posterior_predictive(ts,x,targets,N,R,B,Lambda):
    ts = tf.convert_to_tensor(ts,dtype=tf.float64)
    targets = tf.convert_to_tensor(targets,dtype=tf.float64)
    x = tf.convert_to_tensor(x,dtype=tf.float64)

    mean,Sig_diag,Sig_off,ts2,idxs=insample_posterior_irregular(ts,x,N,R,B,Lambda)
    means,variances=PEG_intercast(mean,Sig_diag,Sig_off,N,R,ts2,targets)
    means2=tf.einsum('kl,il->ik',B,means)
    variances2=tf.einsum('kl,ilm,nm -> ikn',B,variances,B) 
    return means2.numpy(),variances2.numpy()
   
def posterior(ts,x,targets,N,R,B,Lambda):
    ts = tf.convert_to_tensor(ts,dtype=tf.float64)
    targets = tf.convert_to_tensor(targets,dtype=tf.float64)
    x = tf.convert_to_tensor(x,dtype=tf.float64)

    mean,Sig_diag,Sig_off,ts2,idxs=insample_posterior_irregular(ts,x,N,R,B,Lambda)
    means,variances=PEG_intercast(mean,Sig_diag,Sig_off,N,R,ts2,targets)
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

def PEG_intercast(mean,Sig_diag,Sig_off,N,R,ots,ts,thresh=1e-10):
    '''
    interpolation and forecasting for PEGGPs

    Input:
    - mean:        nchain x n
    - Sig_diag:    nchain x n x n
    - Sig_off:     nchain-1 x n x n
    - N:           n x n
    - H:           n x n
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
        p1(Z) = PEGGP(Z;N,R)
    '''

    # collect stuff we'll need
    G = constructions.calc_G(N,R)
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

def insample_posterior_fromblocks(x,dblocks,offblocks,B,Lambda):
    '''
    Input:
    - x: nchain x n
    - dblocks,offblocks denote the sparse representation of the inverse
      covariance of Z(t1),Z(t2),...Z(tm) when Z~PEGGP(N,R) for some (N,R).
    - B, Lambda: matrices

    Output:
    - mean:  E[Z(ti) |X] for each i
    - variance: Cov(Z(ti) |X) for each i

    under the model (Z,X)~LEGGP(N,R,B,Lambda).
    '''

    LLt = constructions.calc_LambdaLambdat(Lambda)

    BtLambdaiB = tf.transpose(B)@tf.linalg.solve(LLt,B)
    Lambdaix = tf.transpose(tf.linalg.solve(LLt,tf.transpose(x))) # nchain x m
    posterior_precision_dblocks = dblocks + BtLambdaiB[None]
    posterior_offset = tf.transpose(tf.transpose(B)@tf.transpose(Lambdaix))
    posterior_decomp = cr.decompose(posterior_precision_dblocks,offblocks)

    mean = cr.solve(posterior_decomp,posterior_offset)
    Sig_diag,Sig_off = cr.inverse_blocks(posterior_decomp)

    return mean,Sig_diag,Sig_off

def insample_posterior_fromblocks_irregular(x,idxs,dblocks,offblocks,B,Lambda):
    '''
    Input:
    - x: nobs x n
    - idxs: nobs
    - dblocks,offblocks denote the sparse representation of the inverse
      covariance of Z(t1),Z(t2),...Z(tm) when Z~PEGGP(N,R) for some (N,R).
    - B, Lambda: matrices

    Output:
    - mean:  E[Z(ti) |X] for each i
    - variance: Cov(Z(ti) |X) for each i

    under the model
          Z~PEGGP(N,R)
          Xi ~ N(B Z(ts[idxs[i]), Lambda Lambda^T)
    '''

    LLt = constructions.calc_LambdaLambdat(Lambda)

    nobs,m=x.shape
    nchain=dblocks.shape[0]

    BtLambdaiB = tf.transpose(B)@tf.linalg.solve(LLt,B)
    Lambdaix = tf.transpose(tf.linalg.solve(LLt,tf.transpose(x))) # nchain x m
    Lambdaix_sumup = tf.scatter_nd(idxs[:,None],Lambdaix,(nchain,m))
    weights=tf.scatter_nd(idxs[:,None],tf.ones(nobs,dtype=dblocks.dtype),(nchain,))
    posterior_precision_dblocks = dblocks + BtLambdaiB[None]*weights[:,None,None]
    posterior_offset = tf.transpose(tf.transpose(B)@tf.transpose(Lambdaix_sumup))
    posterior_decomp = cr.decompose(posterior_precision_dblocks,offblocks)

    mean = cr.solve(posterior_decomp,posterior_offset)
    Sig_diag,Sig_off = cr.inverse_blocks(posterior_decomp)

    return mean,Sig_diag,Sig_off

def log_leg_likelihood_fromblocks(x,dblocks,offblocks,B,Lambda):
    '''
    Input:
    - x: nchain x n
    - dblocks,offblocks denote the sparse representation of the inverse
      covariance of Z(t1),Z(t2),...Z(tm) when Z~PEGGP(N,R) for some (N,R).
    - B, Lambda: matrices

    Output: the marginal log likelihood
        log p(
           X(t0) = x[0],
           X(t1) = x[1],
           X(t2) = x[2],
           ...
        )
        under the model (Z,X)~LEGGP(N,R,B,Lambda).
    '''
    nchain,n,n=dblocks.shape
    m=B.shape[0]

    LLt = constructions.calc_LambdaLambdat(Lambda)

    ############## MAHAL ################
    # MAHAL PART I,   <x | (LLt)^-1 | x>
    Lambdaix = tf.transpose(tf.linalg.solve(LLt,tf.transpose(x))) # nchain x m
    mahal1 = tf.reduce_sum(x*Lambdaix)

    # MAHAL PART II,  <B^T LLt^-1 x | (J + B^T LLt^-1 B)^-1 | B^T LLt^-1 x >
    BtLambdaiB = tf.transpose(B)@tf.linalg.solve(LLt,B)
    dblocks2 = dblocks + BtLambdaiB[None]

    BTLambdaix = tf.transpose(tf.transpose(B)@tf.transpose(Lambdaix))
    post_decomp = cr.decompose(dblocks2,offblocks)
    mahal2 = cr.mahal(post_decomp,BTLambdaix)

    # COLLECT
    mahal = -.5*(mahal1 - mahal2)

    ############## DET ################

    ldet = nchain*tf.linalg.slogdet(LLt)[1]
    prior_decomp= cr.decompose(dblocks,offblocks)
    sigdet = -cr.det(prior_decomp)
    postdet = cr.det(post_decomp)
    twopidet = np.log(2*np.pi)*nchain*m

    det = -.5*(ldet + sigdet + postdet+twopidet)

    ############ collect ###########

    return mahal+det


def log_leg_likelihood_fromblocks_irregular(x,idxs,dblocks,offblocks,B,Lambda):
    '''
    Input:
    - x: nobs x n
    - idxs: nobs x n
    - dblocks,offblocks denote the sparse representation of the inverse
      covariance of Z(ts[0]),Z(ts[1]),... when Z~PEGGP(N,R) for some (N,R).
    - B, Lambda: matrices

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
    nobs=x.shape[0]
    nchain,n,n=dblocks.shape
    m=B.shape[0]

    LLt = constructions.calc_LambdaLambdat(Lambda)

    ############## MAHAL ################
    # MAHAL PART I,   <x | (LLt)^-1 | x>
    Lambdaix = tf.transpose(tf.linalg.solve(LLt,tf.transpose(x))) # nchain x m
    mahal1 = tf.reduce_sum(x*Lambdaix)

    # MAHAL PART II,  <B^T LLt^-1 x | (J + B^T LLt^-1 B)^-1 | B^T LLt^-1 x >
    weights=tf.scatter_nd(idxs[:,None],tf.ones(nobs,dtype=dblocks.dtype),(nchain,))
    BtLambdaiB = tf.transpose(B)@tf.linalg.solve(LLt,B)
    dblocks2 = dblocks + BtLambdaiB[None]*weights[:,None,None]

    Lambdaix_sumup = tf.scatter_nd(idxs[:,None],Lambdaix,(nchain,m))
    BTLambdaix = tf.transpose(tf.transpose(B)@tf.transpose(Lambdaix_sumup))
    post_decomp = cr.decompose(dblocks2,offblocks)
    mahal2 = cr.mahal(post_decomp,BTLambdaix)

    # COLLECT
    mahal = -.5*(mahal1 - mahal2)

    ############## DET ################

    ldet = nobs*tf.linalg.slogdet(LLt)[1]
    prior_decomp= cr.decompose(dblocks,offblocks)
    sigdet = -cr.det(prior_decomp)
    postdet = cr.det(post_decomp)
    twopidet = np.log(2*np.pi)*nchain*m

    det = -.5*(ldet + sigdet + postdet+twopidet)

    ############ collect ###########

    return mahal+det