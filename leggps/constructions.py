import tensorflow as tf
import numpy as np

r'''
DEFINITIONS
to be used in comments below
===========

- G = NN^T + R-R^T
- PEGSig(ts,N,R) is the covariance matrix with len(ts) blocks so that the (i,j)th block is given by 
    EGSig_ij = exp(-.5 |ts[i]-ts[j]| G)      for i>j
    EGSig_ij = exp(-.5 |ts[i]-ts[j]|d G^T)   for i<j
    I                                        for i=0
- PEGSig_regular(n,d,N,R) = EGSig([0,d,2*d,...,(n-1)*d],N,R)
- PEGSigi(ts,N,R) = the matrix inverse of PEGSig(ts,N,R)
- PEGSigi_regular(n,d,N,R) = the matrix inverse of PEGSig_regular(n,d,N,R)
- LEGSig(Sig,B,Lambda) is the covariance matrix so that the (i,j)th block is given by 
    Sig_ij = B Sig[i,j] B^T                        for i!=j
    Sig_ij = B Sig[i,j] B^T + Lambda Lambda^T      for i=0

'''

def calc_LambdaLambdat(Lambda):
    if len(Lambda.shape)==2:
        return Lambda @ tf.transpose(Lambda) + tf.linalg.eye(Lambda.shape[0],dtype=Lambda.dtype)*1e-9
    else:
        return tf.linalg.diag(Lambda**2+1e-9)

def calc_G(N,R):
    return N@tf.transpose(N) + R - tf.transpose(R) + tf.linalg.eye(N.shape[0],dtype=N.dtype)*(1e-5)

def calc_Gvvv(G):
    Gval,Gvec=tf.linalg.eig(G)
    Gveci=tf.linalg.inv(Gvec)
    return Gval,Gvec,Gveci

def calc_eG1(Gval,Gvec,Gveci,d):
    '''
    Input:
    - Gval:  n
    - Gvec:  n x n
    - Gveci: n x n  (=Gvec^-1)
    - d:     scalar

    Output:
    - exp(-.5*d*Gvec diag(Gval) Gveci),  n x n
    '''

    return calc_eG(Gval,Gvec,Gveci,[d])[0]

def calc_eG(Gval,Gvec,Gveci,d):
    '''
    Input:
    - Gval:  n
    - Gvec:  n x n
    - Gveci: n x n  (=Gvec^-1)
    - d:     m

    Output:
    - exp(-.5*d*Gvec diag(Gval) Gveci),  m x n x n
    '''
    d=tf.cast(d,Gval.dtype)

    # GD=tf.matmul(Gvec[None],tf.linalg.diag(tf.math.exp(-.5*d[:,None]*Gval[None,:])))
    GD = Gvec[None] * tf.math.exp(-.5*d[:,None,None]*Gval[None,None,:]) # m x n x n
    GDGi = tf.matmul(GD,Gveci[None])

    return tf.math.real(GDGi)

def PEGSigi_regular(n,d,N,R):
    ''' 
    Input:
    - n, natural number
    - d, postive floating point
    - N, diffusion
    - H, torsion

    Output:
    - dblocks, the diagonal blocks of EGSigi(n,d,G)
    - offblocks, the lower diagonal blocks of EGSigi(n,d,G)
    '''

    G = calc_G(N,R)

    expd=tf.linalg.expm(-.5*d*G)
    expdT=tf.transpose(expd)
    eye  = tf.linalg.eye(expd.shape[0],dtype=expd.dtype)

    '''
    offblock  = -(I - G^T G) ^-1 G^T
    Dcontrib1 = G (I - G^T G) ^-1 G^T
    Dcontrib2 = G^T (I - G G^T) ^-1 G
    '''

    imgtgigt = tf.linalg.solve(eye-expdT@expd,expdT)
    imggtig = tf.linalg.solve(eye-expd@expdT,expd)

    offblock = -imggtig
    Dcontrib1 = expd @ imgtgigt
    Dcontrib2 = expdT @ imggtig
   
    dblocks_inner = tf.tile((eye+Dcontrib1+Dcontrib2)[None],(n-2,1,1))
    dblocks = tf.concat([[eye+Dcontrib2],dblocks_inner,[eye+Dcontrib1]],axis=0)

    offblocks = tf.tile(offblock[None],(n-1,1,1))

    return dblocks,offblocks

def PEGSigi_irregular(ts,N,R):
    ''' 
    Input:
    - ts, a vector
    - N, diffusion
    - H, torsion

    Output:
    - dblocks, the diagonal blocks of EGSigi(ts,N,R)
    - offblocks, the lower diagonal blocks of EGSigi(ts,N,R)
    '''

    G = calc_G(N,R)
    # Gval,Gvec,Gveci=calc_Gvvv(G)

    ds=ts[1:]-ts[:-1]

    # expd=calc_eG(Gval,Gvec,Gveci,ds)  
    expd = tf.linalg.expm(-.5*G[None]*ds[:,None,None])  
    expdT=tf.transpose(expd,[0,2,1])
    eye  = tf.linalg.eye(G.shape[0],dtype=expd.dtype)

    '''
    offblock  = -(I - G^T G) ^-1 G^T
    Dcontrib1 = G (I - G^T G) ^-1 G^T
    Dcontrib2 = G^T (I - G G^T) ^-1 G
    '''

    imgtgigt = tf.linalg.solve(eye[None]-expdT@expd,expdT)
    imggtig = tf.linalg.solve(eye[None]-expd@expdT,expd)

    offblocks = -imggtig
    Dcontrib1 = tf.matmul(expd,imgtgigt)  # Dcontrib1[-1] connects ts[-2] to ts[-1], and isn't applic to 0
    Dcontrib2 = tf.matmul(expdT,imggtig)  # Dcontrib2[0] connects ts[0] to ts[1], and isn't applicable to -1
   
    dblocks_inner = eye[None]+Dcontrib1[:-1]+Dcontrib2[1:]
    dblocks = tf.concat([[eye+Dcontrib2[0]],dblocks_inner,[eye+Dcontrib1[-1]]],axis=0)

    return dblocks,offblocks

def C_PEG(ts,N,R,**kwargs):
    '''
    Input:
    - ts, some times (in increasing order)
    - N, diffusion
    - H, torsion

    Output:
    - cov, the first column of blocks of EGSig(ts,G)
    '''
    G = calc_G(N,R)
    return tf.linalg.expm(-.5*ts[:,None,None]*G[None,:,:]).numpy()

def C_LEG(ts,N,R,B,Lambda):
    '''
    Input:
    - ts, some times (in increasing order)
    - N, diffusion
    - H, torsion
    - B, n x ell, a matrix
    - Lambda, n x n

    Output:
    - cov, the first column of blocks of noised_transformed_Sig(Sig,B,Lambda)
    '''

    Sig=C_PEG(ts,N,R)
    LLt = calc_LambdaLambdat(Lambda)

    sig2 = tf.einsum('ijk,aj,bk->iab',Sig,B,B)
    sig3 = tf.concat([[sig2[0] + LLt],sig2[1:]],axis=0)

    return sig3.numpy()


def dedup_ts(allts,thresh=1e-10,check=True):
    '''
    Input:
    - allts: nobs
    - thresh: scalar

    Output:
    - ts: nchain
    - idxs: nobs
    '''

    diff=allts[1:]-allts[:-1]
    if check:
        assert tf.reduce_all(diff>=0)

    good=tf.concat([[True],diff>thresh],axis=0)
    ts=tf.boolean_mask(allts,good)
    idxs=tf.cumsum(tf.cast(good,dtype=tf.int64))-1

    return ts,idxs