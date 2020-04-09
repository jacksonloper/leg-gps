import tensorflow as tf
import numpy as np
from . import gradexps

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


def exponentiate_generator(ts,G):
    ds=ts[1:]-ts[:-1]

    # expd = tf.linalg.expm(-.5*G[None]*ds[:,None,None])
    expd=gradexps.expm(G,-.5*ds)
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
    return exponentiate_generator(ts,G)

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
    - ts, some times
    - N, diffusion
    - R, torsion
    - B, n x ell, a matrix
    - Lambda, n x n

    Output: the values of C_LEG(t;N,R,B,Lambda) for each t in ts
    '''

    ts=tf.convert_to_tensor(ts,dtype=tf.float64)
    N=tf.convert_to_tensor(N,dtype=tf.float64)
    R=tf.convert_to_tensor(R,dtype=tf.float64)
    B=tf.convert_to_tensor(B,dtype=tf.float64)
    Lambda=tf.convert_to_tensor(Lambda,dtype=tf.float64)

    Sig=C_PEG(ts,N,R)
    LLt = calc_LambdaLambdat(Lambda)

    sig2 = tf.einsum('ijk,aj,bk->iab',Sig,B,B)
    stooge_term = tf.cast(ts==0,dtype=N.dtype)[:,None,None]*LLt[None,:,:]

    return (sig2+stooge_term).numpy()

def B_C_PEG_BT(ts,N,R,B,**kwargs):
    '''
    Input:
    - ts, some times
    - N, diffusion
    - R, torsion
    - B, n x ell, a matrix

    Output: the values of B @ C_PEG(t;N,R) @ B.T for each t in ts
    '''

    ts=tf.convert_to_tensor(ts,dtype=tf.float64)
    N=tf.convert_to_tensor(N,dtype=tf.float64)
    R=tf.convert_to_tensor(R,dtype=tf.float64)
    B=tf.convert_to_tensor(B,dtype=tf.float64)
    Sig=C_PEG(ts,N,R)
    sig2 = tf.einsum('ijk,aj,bk->iab',Sig,B,B)
    return sig2.numpy()


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

    return tf.cast(ts,tf.float64),idxs

def from_celerite(a,b,c,d):
    '''
    finds N,R,B such that

        LEG(tau;N,R,B,0) = a*cos(d*tau)*exp(-c*tau) + b*sin(d*tau)*exp(-c*tau)

    '''

    assert np.abs(b*d)<a*c,'not positive definite'

    b=b/a

    N1=np.sqrt(2*c-2*b*d)
    Rv = np.sqrt(2*c**2 + 4*d**2 + 2*b**2*d**2)
    No = np.sqrt(c+b*d)
    N2 = No

    N=np.r_[N1,0,No,N2].reshape((2,2))
    R=np.r_[0,Rv,0,0].reshape((2,2))
    B=np.r_[np.sqrt(a),0]
    return N,R,B

def test_from_celerite(N):
    import tqdm.notebook
    import numpy.random as npr
    import numpy as np
    import scipy as sp
    import scipy.linalg
    def genexample():
        while True:
            a,b,c,d=npr.randn(4)
            a=np.abs(a)+.01
            c=np.abs(c)+.01

            if np.abs(b*d)<a*c:
                return a,b,c,d

    for i in tqdm.notebook.tqdm(range(N)):
        a,b,c,d=genexample()
        N,R,B = from_celerite(a,b,c,d)
        G=N@N.T + R-R.T
        tau=npr.randn()**2

        Ccel=a*np.cos(d*tau)*np.exp(-c*tau) + b*np.sin(d*tau)*np.exp(-c*tau)
        Cleg=B@sp.linalg.expm(-G*tau/2)@B.T

        assert np.allclose(Ccel,Cleg)
