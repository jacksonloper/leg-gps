__all__ = ['decompose', 'mahal','mahal_and_det','solve','sample','inverse_blocks']
import tensorflow as tf

r'''
             _     _ _                    _
 _ __  _   _| |__ | (_) ___    __ _ _ __ (_)
| '_ \| | | | '_ \| | |/ __|  / _` | '_ \| |
| |_) | |_| | |_) | | | (__  | (_| | |_) | |
| .__/ \__,_|_.__/|_|_|\___|  \__,_| .__/|_|
|_|                                |_|
'''

# @tf.function(autograph=False)
def decompose(Rs,Os):
    '''
    Let J denote a symmetric positive-definite block-tridiagonal matrix whose
    - diagonal blocks are given by Rs
    - lower off-diagonal blocks are given by Os

    We here obtain a cyclic reduction representation of J which can be used
    for further analysis.
    '''

    ms=[]
    Ds=[]
    Fs=[]
    Gs=[]

    while Rs.shape[0]>1:
        # do the work
        (nchain,D1,F,G),(Rs,Os) = _decompose_workhorse(Rs,Os)

        # collect the information
        ms=ms+[nchain]
        Ds=Ds+[D1]
        Fs=Fs+[F]
        Gs=Gs+[G]

    ms=ms+[1]
    Ds=Ds+[tf.linalg.cholesky(Rs)]

    return tf.stack(ms,axis=0),Ds,Fs,Gs

def mahal_and_det(Rs,Os,x):
    '''
    Let J denote a symmetric positive-definite block-tridiagonal matrix whose
    - diagonal blocks are given by Rs
    - lower off-diagonal blocks are given by Os

    returns
    - mahal: x J^-1 x
    - det: |J|

    We here obtain a cyclic reduction representation of J which can be used
    for further analysis.
    '''

    ms=[]
    Ds=[]
    Fs=[]
    Gs=[]

    det=0
    mahal=0

    ytilde=x

    while Rs.shape[0]>1:
        # do the work
        (nchain,D1,F,G),(Rs,Os) = _decompose_workhorse(Rs,Os)

        # det
        det+=tf.reduce_sum(tf.math.log(tf.linalg.diag_part(D1)))

        # process the even entries
        y=ytilde[::2]
        newx = tf.linalg.triangular_solve(D1,y[...,None])[...,0]
        mahal+= tf.reduce_sum(newx**2)

        # recurse on the odd entries
        ytilde = ytilde[1::2] - Ux(F,G,newx)

    # finish it off
    D1=tf.linalg.cholesky(Rs)
    det += tf.reduce_sum(tf.math.log(tf.linalg.diag_part(D1)))

    y=ytilde[::2]
    newx = tf.linalg.triangular_solve(D1,y[...,None])[...,0]
    mahal+= tf.reduce_sum(newx**2)

    return mahal,2*det

def det(decomp):
    '''
    get log |J|
    '''

    ms,Ds,Fs,Gs = decomp
    return 2*tf.reduce_sum([tf.reduce_sum(tf.math.log(tf.linalg.diag_part(x))) for x in Ds])

def mahal(decomp,y):
    '''
    get y^T J^-1 y
    '''

    v=halfsolve(decomp,y)
    return tf.reduce_sum(tf.concat(v,axis=0)**2)

def solve(decomp,y):
    '''
    get J^-1 y

    (This actually takes two times longer than evaluating mahal)
    '''

    v=halfsolve(decomp,y)
    w=backhalfsolve(decomp,v)
    return w

def sample(decomp):
    '''
    sample Z ~ N(0,J^-1)
    '''
    ms,Ds,Fs,Gs = decomp

    n=Ds[0].shape[1]

    noise = [tf.random.normal((D.shape[0],n),dtype=D.dtype) for D in Ds]
    return backhalfsolve(decomp,noise)

def inverse_blocks(decomp):
    '''
    returns diagonal and lower off-diagonal blocks of J^-1

    ((Note that J^-1 is NOT sparse, so this doesn't tell you everything
    about the inverse of J))
    '''

    ms,Ds,Fs,Gs = decomp

    Sig_diag=tf.linalg.inv(Ds[-1])
    Sig_diag=tf.matmul(Sig_diag,Sig_diag,transpose_a=True)
    Sig_off = tf.zeros((0,)+Sig_diag.shape[1:],dtype=Sig_diag.dtype)

    for i in range(1,len(Ds)):
        D=Ds[-i-1]; F=Fs[-i]; G=Gs[-i]
        Sig_diag,Sig_off=_inverse_blocks_workhorse(D,F,G,Sig_diag,Sig_off)

    return Sig_diag,Sig_off

r'''
                    _    _
__      _____  _ __| | _| |__   ___  _ __ ___  ___  ___
\ \ /\ / / _ \| '__| |/ / '_ \ / _ \| '__/ __|/ _ \/ __|
 \ V  V / (_) | |  |   <| | | | (_) | |  \__ \  __/\__ \
  \_/\_/ \___/|_|  |_|\_\_| |_|\___/|_|  |___/\___||___/

'''


def _inverse_blocks_workhorse(D,F,G,Sig_diag,Sig_off):
    # invert D
    Di=tf.linalg.inv(D)
    DtiDi=tf.matmul(Di,Di,transpose_a=True)

    # compute U D^-1
    FDi = tf.matmul(F,Di[:len(F)])
    GDi = tf.matmul(G,Di[1:])

    # compute the diagonal and upper-diagonal parts of Sig UD^-1
    SUDi_diag,SUDi_off = SigU(-Sig_diag,-Sig_off,FDi,GDi)

    # compute the diagonal parts of D^-T U^T Sig UD^-1
    UtSUDi_diag = -UtV_diags(FDi,GDi,SUDi_diag,SUDi_off) + DtiDi

    # stitch everything together
    return interleave(UtSUDi_diag,Sig_diag),interleave(SUDi_diag,tf.transpose(SUDi_off,[0,2,1]))

# @tf.function(autograph=False)
def halfsolve(decomp,y):
    '''
    Input
    - decomp, the cyclic reduction representation of a block tridiagonal matrix
      with nchain diagonal blocks, each of size nblock
    - y: nchain x nblock

    returns the value of L^-1 T_n y
    '''

    ms,Ds,Fs,Gs = decomp

    ytilde=y
    xs=[]
    for i in range(ms.shape[0]):
        y=ytilde[::2]

        xs.append(tf.linalg.triangular_solve(Ds[i],y[...,None])[...,0])
        if ytilde.shape[0]>1:
            ytilde = ytilde[1::2] - Ux(Fs[i],Gs[i],xs[-1])
        else:
            break

    return xs

# @tf.function(autograph=False)
def backhalfsolve(decomp,ycrr):
    '''
    Input:
    - decomp, the cyclic reduction representation of a block tridiagonal matrix
      with nchain diagonal blocks, each of size nblock
    - ycrr, the cyclic reduction representation of a vector y:

    returns the value of T_n^T L^-T y
    '''

    ms,Ds,Fs,Gs = decomp

    ytilde=ycrr[-1]
    xs=tf.linalg.triangular_solve(Ds[-1],ytilde[...,None],adjoint=True)[...,0]
    for i in range(1,ms.shape[0]+1):
        if len(ycrr)-i-1>=0:
            ytilde = ycrr[-i-1] -Utx(Fs[-i],Gs[-i],xs)
            xtilde = tf.linalg.triangular_solve(Ds[-i-1],ytilde[...,None],adjoint=True)[...,0]
            xs = interleave(xtilde,xs)
        else:
            break

    return xs

def _decompose_workhorse(Rs,Os):
    nblock=Rs.shape[1]
    nchain=Rs.shape[0]

    # do our part!
    Ks = tf.linalg.cholesky(Rs)
    D1 = Ks[::2]

    Os_1=Os[::2]
    Os_2=Os[1::2]
    Os_1T = tf.transpose(Os_1,[0,2,1])

    N2=Os_1.shape[0]
    F = tf.transpose(tf.linalg.triangular_solve(Ks[::2][:N2],Os_1T),[0,2,1])
    G = tf.transpose(tf.linalg.triangular_solve(Ks[2::2][:N2],Os_2),[0,2,1])
    UUt_diags, UUt_offdiag = UUt(F,G)

    # get the residual
    Rt=Rs[1::2] - UUt_diags
    Ot= -UUt_offdiag

    return (nchain,D1,F,G),(Rt,Ot)


r'''
                                  _
 ___ _   _ _ __  _ __   ___  _ __| |_
/ __| | | | '_ \| '_ \ / _ \| '__| __|
\__ \ |_| | |_) | |_) | (_) | |  | |_
|___/\__,_| .__/| .__/ \___/|_|   \__|
          |_|   |_|

'''

def interleave(a,b):
    '''
    V=np.zeros((a.shape[0]+b.shape[0],)+a.shape[1:])
    V[::2]=a
    V[1::2]=b
    '''
    a_shape=a.shape
    b_shape=b.shape
    n=a_shape[0]
    m=b_shape[0]
    if n<m:
        shp = (n*2,)+a_shape[1:]
        first_part = tf.reshape(tf.stack([a,b[:n]],axis=1),shp)
        last_bit = b[n:]
        return tf.concat([first_part,last_bit],axis=0)
    else:
        shp = (m*2,)+b_shape[1:]
        first_part = tf.reshape(tf.stack([a[:m],b],axis=1),shp)
        last_bit = a[m:]
        return tf.concat([first_part,last_bit],axis=0)

def Ux(diags,offdiags,x):
    '''
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute U@x
    '''

    n=diags.shape[0]
    m=offdiags.shape[0]
    k=diags.shape[1]

    if n==m:
        return tf.einsum('ijk,ik->ij',diags,x[:-1]) + tf.einsum('ijk,ik->ij',offdiags,x[1:])
    else:
        leaf1=tf.einsum('ijk,ik->ij',diags,x)
        leaf2=tf.einsum('ijk,ik->ij',offdiags,x[1:])
        return tf.concat([leaf1[:-1]+leaf2,[leaf1[-1]]],axis=0)

def Utx(diags,offdiags,x):
    '''
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute U.T@x
    '''

    n=diags.shape[0]
    m=offdiags.shape[0]
    k=diags.shape[1]

    if n==m:
        leaf1= tf.einsum('ikj,ik->ij',diags,x)
        leaf2= tf.einsum('ikj,ik->ij',offdiags,x)
        return tf.concat([[leaf1[0]],leaf1[1:]+leaf2[:-1],[leaf2[-1]]],axis=0)
    else:
        leaf1= tf.einsum('ikj,ik->ij',diags,x)
        leaf2= tf.einsum('ikj,ik->ij',offdiags,x[:-1])
        return tf.concat([[leaf1[0]],leaf1[1:]+leaf2],axis=0)

def UUt(diags,offdiags):
    '''
    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute the diagonal and lower-off-diagonal blocks of
    U@U.T
    '''

    n=diags.shape[0]
    m=offdiags.shape[0]

    if n==m:
        tq=tf.einsum('ijk,ilk->ijl',diags,diags)
        tq+=tf.einsum('ijk,ilk->ijl',offdiags,offdiags)

        offdiags=tf.einsum('ijk,ilk->ilj',offdiags[:-1],diags[1:])
        return tq,offdiags

    else:
        leaf1=tf.einsum('ijk,ilk->ijl',diags,diags)
        leaf2=tf.einsum('ijk,ilk->ijl',offdiags,offdiags)
        tq= tf.concat([leaf1[:-1]+leaf2,[leaf1[-1]]],axis=0)

        offdiags=tf.einsum('ijk,ilk->ilj',offdiags,diags[1:])
        return tq,offdiags

def UtV_diags(diags,offdiags,diags2,offdiags2):
    '''
    V is upper didiagonal with (diags2,offdiags2)

    U is upper diagonal with (diags,offdiags)

    We want the diagonal blocks of U.T @ V
    '''

    if diags.shape[0]==offdiags.shape[0]+1:
        return tf.concat([
            [tf.transpose(diags[0])@diags2[0]],
            tf.matmul(diags[1:],diags2[1:],transpose_a=True) + tf.matmul(offdiags,offdiags2,transpose_a=True)
        ],axis=0)
    else:
        return tf.concat([
            [tf.transpose(diags[0])@diags2[0]],
            tf.matmul(diags[1:],diags2[1:],transpose_a=True) + tf.matmul(offdiags[:-1],offdiags2[:-1],transpose_a=True),
            [tf.transpose(offdiags[-1])@offdiags2[-1]]
        ],axis=0)

def SigU(dblocks,offblocks,diags,offdiags):
    '''
    Let Sig be a symmetric block-tridiagonal matrix whose
    - diagonal blocks are dblocks
    - lower off-diagonals are offblocks

    Let U be an upper block-bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We would like to compute block-tridiagonal blocks of Sig @ U
    '''

    if diags.shape[0]==offdiags.shape[0]+1:
        mainline = tf.concat([
            [dblocks[0]@diags[0]],
            tf.matmul(dblocks[1:],diags[1:]) + tf.matmul(offblocks,offdiags)
        ],axis=0)

        upline = tf.matmul(dblocks[:-1],offdiags)+tf.matmul(offblocks,diags[1:],transpose_a=True)

    else:
        mainline = tf.concat([
            [dblocks[0]@diags[0]],
            tf.matmul(dblocks[1:],diags[1:]) + tf.matmul(offblocks,offdiags[:-1])
        ],axis=0)

        upline = tf.concat([
            tf.matmul(dblocks[:-1],offdiags[:-1])+tf.matmul(offblocks,diags[1:],transpose_a=True),
            [dblocks[-1]@offdiags[-1]]
        ],axis=0)


    return mainline,upline

r'''
 _            _   _
| |_ ___  ___| |_(_)_ __   __ _
| __/ _ \/ __| __| | '_ \ / _` |
| ||  __/\__ \ |_| | | | | (_| |
 \__\___||___/\__|_|_| |_|\__, |
                          |___/

'''

def eo(n):
    import numpy as np
    guys1=np.r_[0:n:2]
    guys2=np.r_[1:n:2]
    return np.concatenate([guys1,guys2])

def recursive_eo(n):
    import numpy as np

    if n==1:
        return np.array([0])
    elif n==2:
        return np.array([0,1])
    else:
        guys1=np.r_[0:n:2]
        guys2=np.r_[1:n:2]
        return np.concatenate([guys1,guys2[recursive_eo(len(guys2))]])

def perm2P(perm):
    import numpy as np

    n=len(perm)
    v=np.zeros((n,n))
    for i in range(n):
        v[i,perm[i]]=1
    return v

def make_U(diags,offdiags):
    '''
    Let U be an upper bidiagonal matrix whose
    - diagonals are given by diags
    - upper off-diagonals are given by offdiags

    We return U
    '''
    import numpy as np


    n=diags.shape[0]
    m=offdiags.shape[0]
    k=diags.shape[1]

    if n==m:
        V=np.zeros((n,k,n+1,k))
        for i in range(n):
            V[i,:,i]=diags[i]
        for i in range(n):
            V[i,:,i+1]=offdiags[i]
        return V.reshape((n*k,(n+1)*k))
    else:
        V=np.zeros((n,k,n,k))
        for i in range(n):
            V[i,:,i]=diags[i]
        for i in range(n-1):
            V[i,:,i+1]=offdiags[i]
        return V.reshape((n*k,n*k))

def _test_U_stuff(nblock,n,even):
    import numpy as np
    import numpy.random as npr

    def fl(A):
        return A.reshape((A.shape[0]*A.shape[1],A.shape[2]*A.shape[3]))
    def unfl(A):
        return A.reshape((A.shape[0]//nblock,nblock,A.shape[1]//nblock,nblock))

    A=npr.randn(n,nblock,nblock)
    if even:
        B=npr.randn(n,nblock,nblock)
        x=npr.randn(n+1,nblock)
    else:
        B=npr.randn(n-1,nblock,nblock)
        x=npr.randn(n,nblock)
    y=npr.randn(n,nblock)
    U=make_U(A,B)

    Sig=npr.randn(n*nblock,n*nblock)
    Sig=unfl(Sig@Sig.T)
    dblocks=np.array([Sig[i,:,i] for i in range(len(Sig))])
    offblocks=np.array([Sig[i+1,:,i] for i in range(len(Sig)-1)])

    # is UUt is what it says it is?
    UUt_d,UUt_o=UUt(A,B)
    UUt_full=unfl(U@U.T)
    for i in range(n):
        assert np.allclose(UUt_d[i],UUt_full[i,:,i])
    for i in range(n-1):
        assert np.allclose(UUt_o[i],UUt_full[i+1,:,i])

    # is Ux right?
    assert np.allclose(U@x.ravel(),Ux(A,B,x).numpy().ravel())

    # is UTx right?
    assert np.allclose(U.T@y.ravel(),Utx(A,B,y).numpy().ravel())

    # is tridi right?
    SigU_full = unfl(fl(Sig)@U)
    SigU_mid,SigU_hi = SigU(dblocks,offblocks,A,B)
    assert np.allclose(
        SigU_mid,
        np.array([SigU_full[i,:,i] for i in range(SigU_full.shape[0])])
    )
    assert np.allclose(
        SigU_hi,
        np.array([SigU_full[i,:,i+1] for i in range(SigU_hi.shape[0])])
    )

    # finally, we need to look at U^T SigU
    UTSigU = unfl(U.T @ fl(SigU_full))
    centrals=np.array([UTSigU[i,:,i] for i in range(UTSigU.shape[0])])
    centrals_guess = UtV_diags(A,B,SigU_mid,SigU_hi)
    assert np.allclose(centrals.ravel(),centrals_guess.numpy().ravel())


def test_U_stuff():
    _test_U_stuff(1,4,True)
    _test_U_stuff(1,4,False)
    _test_U_stuff(2,3,True)
    _test_U_stuff(2,3,False)

def maintest():
    import numpy as np
    import numpy.random as npr
    test_U_stuff()

    for nblock in [1,3]:
        for nchain in [2,6,30,31,32,33]:
            sh1=(nchain,nblock,nchain,nblock)
            sh2=(nchain*nblock,nchain*nblock)
            Ld=[npr.randn(nblock,nblock) for i in range(nchain)]
            Lo=[npr.randn(nblock,nblock) for i in range(nchain-1)]
            L=np.zeros(sh1)
            for i in range(nchain):
                L[i,:,i]=Ld[i]+np.eye(nblock)*3
            for i in range(1,nchain):
                L[i,:,i-1]=Lo[i-1]
            L=L.reshape(sh2); J=L@L.T; J=J.reshape(sh1)

            # the slow analysis
            Tm=np.kron(perm2P(recursive_eo(nchain)),np.eye(nblock))
            L=np.linalg.cholesky(Tm@J.reshape(sh2)@Tm.T)

            # the fast analysis
            Rs=np.array([J[i,:,i] for i in range(nchain)])
            Os=np.array([J[i,:,i-1] for i in range(1,nchain)])
            decomp=decompose(Rs,Os)
            ms,Ds,Fs,Gs=decomp

            # check mahalanobis and halfsolve
            v=npr.randn(nchain,nblock)
            mahal=np.mean(v.ravel()*np.linalg.solve(J.reshape(sh2),v.ravel()))
            mahal2=np.mean(np.concatenate(halfsolve(decomp,v))**2)
            assert np.allclose(mahal,mahal2)
            assert np.allclose(np.linalg.solve(L,Tm@v.ravel()),np.concatenate(halfsolve(decomp,v)).ravel())

            # check determinant
            diags= np.concatenate([[np.diag(x) for x in y] for y in Ds])
            det1=2*np.sum(np.log(diags))
            det2=np.linalg.slogdet(J.reshape(sh2))[1]
            assert np.allclose(det1,det2)

            # check backhalfsolve
            vrep=[npr.randn(x,nblock) for x in (np.array(ms)+1)//2]
            v=np.concatenate(vrep)
            rez=np.linalg.solve(L.T@Tm,v.ravel())
            assert np.allclose(backhalfsolve(decomp,vrep).numpy().ravel(),rez)

            # check Sig
            Sig=np.linalg.inv(J.reshape(sh2)).reshape(sh1)
            Sig_diag=np.array([Sig[i,:,i] for i in range(J.shape[0])])
            Sig_off=np.array([Sig[i+1,:,i] for i in range(J.shape[0]-1)])
            guess_diag,guess_off = inverse_blocks(decomp)
            assert np.allclose(guess_diag.numpy().ravel(),Sig_diag.ravel())
            assert np.allclose(guess_off.numpy().ravel(),Sig_off.ravel())
