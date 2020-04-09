import tensorflow as tf

@tf.custom_gradient
def expm(G,ts):
    '''
    Input:
    - G: ell x ell
    - ts: nsamp

    Output:
    - rez: nsamp x ell x ell

    Where 

    rez[i] = exp(G*ts[i])

    '''

    # get decomposition
    Lambda,U = tf.linalg.eig(G)
    Ui = tf.linalg.inv(U)
    UT = tf.transpose(U)
    Ui=tf.linalg.inv(U)
    UiT = tf.transpose(Ui)
    ts = tf.cast(ts,dtype=Lambda.dtype)

    # compute exp(Lambda*t) for each t
    eL = tf.exp(Lambda[None,:]*ts[:,None]) # nsamps x ell

    # get the result
    result = tf.math.real(tf.matmul(U[None,:,:,],eL[:,:,None] * Ui[None,:,:]))

    def grad(M):
        ######################
        # GRAD w.r.t. ts is simple:
        GeG = tf.matmul(G[None],result) # nsamp x ell x ell
        gradt = tf.einsum('ijk,ijk->i',GeG,M)

        ####################
        # GRAD w.r.t G is tricyk
        # get the Phis (this is the only part that depends on t), nsamp x ell x ell
        Phi = (eL[:,:,None] - eL[:, None,:]) / (Lambda[None,:,None] - Lambda[None,None,:]) 
        Phi = tf.linalg.set_diag(Phi,ts[:,None]*eL)

        # take the Hadamards
        M=tf.cast(M,dtype=tf.complex128)
        H = Phi * tf.matmul(UT[None],tf.matmul(M, UiT[None]))

        # reduce
        H = tf.reduce_sum(H,axis=0)

        # sandwitch with Us
        sand = tf.matmul(UiT,tf.matmul(H,UT))

        # take real part
        sandreal = tf.math.real(sand)

        ##########
        return sandreal,gradt

    return result,grad

def test():
    ell=5

    import numpy.random as npr
    import numpy as np

    # check evaluation
    A=tf.convert_to_tensor(npr.randn(ell,ell))
    ts=tf.convert_to_tensor([1.2],dtype=A.dtype)
    eA=expm(A,ts)
    eA_other = tf.linalg.expm(A*ts[0])
    assert np.allclose(eA[0].numpy(),eA_other.numpy())

    # check gradient w.r.t. A
    M=tf.convert_to_tensor(npr.randn(ell,ell))
    with tf.GradientTape() as g:
        g.watch(A)
        rez=tf.reduce_sum(tf.linalg.expm(A*ts[0])*M)
    foo1=g.gradient(rez,A)

    with tf.GradientTape() as g:
        g.watch(A)
        rez=tf.reduce_sum(expm(A,ts)[0]*M)
    foo2=g.gradient(rez,A)
    assert np.allclose(foo1.numpy(),foo2.numpy())

    # check gradient w.r.t T
    M=tf.convert_to_tensor(npr.randn(ell,ell))
    with tf.GradientTape() as g:
        g.watch(ts)
        rez=tf.reduce_sum(tf.linalg.expm(A*ts[0])*M)
    foo1=g.gradient(rez,ts)

    with tf.GradientTape() as g:
        g.watch(ts)
        rez=tf.reduce_sum(expm(A,ts)[0]*M)
    foo2=g.gradient(rez,ts)
    assert np.allclose(foo1.numpy(),foo2.numpy())
