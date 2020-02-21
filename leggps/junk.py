    # start=tf.reduce_min(ots)
    # end =tf.reduce_max(ots)

    # backcasts=(new_loc_idxs==0)
    # forecasts=(new_loc_idxs==ots.shape[0])
    # noncasts =(~backcasts)&(~forecasts)

    # selfcasts=tf.abs(ots[new_loc_idxs[noncasts]] - ts[noncasts]) < thresh
    # interpolations = 


        - scattered_x: nchain x n,  scattered_x[i] = sum_{j: idxs[j]=i} xs[j]
    - weights: nchain, weights[i] = #{j: idxs[j]=i} 1


        if Rt.shape[0]==1:
        flimmy=Rs,Os
        assert tf.reduce_all(tf.linalg.eigvalsh(Rt)>0)



def gradient_expA(A,M):
    '''
    f(A) = sum(A*M)
    
    returns gradient of f with respect to A

    Najfeld, Igor, and Timothy F. Havel. 
    "Derivatives of the matrix exponential and their computation." 
    Advances in applied mathematics 16.3 (1995): 321-375.
    '''
    
    # get info we need
    Lambda,U=tf.linalg.eig(A)
    
    eL = tf.math.exp(Lambda)
    M = tf.cast(M,dtype=tf.complex128)

    # get off-diagonal entries of 
    Phi = (eL[:,None] - eL[None,:]) / (Lambda[:,None] - Lambda[None,:])

    # get diagonal entries
    Phi = tf.linalg.set_diag(Phi,eL)

    # take the Hadamard
    H = Phi * (UT @ M @ UiT)

    # sandwitch with Us
    sand = UiT @ H @ UT

    # take real part
    sandreal = tf.math.real(sand)

    # done!
    return sandreal


@tf.custom_gradient
def expm_many(G,ds):
    '''
    returns expm(-.5*G[None]*ds[:,None,None])
    '''

    eigvals,eigvecs = tf.linalg.eig(G)
    eigvecsi = tf.linalg.inv(eigvecs)

    expvecs = tf.exp(-.5*eigvals[None,:]*ds[:,None]) # nsamps x ndims

    result = tf.matmul(eigvecs[None,:,:,],expvecs[:,None,:] * eigvecsi[None,:,:])

    return result


def gradient_expA(A,M):
    '''
    f(A) = sum(A*M)
    
    returns gradient of f with respect to A

    Najfeld, Igor, and Timothy F. Havel. 
    "Derivatives of the matrix exponential and their computation." 
    Advances in applied mathematics 16.3 (1995): 321-375.
    '''
    
    # get info we need
    Lambda,U=tf.linalg.eig(A)
    UT = tf.transpose(U)
    Ui=tf.linalg.inv(U)
    UiT = tf.transpose(Ui)
    eL = tf.math.exp(Lambda)
    M = tf.cast(M,dtype=tf.complex128)

    # get off-diagonal entries of 
    Phi = (eL[:,None] - eL[None,:]) / (Lambda[:,None] - Lambda[None,:])

    # get diagonal entries
    Phi = tf.linalg.set_diag(Phi,eL)

    # take the Hadamard
    H = Phi * (UT @ M @ UiT)

    # sandwitch with Us
    sand = UiT @ H @ UT

    # take real part
    sandreal = tf.math.real(sand)

    # done!
    return sandreal

def to_celerite(N,R,B):
    '''
    finds a,b,c,d such that 

        LEG(tau;N,R,B,0) = a*cos(d*tau)*exp(-c*tau) + b*sin(d*tau)*exp(-c*tau)
    '''
    
    assert N.shape==(2,2)
    assert R.shape==(2,2)

    G=N@N.T + R-R.T 
    eigvals,eigvecs=np.linalg.eig(G/2)

    if np.allclose(np.imag(eigvals),0):
        c=np.real(eigvals[0])
        d=0
        return dict(type="pair_of_simples",kernel1=(B[0,0]**2,np.real(eigvals[0])
    else:
        c=np.real(eigvals[0])
        d=np.imag(eigvals[0])
        BV1 = B @ eigvecs
        BV2 = B@np.linalg.inv(eigvecs).T
        x= BV1[0,0]*BV2[0,0]
        a=2*np.real(x)
        b=2*np.imag(x)
        
        return dict(type='complex',a=a,b=b,c=c,d=d)

def gradient_expA_alternative(A,M):
    '''
    f(A) = sum(A*M)
    
    returns gradient of f with respect to A

    Najfeld, Igor, and Timothy F. Havel. 
    "Derivatives of the matrix exponential and their computation." 
    Advances in applied mathematics 16.3 (1995): 321-375.
    '''
    
    # get info we need
    Lambda,U=tf.linalg.eig(A)
    Ui=tf.linalg.inv(U)
    eLo2 = tf.math.exp(.5*Lambda)
    M = tf.cast(M,dtype=tf.complex128)

    # get off-diagonal entries 
    df = .5*(Lambda[:,None] - Lambda[None,:])
    Phi = tf.math.sinh(df)/df

    # get diagonal entries
    Phi = tf.linalg.set_diag(Phi,tf.ones((Phi.shape[0]),dtype=tf.complex128))

    # take the Hadamard
    H = eLo2[:,None]*eLo2[None,:]*(Ui@tf.transpose(M)@U) *tf.transpose(Phi)

    # sandwitch with Us
    sand = tf.transpose(Ui) @ tf.transpose(H) @ tf.transpose(U)

    # take real part
    sandreal = tf.math.real(sand)

    # done!
    return sandreal