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
