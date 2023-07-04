import os
import sys
import numpy as np

"""
====Two-Step Algorithm====
A dataset X of non-Gaussian variables is required as input, together with a positive value for the penalization parameter, lambda.
Two-Step outputs the causal coefficients matrix B, from X = BX + E.
In B, the causal direction goes from column to row, such that a matrix entry Bij, implies Xj --> Xi
two_step_CD.m runs Two-Step with Adaptative Lasso as a first step.
two_step_CD_mask.m should be used if the adjacency matrix of the first step was computed with some other algorithm.
"""

def pdinv(A):
    """
    PDINV Computes the inverse of a positive definite matrix
    Input:
        A: assuming positive definite square matrix in numpy ndarray (n,n)
    Output:
        Ainv: square matrix in numpy ndarray (n,n)
    translated in python by Sunghwan Kim, KAIST, 2022
    """
    from numpy.linalg import cholesky, svd, inv
    assert A.shape[0] == A.shape[1]
    N = A.shape[0]
    try:
        U = cholesky(A)
        Uinv = inv(U)
        Ainv = Uinv.dot(Uinv.T)
    except:
        print("matrix is not positive definite in pdinv, inverting using SVD.")
        U, S, V = svd(A)
        Ainv = V.dot(np.diag(1/S)).dot(U.T)
    return A

def vcorrcoef(X,y):
  """
  Calculating vectorized correlation coefficients
  Input: 
    X --> (n, k) ndarray
    y --> (n, 1) ndarray
  Output: r --> (k, 1) ndarray 
  """
  Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
  ym = np.mean(y)
  r_num = np.sum((X-Xm)*(y-ym),axis=1)
  r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
  r = r_num/r_den    

  # print(X.shape)
  # print(y.shape)
  # print(r.shape)
  return r



def thresholding(X, thr):
      from numpy import abs, sign
      A = X.copy()
      A[abs(A)<thr] = thr * sign(A[abs(A)<thr]) # thresholding to beta_min2
      return A

def betaAlasso_grad_2step(x, y, var_noise, lam, verbose=False):
      """
      Aim: 
            to find the solution of adaptive Lasso with the given lambda
      Inputs: 
            x is (k,n), and y is (n,). var_noise is the variance of the noise.
      Outputs: 
            beta_al contains the obtained beta after step 1 (conventioanl ALasso). 
            After convergence of step 1, we repeat upadte \hat{beta} and repeat the adaptive Lasso procedure to refine the result.
            beta_new_n contains the penalty term (beta_al/\hat{beta}) after step 1.
            beta2_al and beta2_new_n contain the results after step 2.
      Note that in the first step, a combination of gradient-based methods and Newton method is used.
      by Kun Zhang 2016,2017 Carnegie Mellon University
      translated in python by Sunghwan Kim 2022, KAIST
      """
      from numpy.linalg import inv, norm
      from numpy import abs, matmul, dot, multiply

      # initial value setting
      N, T = x.shape
      var_noise_back = var_noise
      weight = 0.2 # trade_off
      tol = 1E-2 # 1E-10; temp: Aug 5, 17:27
      beta_min = 1E-12
      beta_min2 = 1E-2

      ## STEP 1: conventioanl ALasso
      beta_hat = matmul(dot(x,y), inv(matmul(x, x.T)))
      #print("beta_hat:", beta_hat.shape)
      if var_noise==0: var_noise = np.var(y - dot(x,beta_hat))
      x_new = matmul(np.diag(beta_hat),x) 
      #print("x_new:", x_new.shape)
      beta_new_o = np.ones(N)
      #print("beta_new_o:", beta_new_o.shape)

      # store for curve plotting
      sum_adjust_beta_list, pl_list = [], []
      sum_adjust_beta = np.sum(abs(beta_new_o))
      sum_adjust_beta_list.append(sum_adjust_beta)
      #print("sum_adjust_beta:", sum_adjust_beta)
      y_err = y - matmul(beta_new_o.T, x_new)
      pl = np.squeeze(np.matmul(y_err, y_err.T)/(2*var_noise) + lam * sum_adjust_beta)
      pl_list.append(pl)
      #print("pl:", pl)

      # loop for reducing error
      error = 1
      while error > tol:
            sigma = np.diag(1./np.abs(beta_new_o))
            # with gradient trad-off
            
            beta_new_n = matmul(inv(matmul(x_new,x_new.T) + var_noise * lam * sigma), dot(x_new, y)) * weight + beta_new_o * (1-weight)
            beta_new_n = thresholding(X=beta_new_n, thr=beta_min) # thresholding to beta_min
            error = norm(beta_new_n - beta_new_o)
            beta_new_o = beta_new_n
            sum_adjust_beta = np.sum(abs(beta_new_n))
            sum_adjust_beta_list.append(sum_adjust_beta)
            y_err = y - matmul(beta_new_o.T, x_new)
            pl = np.squeeze(np.matmul(y_err, y_err.T)/(2*var_noise) + lam * sum_adjust_beta)
            pl_list.append(pl)
            if verbose: print("Step1", error)
      beta_new_n = multiply(beta_new_n, abs(beta_new_n)>1E4*beta_min)
      beta_al = multiply(beta_new_n, beta_hat.T) 

      ## STEP 2: 
      idx = np.where(abs(beta_al)>1E4*beta_min)[0]
      N2 = idx.shape[0]
      x2 = x[idx]
      beta2_hat = matmul(dot(x2,y), inv(matmul(x2, x2.T)))
      if var_noise_back==0: var_noise = np.var(y - dot(x2,beta2_hat))
      x2_new = matmul(np.diag(beta2_hat),x2) 
      beta2_new_o = np.ones(N2)
      sum_adjust_beta2_list, pl2_list = [], []
      sum_adjust_beta2 = np.sum(abs(beta2_new_o))
      sum_adjust_beta2_list.append(sum_adjust_beta2)
      #print("sum_adjust_beta2:", sum_adjust_beta2)
      y_err2 = y - matmul(beta2_new_o.T, x2_new)
      pl2 = np.squeeze(np.matmul(y_err2, y_err2.T)/(2*var_noise) + lam * sum_adjust_beta2)
      pl2_list.append(pl2)
      error = 1
      for i in range(100):
            if error<=tol: break
            sigma2 = np.diag(1./np.abs(beta2_new_o))
            # print("x_new:", x_new.shape)
            # print("beta_new_o:", beta_new_o.shape)
            # print("sigma:", sigma.shape)
            # print("var_noise:", var_noise.shape)
            # with gradient trad-off
            beta2_new_n = matmul(pdinv(matmul(x2_new,x2_new.T) + var_noise * lam * sigma2), dot(x2_new, y)) * weight + beta2_new_o * (1-weight)
            beta2_new_n = thresholding(X=beta2_new_n, thr=beta_min) # thresholding to beta_min
            error = norm(beta2_new_n - beta2_new_o)
            beta2_new_o = beta2_new_n
            sum_adjust_beta2 = np.sum(abs(beta2_new_n))
            sum_adjust_beta2_list.append(sum_adjust_beta2)
            y_err2 = y - matmul(beta2_new_o.T, x2_new)
            pl2 = np.squeeze(np.matmul(y_err2, y_err2.T)/(2*var_noise) + lam * sum_adjust_beta2)
            pl2_list.append(pl2)
            if verbose: print(f"Step2 {i}:", error)
      beta2_new_n = thresholding(X=beta2_new_n, thr=beta_min2) # thresholding to beta_min2
      beta2_al = np.zeros(N)
      beta2_al[idx] = multiply(beta2_new_n, beta2_hat)
            
      return beta_al, beta_new_n, beta2_al, beta2_new_n

                  

def estimate_beta_pham(x):
    t1, t2 = x.shape
    if t1>t2:
        print("Error in estim_beta_pham(x): data must be organized in x in a row fashion")
        return

    #beta = np.zeros_like(x)
    psi1, entropy1 = score_cond(x.T)
    psi2, entropy2 = score_cond(np.flipud(x).T)
    beta = np.vstack([-psi1[:,0], -psi2[:,0]])
    return beta
    
def score_cond(data, q=0, bw=None, cov=None):
    """
    Estimate the conditional score function defined as minus the gradient of the conditional density of of a random variable X_p
    given x_{p-1}, \dots, x_{q+1}. The values of these random variables are provided in the n x p array data.
    The estimator is based on the partial derivatives of the conditional entropy with respect to the data values, the entropy
    being estimated through the kernel density estimate _after a prewhitening operation_, with the kernel being the density of the
    sum of 3 independent uniform random variables in [.5,.5]. 
    The kernel bandwidth is set to bdwidth*(standard deviation) and the density is evaluated at bandwidth apart. 
    bdwidth defaults to  2*(11*sqrt(pi)/20)^((p-q)/(p-q+4))*(4/(3*n))^(1/(p-q+4) (n = sample size), which is optimal for estimating a normal density.

    If cov (a p x p matrix) is present, it is used as the covariance matrix of the data and the mean is assume 0. 
    This prevents recomputation of the mean and covariance if the data is centered and/or prewhitenned.
    The score function is computed at the data points and returned in psi.
    
    Inputs: 
        data: n x p array
        q: delete first q columns
        bw
        cov: p x p array
    Outputs:
        psi
        entropy
    """
    from scipy.sparse import coo_matrix
    N, P = data.shape
    if P < q+1: 
        print("Error: not enough variables")
        return
    if q>0: 
        data = data[:, q:]
        p = p - q
    if bw is None: bw = 2*(11*np.sqrt(np.pi)/20)**(P/(P+4))*(4/(3*N))**(1/(P+4))
    if cov is None: 
        data_centered = data - np.expand_dims(np.mean(data, axis=0), axis=0).dot(np.ones((P,1)))
        cov = data_centered.T.dot(data_centered)/N
    # prewhitening
    T = np.linalg.cholesky(cov)
    data = np.matmul(data, np.linalg.inv(T))

    # Grouping the data into cells, idx gives the index of the cell containing a datum, r gives its relative distance to the leftmost border of the cell
    r = data / bw
    idx = np.floor(r).astype(np.int16)
    r = r - idx
    tmp = np.min(idx, axis=0)
    #print("tmp:", tmp)
    idx = idx - np.tile(tmp, (N,1)) #+ 1  # 0 <= idx-1
    #print("idx:", idx)
    
    # Compute the probabilities at grid cells.
    """
    The kernel function is
           1/2 + (1/2+u)(1/2-u) for |u| <= 1/2
    k(u) = (3/2 - |u|)^2/2 for 1/2 <= |u| <= 3/2
           0 otherwise
    The contribution to the probability at i-1, i, i+1 by a data point at distance r (0 <= r < 1) to i, are respectively:
    (1-r)^2/2, 1/2 + r*(1-r), r^2/2
    The derivative of the contribution to the probability at i-1, i, i+1 by a data point at distance r to i are respectively:
     r-1, 1-2r, r
    The array ker contains the contributions to the probability of cells
    The array kerp contains the gradient of these contributions
    The array ix contains the indexes of the cells, arranged in _lexicographic order_
    """
    def k1(u):
        return np.vstack([(1-u)**2/2, 0.5+u*(1-u), u**2/2]).T
    def k2(u):
        return np.vstack( [1-u, 2*u-1, -u]).T
    ker = k1(r[:,0])
    kerp = k2(r[:,0]) 
    #print(ker.shape, kerp.shape)
    ix = np.vstack([idx[:,0], idx[:,0]+1, idx[:,0]+2]).T
    #print(ix.shape)
    mx = np.max(idx, axis=0) + 2 + 1
    #print("mx", mx)
    M = np.cumprod(mx)
    #print(M.shape)
    for i in range(1,P):
        #print(i, 3**i)
        rr1 = np.tile(k1(r[:,i]), (i, 3**i))  # i repeated 3^(i-1) times
        rr2 = np.tile(k2(r[:,i]), (1, 3**i))
        rr3 = np.tile(k1(r[:,i]), (1, 3**i))
        #print(rr1.shape, rr2.shape)
        #print(np.tile(kerp,(1,3)).shape, np.tile(ker,(1,3)).shape)
        kerp = np.vstack([np.multiply(np.tile(kerp,(1,3)), rr1), np.multiply(np.tile(ker,(1,3)), rr2)])
        ker = np.multiply(np.tile(ker,(1,3)), rr3)
        #print(kerp.shape, ker.shape)
        
        Mi = M[i-1]
        #print("Mi:", Mi, "/ix:", ix.shape)
        ix1 = ix + np.tile(Mi *idx[:,i]-1, (3**i,1)).T
        ix2 = ix + np.tile(Mi *idx[:,i]+0, (3**i,1)).T
        ix3 = ix + np.tile(Mi *idx[:,i]+1, (3**i,1)).T
        ix = np.hstack([ix1, ix2, ix3])
        
    # Compute the conditional entropy (if asked)
    #print(ix.flatten())
    #print(ker.flatten().shape, ix.flatten().shape, np.zeros_like(ix.flatten()).shape, M[P-1])
    #print(M)
    pr = coo_matrix((ker.flatten(), (ix.flatten(), np.zeros_like(ix.flatten()))), shape=(M[P-1],1))/N # joint prob. of cells
    #logp = coo_matrix((M[P-1],1), dtype=np.float32) # to contain log(cond. prob.)
    if P>1:
        pm = np.sum(pr.reshape((Mi,mx(P-1))), axis=1) # marginal prob. (Mi = M(p-1))
        pm = np.tile(pm, (1,mx[P-1]))
        pm = pm.reshape((M[P-1],1))
        pr_div = coo_matrix(np.divide(pr.toarray(),pm))
        logp = pr_div.copy()
        logp.data = np.log(pr_div.data)
    else:
        #print(pr, pr.shape)        
        logp = pr.copy()
        logp.data = np.log(pr.data)

    #print(pr.shape, logp.shape)
    entropy = np.squeeze(np.log(bw*T[-1,-1]) - (pr.T).dot(logp).toarray())
    logp = logp.toarray()
    
    # Compute the conditional score
    #print(ix.shape)
    #print(logp)
    psi = np.sum(np.multiply(np.squeeze(logp[ix]), kerp),axis=1) # nr = 1:n repeated p times
    psi = psi.reshape((N,P))/bw
    psi = psi - np.expand_dims(np.mean(psi, axis=1), axis=1).dot(np.ones((1,P))) # centering
    lam = np.matmul(psi.T, data)/N
    lam = np.tril(lam) + np.tril(lam, -1).T
    lam[P-1,P-1] -= 1
    if q>0:
        psi_tmp = np.zeros((N,P+q))
        psi_tmp[:,q:] = np.matmul(np.linalg.inv(T).T * (psi - np.matmul(data, lam)))
        psi = psi_tmp
    else:
        #print(np.linalg.inv(T).T.shape, (psi - np.matmul(data, lam)).shape)
        psi = np.matmul( (psi - np.matmul(data, lam)), np.linalg.inv(T).T)
    return psi, entropy


def adaptive_size(grad_new, grad_old, eta_old, z_old):
    alpha = 0
    up, down = 1.05, 0.5
    z = grad_new + alpha *z_old
    eta_up = np.multiply(grad_new, grad_old) >= 0
    eta = np.multiply(eta_old, (up * eta_up + down * (1-eta_up)))
    eta[eta>0.03] = 0.03
    return eta, z
    
def natural_grad_Adasize_Mask(X, Mask):
    """
    Inputs:
        X: (n,k)
        Mask: (k,k)
    Outputs:
        W: (n,n)
    """
    N, K = X.shape
    mu = 1E-3
    iter_max = 6000
    tol = 1E-4
    n_edges = np.sum(Mask)

    # initilization of W
    WW = np.eye(N)
    for i in range(N):
        idx = Mask[i,:]!=0
        WW[i,idx] = -0.5*np.matmul((X[i,:].dot(X[idx,:].T)) , pdinv(X[idx,:].dot(X[idx,:].T)))
    W = 0.5*(WW+WW.T) # (n,n)
    z = np.zeros((N,N))
    eta = mu * np.ones_like(W)
    W_old = W
    y_psi = np.zeros((N, K)) # (n,k)
    y_psi0 = np.zeros((N, K)) #(n,k)
    for i in range(iter_max):
        y = W.dot(X) # (n,k)
        # update W: linear ICA with marginal score function estimated from data...
                  
        if i % 12 ==1: 
            for j in range(N):            
                tem = estimate_beta_pham(y[j:j+1,:])
                ind = np.argsort(y[j,:]) # sorting in ascending order
                y_psi[j,:] = tem[0,:]
                y_psi0[j,:] = tem[0,ind]
        else: 
            for j in range(N):
                ind = np.argsort(y[j,:]) # sorting in ascending order
                y_psi[j,ind] = y_psi0[j,:]    
        #G = np.matmul(y_psi, y.T)/K  #(i,n)
        #vv = np.matmul(y, y.T)/K     #(n,n) 
        #I = np.eye(N)
        grad_W_n = np.matmul(y_psi, X.T) + np.linalg.inv(W.T)
        if i==0:
            grad_W_o = grad_W_n
        eta, z = adaptive_size(grad_new=grad_W_n, grad_old=grad_W_o, eta_old=eta, z_old=z)
        delta_W = np.multiply(eta, z)
        W = W + np.multiply(delta_W, Mask)
        
        if np.sum(abs(np.multiply(grad_W_n, Mask)))/n_edges < tol:
            break
        grad_W_o = grad_W_n
        W_old = W

    return W


from tqdm import tqdm

def sparseica_W_adasize_Alasso_mask(X, Mask, lam):
    """
    ICA with SCAD penalized entries of the de-mixing matrix
    Inputs:
        X : Input matrix --> (n,k) ndarray (variable number * sample size) 
        Mask: adjacency matrix
        lam : penalizing parameter
    Outputs:
        y : (n,k)
        W : (n,n)
        WW :(n,n)
        score :
    """
    N, K = X.shape
    refine = 1
    n_edges = np.sum(Mask)
    mu = 1E-3 # learning rate
    beta = 0  
    save_intermediate = False
    m = 60 # for approximate the derivative of |.|
    iter_max = int(1E5)
    tol = 1E-4
    W_bak, eta_bak, z_bak, grad_bak = np.zeros((N,N,iter_max)), np.zeros((N,N,iter_max)), np.zeros((N,N,iter_max)), np.zeros((N,N,iter_max)) #back-up
    # iter_M = 200
    # delta_H = 0
    # w11_bak, w12_bak = [], []
    # normalization via axis 1 to avoid instability
    X_mean_centered = X - np.expand_dims(np.mean(X, axis=1), axis=1).dot(np.ones((1,K)))
    X_normalized = np.matmul(np.diag(1./np.std(X_mean_centered, axis=1)), X_mean_centered)
    
    # initialization
    WW = np.diag(1./np.std(X_mean_centered, axis=1))
    WW = natural_grad_Adasize_Mask(X_mean_centered, Mask)
    omega1 = 1./abs(WW[Mask!=0])
    # to avoid instability
    upper = 3 * np.mean(omega1);
    omega1 = (omega1>upper)*upper + np.multiply(omega1, (omega1<=upper))
    omega = np.zeros((N,N)) 
    omega[Mask!=0] = omega1
    W = WW

    z = np.zeros((N,N))
    eta = mu * np.ones_like(W)
    W_old = W + np.eye(N)
    grad_new = W_old
    y_psi = np.zeros((N,K))
    y_psi0 = np.zeros((N,K))
    print(" Starting penalization ...")
    for i in tqdm(range(iter_max)):
      y = W.dot(X_normalized)
      curr = np.sum(abs(np.multiply(grad_new, Mask)))/n_edges 
      if curr < tol:
            if refine==1:
              Mask = abs(W) > 0.02
              Mask = Mask.astype(np.int8) - np.diag(np.diagonal(Mask)).astype(np.int8)
              lam = 0
              refine = 0
            else:
              break
      W_old = W
      if i % 12 ==1: 
        for j in range(N):            
            tem = estimate_beta_pham(y[j:j+1,:])
            ind = np.argsort(y[j,:]) # sorting in ascending order
            y_psi[j,:] = tem[0,:]
            y_psi0[j,:] = tem[0,ind]
      else: 
        for j in range(N):
            ind = np.argsort(y[j,:]) # sorting in ascending order
            y_psi[j,ind] = y_psi0[j,:]
      dev = np.matmul(omega, np.tanh(m*W))
      grad_new = np.matmul(y_psi, X.T) + np.linalg.inv(W.T) - 4*beta*(np.diag(np.diagonal(np.matmul(y, y.T)/K)) - np.eye(N)) - dev*lam/K
      if i==0: 
        grad_old = grad_new
      eta, z = adaptive_size(grad_new=grad_new, grad_old=grad_old, eta_old=eta, z_old=z)
      delta_W = np.multiply(eta, z)
      W = W + 0.9 * np.multiply(delta_W, Mask)
      grad_old = grad_new
      if save_intermediate:
            W_bak[:,:,i], z_bak[:,:,i], eta_bak[:,:,i], grad_bak[:,:,i] = W, z, eta, grad_new
            

    score = np.matmul(omega, np.abs(W)) 
    return y, W, WW, score


def two_step_CD(X, lam, Mask=None):
  """ 
  Two-step method for linear causal discovery that allows cycles and
  confounders
  Input: 
    numpy ndarray X --> (k,n) ndarray (variable number * sample size) 
    penalizing(regularization) parameter lam (lambda)
    Mask: the initialized adjcency matrix
  Output: 
    B: the causal influence matrix X = BX + E;
    in B causal direction goes from column to row, the entry Bij, means
    Xj -> Xi
    W_m: the ICA de-mixing matrix.
  by Kun Zhang 2016,2017 Carnegie Mellon University
  translated in python by Sunghwan Kim 2022, KAIST
  """

  N, T = X.shape
  print("X:", X.shape)

  if Mask is None:
    var_coef = 0.65**2
    # estimate the mask using adaptative lasso
    Mask = np.zeros((N,N))
    print("Mask:", Mask.shape)
    for i in range(N):
      if T<4*N: # sample size too small, so preselect the features
          if i==0: print("small sample size: T<4N")
          tmp1 = np.delete(X,i,axis=0) # get X\xi
          vcorrcoef(np.delete(X,i,axis=0), X[i])
          corr_abs = np.abs(vcorrcoef(np.delete(X,i,axis=0), X[i])) # compute the correlation of xi with X\xi 
          corr_ind = np.argsort(corr_abs) #sort from larger in absolute value, get values and indices
          X_sel = tmp1[corr_ind[-int(N/4):],: ]# pre-select N/4 features, the ones more correlated to xi
          beta_alt, beta_new_nt, beta2_alt, beta2_new_nt = betaAlasso_grad_2step(X_sel, X[i,:], var_coef*np.var(X[i,:]), np.log(T)/2)
          #print(tmp1.shape)
          #print(X_sel.shape)
          beta2_al = np.zeros(N-1);
          beta2_al[corr_ind[-int(N/4):]] = beta2_alt;
      else:
          beta_al, beta_new_nt, beta2_al, beta2_new_nt = betaAlasso_grad_2step(np.delete(X,i,axis=0), X[i,:], var_coef*np.var(X[i,:]), np.log(T)/2)
      idx = np.delete(np.array(range(N)), i)
      Mask[i, idx] = abs(beta2_al)>0.01
    Mask = Mask + Mask.T
  Mask = (Mask != 0)

  # perform constrained_ICA
  print("Sparse ICA is running ...")
  y_m, W_m, WW_m, Score = sparseica_W_adasize_Alasso_mask(X, Mask, np.log(T)*lam)
  B = np.eye(N) - W_m;
  return B, W_m
