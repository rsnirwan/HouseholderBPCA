import numpy as np
import pystan
import pickle
from hashlib import md5
from sklearn import decomposition


# For synthetic data
#####################################################
def haar_measure(D, Q):
    """
    outputs a matrix from stiefel(D, N)
    """
    z = np.random.normal(0,1,size=(D,D))
    q, r = np.linalg.qr(z)
    sign_r = np.sign(np.diag(r))
    return np.matmul(q, np.diag(sign_r))[:,:Q]
    
    
def get_data(N, D, Q, sigma):
    """
    U from stiefel(Q,D) and fixed sigma -> W = U*diag(sigma)
    X from normal 
    output Y = X*W.T, U
    """
    U = haar_measure(D,Q)
    W = np.matmul(U, np.diag(sigma))
    X = np.random.normal(size=(N, Q))
    return np.matmul(X, W.T) + np.random.normal(0, 0.01, size=(N, D)), U

#####################################################



# initialization with PCA solution
#####################################################
def sign_convention(U):
    """
    sign convention
    """
    return np.array( [-U[:,q] if U[0,q] < 0 else U[:,q] 
                         for q in range(U.shape[1])] ).T


def pca_solution(Y, Q):
    """returns first Q eigenvectors and eigenvalues of Y"""
    pca = decomposition.PCA(n_components=Q)
    pca.fit(Y)
    U_pca = pca.components_.T   
    U_pca = sign_convention(U_pca)
    sigma_pca = np.sqrt(pca.explained_variance_)
    return U_pca, sigma_pca


def householder(v: "array [Q,1]", D: int) -> "array [D,D]":
    """return householder transformation of size len(v) x len(v)"""
    Q = v.shape[0]
    H = np.eye(D,D)
    sgn = v[0,0]/np.fabs(v[0,0])
    u = v+sgn*np.linalg.norm(v)*np.eye(Q,1)
    u = u/np.linalg.norm(u)
    H[-Q:, -Q:] = -sgn*(np.eye(Q,Q) - 2*np.outer(u,u))
    return H


def get_v(U: "array [Q,Q]") -> "array [Q,1]":
    return U[:,0].reshape(-1,1)


def get_vs(U: "array [D,D]", D: int, Q: int) -> "list [list [q]]":
    """get the vs that leads to matrix U by inverse Householder trafos"""
    vs = []
    HU = U
    for q in range(Q):
        vs.append(get_v(HU[q:, q:]))
        H = householder(vs[-1], D)
        HU = H.dot(HU)
    return vs


def get_vs_stan_inp(U_pca: "array [D,Q]", D: int, Q: int) -> "list[floats]":
    """return the vs in a format that the stan code can take as input"""
    vs = get_vs(U_pca[:,::-1], D, Q) # stans ordering is reversed
    v_mat = np.zeros((D,Q))
    for i, v in enumerate(vs):
        v_mat[i:,i] = v.reshape(-1)

    vs_inp = []     # see function V_low_tri_plus_diag in stan code
    for d1 in range(D):
        for d2 in range(D):
            if d1 >= d2 and d2 < Q:
                vs_inp.append(v_mat[d1, d2])
    return vs_inp

#####################################################



def print_mat(A: "array [N,D]"):
    for row in A:
        for a in row:
            print("{:+.4f}".format(a), end="  ")
        print()
    print()


# from: http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html - modified
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'py_stan_code/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'py_stan_code/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
