import numpy as np

from transformation import quaternion_rotate

def compute_closest_points(P, X):
    """
    Args:
        P: データ np.ndarray(3,N)
        X: モデル np.ndarray(3,M)
    Returns: 
        データの各点に対応するX上の最近接点の集合 np.ndarray(3,N)
    """
    Y = np.zeros(P.shape)
    for i,p_i in enumerate(P.T):
        cp = find_closest_point(p_i, X)
        Y[:,i] = cp
    # print(f"dmin= {dmin}, cp = {cp}")
    return Y

def find_closest_point(p, X):
    """
    Args:
        p: データ上の点 np.ndarray(3,1)
        X: モデル np.ndarray(3,M)
    Returns: X上のpからの最近接点 np.ndarray(3,1)
    """
    cp = X[:,0]
    dmin = np.linalg.norm(p - X[:,0])
    for j in range(1,len(X.T)):
        d = np.linalg.norm(p - X[:,j])
        if d < dmin:
            cp = X[:,j]
            dmin = d
    return cp

def compute_registration(p, q):
    """
    Args:
        p: データ np.ndarray(3,N) 
        q: モデル np.ndarray(3,N)
    Returns:
        pからqへのレジストレーション
        qu: quaternion np.ndarray(4,)
        T:  平行移動ベクトル np.ndarray(3,)
    """
    mu_p = p.mean(axis=1)
    mu_q = q.mean(axis=1)
    p_ = p - mu_p.reshape((3,1))
    q_ = q - mu_q.reshape((3,1))
    M = np.zeros((4,4))

    for p_i,q_i in zip(p_.T,q_.T):
        P_i = np.array([[     0, -p_i[0], -p_i[1], -p_i[2]],
                        [p_i[0],       0,  p_i[2], -p_i[1]],
                        [p_i[1], -p_i[2],       0,  p_i[0]],
                        [p_i[2],  p_i[1], -p_i[0],      0]])
        Q_i = np.array([[     0, -q_i[0], -q_i[1], -q_i[2]],
                        [q_i[0],       0, -q_i[2],  q_i[1]],
                        [q_i[1],  q_i[2],       0, -q_i[0]],
                        [q_i[2], -q_i[1],  q_i[0],      0]])
        M += np.matmul(P_i.T, Q_i)

    w,v = np.linalg.eigh(M)
    qu = v[:,3]
    T = mu_q - quaternion_rotate(mu_p, qu)
    return qu,T

def apply_registration(P, qu, T, Y):
    """
    Args:
        P:  データ np.ndarray(3,N) 
            関数の中で更新される
        qu: quaternion np.ndarray(4,)
        T:  平行移動ベクトル np.ndarray(3,)
        Y:  最近接点の集合 np.ndarray(3,N)
            平均二乗誤差計算用
    Returns:
        平均二乗誤差
    """
    d = 0
    for i in range(P.shape[1]):
        P[:,i] = quaternion_rotate(P[:,i], qu) + T
        d += np.sum((P[:,i] - Y[:,i])**2)
    return np.sqrt(d)/P.shape[1]

def icp(P_0, X):
    """
    Args:
        P_0: データ初期値 np.ndarray(3,N)
        X:   モデル np.ndarray(3,M)
    Returns:
        収束後のレジストレーション
        qu:  quaternion np.ndarray(4,)
        T:   平行移動ベクトル np.ndarray(3,)
        d_k: 平均二乗誤差 float
    """
    P_k = np.copy(P_0)

    # mean square point matching error
    d_kprev = -1

    while True:
        # a. Compute the closest points: Y_k
        Y_k = compute_closest_points(P_k, X)

        # b. Compute registration: q_k
        qu,T = compute_registration(P_0, Y_k)

        # c. Apply the registration: P_k+1
        d_k = apply_registration(P_k, qu, T, Y_k)

        # d. Terminate the iteration
        if d_kprev >= 0 and d_kprev - d_k < 1e-9:
            break
        d_kprev = d_k

    return qu,T,d_k
