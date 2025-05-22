import pandas as pd
import numpy as np

def cancor_QR_SVD(X, Y):
    Q1, R1 = np.linalg.qr(X)
    Q2, R2 = np.linalg.qr(Y)

    # Q1 and Q2 are orthogonal.
    assert np.allclose(np.dot(Q1.T, Q1), np.eye(Q1.shape[1]))
    assert np.allclose(np.dot(Q2.T, Q2), np.eye(Q2.shape[1]))

    p1 = R1.shape[1]
    p2 = R2.shape[1]

    # Ranks must be positive.
    assert p1 > 0 and p2 > 0

    mat = np.dot(Q1.T, Q2)[:p1, :p2]

    U, Sigma, Vt = np.linalg.svd(mat)

    A = np.linalg.solve(R1[:p1], U)
    B = np.linalg.solve(R2[:p2], Vt.T)

    return Sigma, A, B

def cancor_SVD(X, Y):
    U1, Sigma1, Vt1 = np.linalg.svd(X, full_matrices=False)
    U2, Sigma2, Vt2 = np.linalg.svd(Y, full_matrices=False)

    mat = np.dot(U1.T, U2)

    U, Sigma, Vt = np.linalg.svd(mat)

    A = Vt1.T @ np.diag(1 / Sigma1) @ U
    B = Vt2.T @ np.diag(1 / Sigma2) @ Vt.T

    return Sigma, A, B

def data_load():
    df = pd.read_csv('data/LifeCycleSavings.csv')
    pop = df[['pop15', 'pop75']]
    oec = df[['sr', 'dpi', 'ddpi']]

    # Centering by mean.
    X = pop - pop.mean(axis=0)
    Y = oec - oec.mean(axis=0)

    # X and Y have the same number of rows.
    assert X.shape[0] == Y.shape[0]

    return X, Y

def main():
    X, Y = data_load()

    Sigma, A, B = cancor_QR_SVD(X, Y)

    print('Sigma =', Sigma)
    print('A =', A)
    print('B =', B)

    S = np.dot(X, A)
    T = np.dot(Y, B)

    # S and T are orthogonal.
    assert np.allclose(np.dot(S.T, S), np.eye(S.shape[1]))
    assert np.allclose(np.dot(T.T, T), np.eye(T.shape[1]))

    Sigma, A, B = cancor_SVD(X, Y)

    print('Sigma =', Sigma)
    print('A =', A)
    print('B =', B)

    S = np.dot(X, A)
    T = np.dot(Y, B)

if __name__ == '__main__':
    main()
