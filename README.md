# Canonical Coordinates

## Introduction

When there are two feature representations or data for a given subject, we can introduce a *relation* between the two data. Based on the relation, there is a method called *canonical correlation analysis*, CCA, that extracts new features from the two data.
In theory, this method had already been discovered by *C. Jordan* in 1875[^Jordan1875]. Here we introduce the algorithms of CCA based on the description by *W.H. Press*[^Press2011].

## Canonical Coordinates

There are two data matrices with a relation for each row, $X = [x_{i,j}]$ of $(n \times p_1)$-matrix and $Y = [y_{i,j}]$ of $(n \times p_2)$-matrix. $n$ is the number of samples, and $p_1$ and $p_2$ are the dimensions of *features*, respectively. We consider to transform the coordinates based on the relations between the data, i.e., take the transformed coordinates in the order of largest correlation for each feature. The destination coordinates are called *canonical coordinates*.

Without loss of generality, we can assume that each row has zero mean,
```math
\begin{equation}\begin{split}
\sum_{i=1}^n x_{i,j} &= 0, \quad j = 1, \dots, p_1 \\
\sum_{i=1}^n y_{i,j} &= 0, \quad j = 1, \dots, p_2.
\end{split}\end{equation}
```

We denote the transformation matrices of coefficients of the linear combinations by $A$ for $X$ of ($p_1 \times d$)-matrix and $B$ for $Y$ of ($p_2 \times d$)-matrix. Denote the canonical coordinates by $S$ and $T$, we have

```math
\begin{equation}
S = X A, \quad T = Y B.
\label{st}
\end{equation}
```

When we perform a linear transformation that maximizes the correlation for each feature, each feature is orthogonal to all but the corresponding feature. That is, denote the i-th features of X and Y by $X_i$ and $Y_i$, then $X_i \prescript{t}{}{Y_j} = 0$ if $i \ne j$.

Therefore, a number of the dimension of the canonical coordinates is equal to the smaller of X and Y ranks, that is, $d = \mathrm{min}\big(\mathrm{rank}(X), \mathrm{rank}(Y)\big)$. Consequently, the correlation coordinates $S$ and $T$ are cross-orthogonal.

```math
\begin{equation}
\prescript{t}{}{S} T = D
\label{orthogonal}
\end{equation}
```
where $D$ is a diagonal matrix in $d$ dimensions.

We can take $A$ and $B$ scaling to satisfy
```math
\begin{equation}
\prescript{t}{}{S} S = I_d, \quad \prescript{t}{}{T} T = I_d,
\label{scale}
\end{equation}
```
where $I_d$ is the identity matrix in $d$ dimensions. Thus $S$ and $T$ are orthogonal matrices.

## Proof

We must prove that canonical coordinates exist.
Here we show the simplest case, where the $X$ and $Y$ have the same dimensions and there are no column degeneracies, that is, $p_1 = p_2 = d$.
Then $(d \times d)$ matrices $A$ and $B$ have $2d^2$ total degrees of freedom. Each equation in \eqref{scale} requests $d(d + 1)/2$ constraints, taking into account that $\prescript{t}{}{S} S$ and $\prescript{t}{}{T} T$ are symmetric matrices.
Equation \eqref{orthogonal} requests $d^2 − d$ constraints, because the elements of $D$ are not constrained. Now,

```math
\begin{equation}
\frac{d(d+1)}{2} + \frac{d(d+1)}{2} + (d^2-d) = 2d^2
\end{equation}
```

Thus, total constraints equals total degrees of freedom, and there can be only countable and isolated solutions. In fact, all the solutions are simply column permutations of one another.

## Algorithm Using QR and SVD

We calculate the canonical coordinates by *QR decomposition* and *singular value decomposition*, SVD. First, decompose $X$ and $Y$ with QR,
```math
\begin{equation}
X = Q_1 R_1, \quad Y = Q_2 R_2
\end{equation}
```
where the $Q$ is column orthogonal, $\prescript{t}{}{Q_1} Q_1 = I_{p_1}, \prescript{t}{}{Q_2} Q_2 = I_{p_2}$, and the $R_1$ and $R_2$ are upper triangular of $(p1 \times p1)$- and $(p2 \times p2)$-matrices, respectively.

Next, form $\prescript{t}{}{Q_1} Q_2$ and decompose with SVD,
```math
\begin{equation}
\prescript{t}{}{Q_1} Q_2 = U \Sigma \prescript{t}{}{V}
\end{equation}
```
where $U$ and $V$ are orthogonal and $\Sigma$ is diagonal.
Now we define
```math
\begin{equation}
A \equiv R^{-1}_1 U, \quad B \equiv R^{-1}_2 V
\label{ab}
\end{equation}
```

Then we can confirm the equations \eqref{orthogonal} and \eqref{scale}:
```math
\begin{equation}
\begin{split}
D &= \prescript{t}{}{S} T = \prescript{t}{}{A} \prescript{t}{}{X} Y B \\
&= (\prescript{t}{}{U} \prescript{t}{}{R}^{-1}_1)(\prescript{t}{}{R_1} \prescript{t}{}{Q_1})(Q_2 R_2)(R^{-1}_2 V) \\
&= \prescript{t}{}{U} (U \Sigma \prescript{t}{}{V}) V \\
&= \Sigma
\end{split}
\end{equation}
```

Note that we do not actually require the computation of the inverse matrices. Since $\Sigma$ is diagonal, so is $D$. Similarly,
```math
\begin{equation}
\begin{split}
\prescript{t}{}{S} S &= \prescript{t}{}{A} \prescript{t}{}{X} X A \\
&= (\prescript{t}{}{U} \prescript{t}{}{R}^{-1}_1)(\prescript{t}{}{R}_1 \prescript{t}{}{Q}_1)(Q_1 R_1)(R^{-1}_1 U) \\
&= I_{p_1}
\end{split}
\end{equation}
```
and an analogous calculation for $\prescript{t}{}{T} T = I_{p_2}$.

If $p_1 \ne p_2$, or if there are degeneracies, then only the first $d$ columns of $S$ and $T$ are kept in equation \eqref{ab}, corresponding to the $d$ largest singular values in $\Sigma$.

## Algorithm Using SVD Only

In most cases, SVD is slower than QR. In fact, most algorithms for SVD have an internal QR decomposition.
However, the algorithm using QR does not yield the variances for and between $X$ and $Y$. In this context, it is useful to give an algorithm for canonical correlation with SVD only because SVD constructs orthogonal bases and the transformations are easily invertible.

First, decompose $X$ and $Y$ with SVD,
```math
\begin{equation}
X = U_1 \Sigma_1 \prescript{t}{}{V}_1, \quad Y = U_2 \Sigma_2 \prescript{t}{}{V}_2
\end{equation}
```

Next, form $\prescript{t}{}{U}_1 U_2$ and decompose it with SVD,
```math
\begin{equation}
\prescript{t}{}{U}_1 U_2 = U \Sigma \prescript{t}{}{V}
\end{equation}
```

Now define
```math
\begin{equation}
A \equiv V_1 \Sigma^{-1}_1 U, \quad B \equiv V_2 \Sigma^{-1}_2 V
\end{equation}
```

Confirm equations \eqref{orthogonal} and \eqref{scale} as above:
```math
\begin{equation}
\begin{split}
D &= \prescript{t}{}{S} T = \prescript{t}{}{A} \prescript{t}{}{X} Y B \\
&= (\prescript{t}{}{U} \Sigma^{-1}_1 \prescript{t}{}{V}_1)(V_1 \Sigma_1 \prescript{t}{}{U}_1)(U_2 \Sigma_2 \prescript{t}{}{V}_2)(V_2 \Sigma^{-1}_2 V) \\
&= \prescript{t}{}{U} (U \Sigma \prescript{t}{}{V}) V \\
&= \Sigma
\end{split}
\end{equation}
```

```math
\begin{equation}
\begin{split}
\prescript{t}{}{S} S &= \prescript{t}{}{A} \prescript{t}{}{X} X A \\
&= (\prescript{t}{}{U} \Sigma^{-1}_1 \prescript{t}{}{V}_1)(V_1 \Sigma_1 \prescript{t}{}{U}_1)(U_1 \Sigma_1 \prescript{t}{}{V}_1)(V_1 \Sigma^{-1}_1 U) \\
&= I_{p_1}
\end{split}
\end{equation}
```
and correspondingly for $\prescript{t}{}{T} T = I_{p_2}$.

[^Jordan1875]: C. Jordan, *Essai sur la géométrie à n dimensions*, **Bull Soc Math France**, 1875.
[^Press2011] W.H. Press, *Canonical Correlation Clarified by Singular Value Decomposition*, 2011.
