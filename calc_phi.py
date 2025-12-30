import numpy as np
import numpy.linalg as la
import math

r"""
see 
Barrett, Adam B., and Anil K. Seth. 2011. “Practical Measures of Integrated Information for Time-Series Data.” 
PLoS Computational Biology 7 (1): e1001052.
"""


def cycle_matrix(n, w):
    """
    create n x n cycle adjacency matrix with weight w
    see fig.1
    0 -> 1 -> 2 -> ... -> n-1 -> 0
    """
    A = np.zeros((n, n))
    for i in range(n):
        A[(i+1) % n, i] = w  # i -> i+1
    return A


def solve_stationary_cov_iter(A, sigma2=1.0, tol=1e-10, max_iter=20000, debug=False):
    """Iteratively solve for the steady-state covariance matrix."""
    n = A.shape[0]
    # check spectral radius (if >=1 diverges)
    eigs = la.eigvals(A)
    rho = np.max(np.abs(eigs))
    if debug:
        print("spectral radius:", rho)
    if rho >= 1.0:
        raise ValueError(f"Matrix A is unstable (spectral radius {rho:.6g} >= 1). Covariance will diverge.")

    Q = sigma2 * np.eye(n)
    S = Q.copy()
    for it in range(max_iter):
        S_new = A @ S @ A.T + Q
        maxval = np.max(np.abs(S_new))
        if not np.isfinite(maxval):
            raise OverflowError(f"Overflow at iteration {it}: max|S_new|={maxval}")
        if debug and (it % 100 == 0 or maxval > 1e8):
            print(f"iter {it} max|S| {maxval:.3e}")
        if np.max(np.abs(S_new - S)) < tol:
            return S_new
        S = S_new
    raise RuntimeError("Max iterations reached without convergence")


def phi_gaussian_atomic(A, sigma2=1.0):
    """ obtain phi at atomic partition(each node separate) """
    n = A.shape[0]
    S = solve_stationary_cov_iter(A, sigma2)
    Q = sigma2 * np.eye(n)

    YY = A @ S @ A.T + Q
    YX = A @ S
    XX = S

    whole = YY - YX @ la.solve(XX, YX.T)

    resid_vars = []
    for i in range(n):
        varY = YY[i, i]
        varX = XX[i, i]
        cov = YX[i, i]
        resid_vars.append(varY - cov**2 / varX)

    part = np.diag(resid_vars)

    signw, logw = la.slogdet(whole)
    signp, logp = la.slogdet(part)
    return 0.5 * (logp - logw)


def phi_gaussian_partition(A, parts, sigma2=1.0):
    """ obtain phi at arbitrary partition(index lists) """
    n = A.shape[0]
    S = solve_stationary_cov_iter(A, sigma2)
    Q = sigma2 * np.eye(n)

    YY = A @ S @ A.T + Q
    YX = A @ S
    XX = S

    whole = YY - YX @ la.solve(XX, YX.T)

    part = np.zeros_like(whole)
    for idxs in parts:
        idxs = np.array(idxs)
        YYi = YY[np.ix_(idxs, idxs)]
        YXi = YX[np.ix_(idxs, idxs)]
        XXi = XX[np.ix_(idxs, idxs)]
        res = YYi - YXi @ la.solve(XXi, YXi.T)
        part[np.ix_(idxs, idxs)] = res

    signw, logw = la.slogdet(whole)
    signp, logp = la.slogdet(part)
    return 0.5 * (logp - logw)

def add_edge(A, to_idx, from_idx, w):
    A[to_idx, from_idx] = w


# --- upper part of fig.1（A:5, B:5, K:6, all independent） ---
w = 0.9
A_A = cycle_matrix(5, w)
A_B = cycle_matrix(5, w)
A_K = cycle_matrix(6, w)

print("A_A:\n", A_A)

A_top = np.block([
    [A_A, np.zeros((5,5)), np.zeros((5,6))],
    [np.zeros((5,5)), A_B, np.zeros((5,6))],
    [np.zeros((6,5)), np.zeros((6,5)), A_K],
])

print("A_top:\n", A_top)

phi_A = phi_gaussian_atomic(A_A)
phi_B = phi_gaussian_atomic(A_B)
phi_K = phi_gaussian_atomic(A_K)
phi_top_sum = phi_A + phi_B + phi_K

# --- upper part of fig.1 (add straddle edges) ---
A_bottom = A_top.copy()
c = 0.3 # weight of straddle edges

# index: A 0-4, B 5-9, K 10-15   top left is the first node, clockwise
# see fig.1
add_edge(A_bottom, 10, 0, c)  # A1 -> K1
add_edge(A_bottom,  6,11, c)  # K2 -> B2
add_edge(A_bottom, 13, 7, c)  # B3 -> K4
add_edge(A_bottom,  3,13, c)  # K4 -> A4


print("A_bottom:\n", A_bottom)

phi_bottom = phi_gaussian_atomic(A_bottom)

# obtain 3-block (A|B|K) partitioning also
parts = [list(range(0,5)), list(range(5,10)), list(range(10,16))]
phi_top_groups = phi_gaussian_partition(A_top, parts)
phi_bottom_groups = phi_gaussian_partition(A_bottom, parts)

ln2 = math.log(2)

print("=== atomic (each node separate) ===")
print("phi_A:", phi_A, "nats")
print("phi_B:", phi_B, "nats")
print("phi_K:", phi_K, "nats")
print("top sum:", phi_top_sum, "nats =", phi_top_sum/ln2, "bits")
print("bottom :", phi_bottom, "nats =", phi_bottom/ln2, "bits")
print("delta  :", phi_bottom - phi_top_sum, "nats =", (phi_bottom - phi_top_sum)/ln2, "bits")

print("\n=== partition (A|B|K blocks) ===")
print("top   :", phi_top_groups, "nats")
print("bottom:", phi_bottom_groups, "nats =", phi_bottom_groups/ln2, "bits")
