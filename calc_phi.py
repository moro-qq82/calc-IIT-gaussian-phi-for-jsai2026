import numpy as np
import numpy.linalg as la
import math

r"""
# see C:\Users\moro-\OneDrive\ドキュメント\05_research_projects\
#      13_hypothesis_for_FEP-IIT_integration\
#      20251229_下書き_脳内ダイナミクスとしての概念と共通概念形成^L7自由エネルギー原理および統合情報理論による解釈.docx

"""


def cycle_matrix(n, w):
    A = np.zeros((n, n))
    for i in range(n):
        A[(i+1) % n, i] = w  # i -> i+1
    return A

def solve_stationary_cov_iter(A, sigma2=1.0, tol=1e-10, max_iter=20000, debug=False):
    n = A.shape[0]
    # スペクトル半径チェック（>=1 なら発散）
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
    """原子分割（各ノード単独）でのΦ"""
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
    """任意分割 parts（index配列のリスト）でのΦ"""
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


# --- 上：3つ独立（A:5, B:5, K:6） ---
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

# --- 下：統合（跨ぎエッジを追加） ---
A_bottom = A_top.copy()
c = 0.3 # 跨ぎエッジの結合の強さ

# index: A 0-4, B 5-9, K 10-15 左上を始まりとして時計回りにカウント
# ここを図1に合わせている
add_edge(A_bottom, 10, 0, c)  # A1 -> K1
add_edge(A_bottom,  6,11, c)  # K2 -> B2
add_edge(A_bottom, 13, 7, c)  # B3 -> K4
add_edge(A_bottom,  3,13, c)  # K4 -> A4


print("A_bottom:\n", A_bottom)

phi_bottom = phi_gaussian_atomic(A_bottom)

# 3塊（A|B|K）分割での統合も見る
parts = [list(range(0,5)), list(range(5,10)), list(range(10,16))]
phi_top_groups = phi_gaussian_partition(A_top, parts)
phi_bottom_groups = phi_gaussian_partition(A_bottom, parts)

ln2 = math.log(2)

print("=== atomic (各ノードばらばら) ===")
print("phi_A:", phi_A, "nats")
print("phi_B:", phi_B, "nats")
print("phi_K:", phi_K, "nats")
print("top sum:", phi_top_sum, "nats =", phi_top_sum/ln2, "bits")
print("bottom :", phi_bottom, "nats =", phi_bottom/ln2, "bits")
print("delta  :", phi_bottom - phi_top_sum, "nats =", (phi_bottom - phi_top_sum)/ln2, "bits")

print("\n=== partition (A|B|K の3塊) ===")
print("top   :", phi_top_groups, "nats")
print("bottom:", phi_bottom_groups, "nats =", phi_bottom_groups/ln2, "bits")
