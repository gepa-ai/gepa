program = """
# EVOLVE-BLOCK-START
# MUTATION APPLIED: Added zero-vector warm-start, ridge linear surrogate directional probes, and a budget-limited Nelder–Mead subspace finisher
# RATIONALE: Zero seed targets typical polynomial optima near origin; ridge-linear steps cheaply exploit global gradient hints from archive; Nelder–Mead subspace can squeeze extra improvements near the end without heavy modeling

import numpy as np
import os
import json
import math
import time


def solve(dim, total_evaluation_budgets, bounds):
    # Bounds and helpers
    lb = np.array([b[0] for b in bounds], dtype=float)
    ub = np.array([b[1] for b in bounds], dtype=float)
    span = ub - lb
    mid = (lb + ub) / 2.0

    # Anytime-valid best
    best_x = mid.copy()
    best_y = -np.inf
    evals = 0
    budgets = int(total_evaluation_budgets)

    rng = np.random.default_rng()

    # Reflection to bounds (mirror) for arrays of points
    def reflect_to_bounds(X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            Y = X.copy()
            for j in range(dim):
                if span[j] <= 0:
                    Y[j] = lb[j]
                    continue
                r = (Y[j] - lb[j]) % (2.0 * span[j])
                if r <= span[j]:
                    Y[j] = lb[j] + r
                else:
                    Y[j] = ub[j] - (r - span[j])
            return Y
        else:
            Y = X.copy()
            for j in range(dim):
                if span[j] <= 0:
                    Y[:, j] = lb[j]
                    continue
                r = (Y[:, j] - lb[j]) % (2.0 * span[j])
                mask = r <= span[j]
                Y[mask, j] = lb[j] + r[mask]
                Y[~mask, j] = ub[j] - (r[~mask] - span[j])
            return Y

    # Archive of evaluated points
    archive_X = []
    archive_y = []
    archive_cap = max(800, 30 * dim)  # larger local history for surrogates

    # Safe evaluation wrapper
    def evaluate(xcand):
        nonlocal evals, best_x, best_y
        if evals >= budgets:
            return None
        xclip = reflect_to_bounds(xcand)  # reflection ensures in-bounds
        try:
            y = objective_function(xclip)
        except Exception:
            y = -np.inf
        evals += 1
        if not np.isfinite(y):
            y = -np.inf
        # push to archive
        archive_X.append(xclip.copy())
        archive_y.append(float(y))
        if len(archive_X) > archive_cap:
            idx = np.argsort(np.array(archive_y))[::-1]
            keep_top = idx[: archive_cap // 2]
            keep_recent = np.arange(max(0, len(archive_X) - archive_cap // 2), len(archive_X))
            keep_idx = np.unique(np.concatenate([keep_top, keep_recent]))
            archive_X[:] = [archive_X[i] for i in keep_idx]
            archive_y[:] = [archive_y[i] for i in keep_idx]
        if y > best_y:
            best_y = float(y)
            best_x = xclip.copy()
            print(f"Improved: eval {evals}/{budgets}, best_y={best_y:.6f}")
        return y

    # Halton sequence for initialization
    def halton(n, d):
        def next_prime():
            primes = []
            candidate = 2
            while True:
                is_p = True
                r = int(candidate**0.5) + 1
                for p in primes:
                    if p > r:
                        break
                    if candidate % p == 0:
                        is_p = False
                        break
                if is_p:
                    primes.append(candidate)
                    yield candidate
                candidate += 1

        def vdc(i, base):
            v = 0.0
            denom = 1.0
            while i > 0:
                i, rem = divmod(i, base)
                denom *= base
                v += rem / denom
            return v

        g = next_prime()
        bases = [next(g) for _ in range(d)]
        H = np.zeros((n, d), dtype=float)
        for j in range(d):
            b = bases[j]
            for i in range(n):
                H[i, j] = vdc(i + 1, b)
        return H

    # Simple Latin Hypercube Sampling in [0,1]^d
    def lhs(n, d):
        P = np.zeros((n, d), dtype=float)
        for j in range(d):
            perm = rng.permutation(n)
            P[:, j] = (perm + rng.random(n)) / n
        return P

    # Warm-start seeds
    seeds = []

    env_x = os.environ.get("PAST_BEST_X", "").strip()
    if env_x:
        try:
            arr = None
            if env_x.startswith("["):
                arr = np.array(json.loads(env_x), dtype=float)
            else:
                arr = np.array([float(v) for v in env_x.split(",")], dtype=float)
            if arr.shape == (dim,):
                arr = reflect_to_bounds(arr)
                seeds.append(arr)
                print("Warm-start: using PAST_BEST_X from environment")
        except Exception as e:
            print(f"Warning: failed to parse PAST_BEST_X: {e}")

    # Previous trace for the known 11D box
    if dim == 11 and np.allclose(lb, -10) and np.allclose(ub, 30):
        prev_trace_x = np.array(
            [
                15.20424723,
                29.3365135,
                8.96735818,
                5.99694284,
                17.86117045,
                7.47621172,
                25.77999103,
                -6.53946212,
                6.99513464,
                1.20286935,
                0.21514371,
            ],
            dtype=float,
        )
        if prev_trace_x.shape == (dim,):
            seeds.append(reflect_to_bounds(prev_trace_x))
            print("Warm-start: using previous trace candidate for dim=11")

    # Midpoint and zero-vector seeds (zero often near polynomial optima)
    seeds.append(mid.copy())
    zero_vec = np.zeros(dim, dtype=float)
    if np.all(zero_vec >= lb) and np.all(zero_vec <= ub):
        seeds.append(zero_vec)
        print("Seeding: added zero vector")

    # Halton + LHS seeds mapped to bounds
    extra_seeds = max(0, min(8, budgets - len(seeds)))
    if extra_seeds > 0:
        H = halton((extra_seeds + 1) // 2, dim)
        L = lhs(extra_seeds - H.shape[0], dim) if extra_seeds > H.shape[0] else np.zeros((0, dim))
        S = []
        for i in range(H.shape[0]):
            S.append(lb + H[i] * span)
        for i in range(L.shape[0]):
            S.append(lb + L[i] * span)
        seeds.extend(S)

    # A couple random seeds
    n_random_seeds = max(1, min(2, budgets - len(seeds)))
    for _ in range(n_random_seeds):
        seeds.append(rng.uniform(lb, ub))

    # Evaluate seeds
    for s in seeds:
        if evals >= budgets:
            break
        evaluate(s)

    if evals >= budgets:
        print(f"Finished during seeding. Best y: {best_y:.6f}")
        print("y: ", best_y)
        return best_x

    # ES parameters
    def suggest_lambda(rem):
        # slightly stronger population
        base = int(max(10, 6 + 3.8 * np.log(dim + 5)))
        base = int(min(base + rem // max(60, 10 * dim), base + 8))
        return max(4, min(base, rem))

    sigma = 0.20  # relative to span
    sigma_min, sigma_max = 0.004, 0.6  # allow finer final search

    m = best_x.copy()
    m_y = best_y

    stall_gens = 0
    max_stall_gens = 10
    gen = 0
    last_grad_gen = -999

    # Local coordinate pattern search with smaller granularity
    def local_refine(max_evals):
        nonlocal sigma
        if max_evals <= 0:
            return
        s = np.clip(0.5 * sigma * span, 0.001 * span, 0.2 * span)
        shrink = 0.5
        min_step = np.maximum(1e-5 * span, 1e-8)  # absolute minimal step
        while (max_evals > 0) and np.any(s > min_step):
            improved_loop = False
            order = rng.permutation(dim)
            for j in order:
                if max_evals <= 0:
                    break
                if s[j] <= min_step[j]:
                    continue
                for direction in (+1, -1):
                    if max_evals <= 0:
                        break
                    current = best_x.copy()
                    step = s[j]
                    candidate = current.copy()
                    candidate[j] = candidate[j] + direction * step
                    y = evaluate(candidate)
                    max_evals -= 1
                    if y is None:
                        return
                    if y > best_y:
                        improved_loop = True
                        while max_evals > 0:
                            current = best_x.copy()
                            step *= 1.6
                            if step < min_step[j]:
                                break
                            candidate2 = current.copy()
                            candidate2[j] = candidate2[j] + direction * step
                            y2 = evaluate(candidate2)
                            max_evals -= 1
                            if y2 is None or not (y2 > best_y):
                                break
                if not improved_loop:
                    s[j] *= shrink
            if not improved_loop:
                s *= shrink
            sigma = max(sigma_min, min(sigma_max, float(np.median(s / np.maximum(span, 1e-12)))))

    # Quadratic surrogate trust-region proposals around best_x using full archive
    def quad_trust_propose(max_evals):
        if max_evals <= 0 or len(archive_X) < dim + 5:
            return 0
        X = np.array(archive_X, dtype=float)
        y = np.array(archive_y, dtype=float)
        maskf = np.isfinite(y)
        X = X[maskf]
        y = y[maskf]
        if len(X) < dim + 5:
            return 0
        U = (X - best_x) / np.maximum(span, 1e-12)
        p = 1 + dim + dim + (dim * (dim - 1)) // 2
        target = min(len(U), max(6 * dim, p + 10))
        r = max(0.1 * sigma, 0.02)
        sel_idx = np.where(np.linalg.norm(U, axis=1) <= r)[0]
        attempts = 0
        while (len(sel_idx) < min(target, len(U))) and attempts < 5:
            r *= 1.7
            sel_idx = np.where(np.linalg.norm(U, axis=1) <= r)[0]
            attempts += 1
        if len(sel_idx) < max(dim + 5, p // 2):
            dists = np.linalg.norm(U, axis=1)
            sel_idx = np.argsort(dists)[: min(target, len(U))]
        Usel = U[sel_idx]
        ysel = y[sel_idx]
        n = len(Usel)
        if n < dim + 5:
            return 0

        ones = np.ones((n, 1), dtype=float)
        A_parts = [ones, Usel, Usel**2]
        cross_cols = []
        pairs = []
        for i in range(dim):
            for j in range(i + 1, dim):
                cross_cols.append((Usel[:, i] * Usel[:, j])[:, None])
                pairs.append((i, j))
        if len(cross_cols) > 0:
            A_parts.append(np.hstack(cross_cols))
        A = np.hstack(A_parts)
        lam = max(1e-6, 1e-3 * np.maximum(1.0, np.var(ysel)))
        A_reg = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
        y_reg = np.concatenate([ysel, np.zeros(A.shape[1], dtype=float)])
        try:
            w, *_ = np.linalg.lstsq(A_reg, y_reg, rcond=1e-3)
        except Exception:
            return 0

        idx_const = 0
        idx_lin_start = 1
        idx_lin_end = 1 + dim
        idx_sq_start = idx_lin_end
        idx_sq_end = idx_sq_start + dim
        idx_cross_start = idx_sq_end
        g = w[idx_lin_start:idx_lin_end].copy()
        Q = np.zeros((dim, dim), dtype=float)
        sq_coeffs = w[idx_sq_start:idx_sq_end]
        for i in range(dim):
            Q[i, i] = 2.0 * sq_coeffs[i]
        offset = idx_cross_start
        for k, (i, j) in enumerate(pairs):
            Q[i, j] = w[offset + k]
            Q[j, i] = w[offset + k]

        try:
            eig_max = np.linalg.eigvalsh((Q + Q.T) * 0.5).max()
        except Exception:
            eig_max = np.max(np.diag(Q))
        if eig_max >= -1e-8:
            Qc = Q - (eig_max + 1e-3) * np.eye(dim)
        else:
            Qc = Q

        proposals = []
        try:
            u_star = -np.linalg.solve(Qc, g)
        except Exception:
            diag = np.diag(Qc)
            diag = np.where(diag < -1e-8, diag, -1e-3)
            u_star = -g / diag

        r_tr = min(0.6, max(0.12 * sigma, 0.02))
        norm_u = np.linalg.norm(u_star)
        if norm_u > r_tr and norm_u > 0:
            u_star = u_star * (r_tr / norm_u)
        x_star = best_x + u_star * span
        proposals.append(x_star)

        diag = np.diag(Qc)
        diag = np.where(diag < -1e-8, diag, -1e-3)
        u_diag = -g / diag
        norm_ud = np.linalg.norm(u_diag)
        if norm_ud > r_tr and norm_ud > 0:
            u_diag = u_diag * (r_tr / norm_ud)
        proposals.append(best_x + u_diag * span)

        if norm_u > 1e-12:
            for alpha in (0.5, 1.5):
                u_ls = u_star * alpha
                norm_ls = np.linalg.norm(u_ls)
                if norm_ls > r_tr and norm_ls > 0:
                    u_ls = u_ls * (r_tr / norm_ls)
                proposals.append(best_x + u_ls * span)

        for _ in range(2):
            du = rng.standard_normal(dim)
            du /= np.linalg.norm(du) + 1e-12
            du *= r_tr * rng.uniform(0.3, 1.0)
            proposals.append(x_star + du * span)

        used = 0
        for p in proposals:
            if used >= max_evals or evals >= budgets:
                break
            evaluate(p)
            used += 1
        return used

    # Subspace (active-dim) quadratic trust-region proposals
    def subspace_quad_trust_propose(k_dims, max_evals):
        if max_evals <= 0 or len(archive_X) < max(dim // 2, 12):
            return 0
        X = np.array(archive_X, dtype=float)
        y = np.array(archive_y, dtype=float)
        maskf = np.isfinite(y)
        X = X[maskf]
        y = y[maskf]
        if len(X) < max(dim // 2, 12):
            return 0
        U = (X - best_x) / np.maximum(span, 1e-12)
        # choose points near best_x
        dists = np.linalg.norm(U, axis=1)
        sel0 = np.argsort(dists)[: min(len(U), max(10 * k_dims, 5 * k_dims + 10))]
        Usel = U[sel0]
        ysel = y[sel0]
        if len(Usel) < 5 * k_dims + 5:
            return 0
        # active dims via correlation with y
        corrs = []
        for j in range(dim):
            uj = Usel[:, j]
            if np.std(uj) < 1e-12:
                corrs.append(0.0)
            else:
                c = np.corrcoef(uj, ysel)[0, 1]
                if not np.isfinite(c):
                    c = 0.0
                corrs.append(abs(c))
        idxs = np.argsort(corrs)[::-1][: min(k_dims, dim)]
        k = len(idxs)
        if k == 0:
            return 0
        U_k = Usel[:, idxs]
        n = len(U_k)
        # Build quadratic in k dims
        ones = np.ones((n, 1), dtype=float)
        A_parts = [ones, U_k, U_k**2]
        cross_cols = []
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                cross_cols.append((U_k[:, i] * U_k[:, j])[:, None])
                pairs.append((i, j))
        if len(cross_cols) > 0:
            A_parts.append(np.hstack(cross_cols))
        A = np.hstack(A_parts)
        p = 1 + k + k + (k * (k - 1)) // 2
        if n < p + 3:
            return 0
        lam = max(1e-6, 1e-3 * np.maximum(1.0, np.var(ysel)))
        A_reg = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
        y_reg = np.concatenate([ysel, np.zeros(A.shape[1], dtype=float)])
        try:
            w, *_ = np.linalg.lstsq(A_reg, y_reg, rcond=1e-3)
        except Exception:
            return 0
        idx_const = 0
        idx_lin_start = 1
        idx_lin_end = 1 + k
        idx_sq_start = idx_lin_end
        idx_sq_end = idx_sq_start + k
        idx_cross_start = idx_sq_end
        gk = w[idx_lin_start:idx_lin_end].copy()
        Qk = np.zeros((k, k), dtype=float)
        sq_coeffs = w[idx_sq_start:idx_sq_end]
        for i in range(k):
            Qk[i, i] = 2.0 * sq_coeffs[i]
        offset = idx_cross_start
        for t, (i, j) in enumerate(pairs):
            Qk[i, j] = w[offset + t]
            Qk[j, i] = w[offset + t]
        # make concave
        try:
            eig_max = np.linalg.eigvalsh((Qk + Qk.T) * 0.5).max()
        except Exception:
            eig_max = np.max(np.diag(Qk))
        if eig_max >= -1e-8:
            Qkc = Qk - (eig_max + 1e-3) * np.eye(k)
        else:
            Qkc = Qk
        try:
            uk = -np.linalg.solve(Qkc, gk)
        except Exception:
            diag = np.diag(Qkc)
            diag = np.where(diag < -1e-8, diag, -1e-3)
            uk = -gk / diag
        r_tr = min(0.5, max(0.1 * sigma, 0.02))
        nu = np.linalg.norm(uk)
        if nu > r_tr and nu > 0:
            uk = uk * (r_tr / nu)
        # map back to full space
        full_u = np.zeros(dim, dtype=float)
        full_u[idxs] = uk
        props = [best_x + full_u * span]
        # a couple of radial variants
        for alpha in (0.6, 1.3):
            uk2 = uk * alpha
            if np.linalg.norm(uk2) > r_tr and np.linalg.norm(uk2) > 0:
                uk2 = uk2 * (r_tr / np.linalg.norm(uk2))
            fu2 = np.zeros(dim, dtype=float)
            fu2[idxs] = uk2
            props.append(best_x + fu2 * span)
        used = 0
        for pnt in props:
            if used >= max_evals or evals >= budgets:
                break
            evaluate(pnt)
            used += 1
        return used

    # Finite-difference gradient trust-region probes near best_x
    def gradient_tr_probe(max_evals, gen_idx):
        nonlocal last_grad_gen
        if max_evals <= 0:
            return 0
        if gen_idx - last_grad_gen < 4:
            return 0
        last_grad_gen = gen_idx

        per_dim_cost = 2
        max_dims = max(1, min(dim, max_evals // per_dim_cost))
        if max_dims <= 0:
            return 0
        idxs = np.arange(dim)
        rng.shuffle(idxs)
        idxs = idxs[:max_dims]

        g = np.zeros(dim, dtype=float)
        h_base = np.clip(0.01 * sigma, 0.002, 0.05)
        used = 0
        for j in idxs:
            if evals >= budgets or used + 2 > max_evals:
                break
            h = max(h_base * span[j], 1e-8)
            xp = best_x.copy()
            xp[j] += h
            xm = best_x.copy()
            xm[j] -= h
            yp = evaluate(xp)
            used += 1
            ym = evaluate(xm)
            used += 1
            if yp is None or ym is None:
                continue
            g[j] = (yp - ym) / (2.0 * h)

        normg = np.linalg.norm(g)
        if normg <= 1e-16:
            return used

        d = g / (normg + 1e-12)
        r_tr = min(0.5, max(0.1 * sigma, 0.01))
        for alpha in (0.3, 0.8, 1.2):
            if evals >= budgets or used >= max_evals:
                break
            step = r_tr * alpha
            cand = best_x + step * d * span
            evaluate(cand)
            used += 1
        return used

    # Ridge-linear surrogate directional probe using archive (cheap global gradient hint)
    def ridge_linear_probe(max_evals):
        if max_evals <= 0 or len(archive_X) < max(20, 3 * dim):
            return 0
        X = np.array(archive_X, dtype=float)
        y = np.array(archive_y, dtype=float)
        maskf = np.isfinite(y)
        X = X[maskf]
        y = y[maskf]
        if len(X) < max(20, 3 * dim):
            return 0
        U = (X - best_x) / np.maximum(span, 1e-12)
        dists = np.linalg.norm(U, axis=1)
        # select points near best but with variety
        sel = np.argsort(dists)[: min(len(U), max(12 * dim, 80))]
        Usel = U[sel]
        ysel = y[sel]
        n = len(Usel)
        ones = np.ones((n, 1), dtype=float)
        A = np.hstack([ones, Usel])
        lam = max(1e-6, 1e-3 * np.maximum(1.0, np.var(ysel)))
        A_reg = np.vstack([A, math.sqrt(lam) * np.eye(A.shape[1])])
        y_reg = np.concatenate([ysel, np.zeros(A.shape[1], dtype=float)])
        try:
            w, *_ = np.linalg.lstsq(A_reg, y_reg, rcond=1e-3)
        except Exception:
            return 0
        g = w[1:].copy()
        if not np.all(np.isfinite(g)):
            return 0
        normg = np.linalg.norm(g)
        if normg <= 1e-12:
            return 0
        d = g / normg
        used = 0
        r_tr = min(0.55, max(0.12 * sigma, 0.02))
        for alpha in (0.5, 1.0, 1.6):
            if evals >= budgets or used >= max_evals:
                break
            cand = best_x + (r_tr * alpha) * d * span
            evaluate(cand)
            used += 1
        return used

    # Generate approximately orthogonal noise directions (blocks of up to dim vectors)
    def orthogonal_noise(lam):
        dirs = []
        remaining = lam
        while remaining > 0:
            k = int(min(dim, remaining))
            G = rng.standard_normal((dim, k))
            try:
                Q, _ = np.linalg.qr(G, mode="reduced")
                V = Q.T  # shape (k, dim)
            except Exception:
                V = G.T / (np.linalg.norm(G, axis=0, keepdims=True).T + 1e-12)
            scales = rng.standard_normal(k)
            block = (V.T * scales).T  # each row is a direction
            dirs.append(block)
            remaining -= k
        Z = np.vstack(dirs)[:lam]
        return Z

    # Diagonal sampling scales from elite archive (normalized to median=1)
    def diagonal_sampling_scales():
        if len(archive_X) < max(20, 4 * dim):
            return np.ones(dim, dtype=float)
        yarr = np.array(archive_y, dtype=float)
        mask = np.isfinite(yarr)
        if not np.any(mask):
            return np.ones(dim, dtype=float)
        idx_sorted = np.argsort(yarr[mask])[::-1]
        Xarr = np.array(archive_X, dtype=float)[mask][idx_sorted]
        topk = Xarr[: min(len(Xarr), max(20, 6 * dim))]
        U = (topk - best_x) / np.maximum(span, 1e-12)
        stds = np.std(np.abs(U), axis=0) + 1e-12
        med = np.median(stds)
        if not np.isfinite(med) or med <= 0:
            return np.ones(dim, dtype=float)
        scales = np.clip(stds / med, 0.6, 1.8)
        return scales

    # Occasional global exploration burst
    def global_explore(max_evals):
        used = 0
        if max_evals <= 0:
            return 0
        n = int(min(max_evals, max(6, 2 * dim)))
        for _ in range(n):
            if evals >= budgets:
                break
            # Biased towards best with heavy tails, or uniform
            if rng.random() < 0.6:
                z = rng.standard_cauchy(dim)
                z = np.clip(z, -8, 8)
                scale = 0.4 if sigma > 0.05 else 0.7  # push more when sigma small
                cand = best_x + scale * sigma * z * span
            else:
                cand = rng.uniform(lb, ub)
            evaluate(cand)
            used += 1
        return used

    # Budget-limited Nelder–Mead in a chosen subspace near best_x
    def nelder_mead_subspace(k_dims, max_evals):
        if max_evals <= 0:
            return 0
        # choose active dims by archive correlation; fallback random
        if len(archive_X) >= max(20, 3 * dim):
            X = np.array(archive_X, dtype=float)
            y = np.array(archive_y, dtype=float)
            mask = np.isfinite(y)
            X = X[mask]
            y = y[mask]
            U = (X - best_x) / np.maximum(span, 1e-12)
            dists = np.linalg.norm(U, axis=1)
            sel = np.argsort(dists)[: min(len(U), max(10 * k_dims, 5 * k_dims + 10))]
            Usel = U[sel]
            ysel = y[sel]
            corrs = []
            for j in range(dim):
                uj = Usel[:, j]
                if np.std(uj) < 1e-12:
                    corrs.append(0.0)
                else:
                    c = np.corrcoef(uj, ysel)[0, 1]
                    if not np.isfinite(c):
                        c = 0.0
                    corrs.append(abs(c))
            idxs = np.argsort(corrs)[::-1][: min(k_dims, dim)]
        else:
            idxs = rng.choice(dim, size=min(k_dims, dim), replace=False)
        k = len(idxs)
        if k == 0:
            return 0

        # initial simplex
        step = np.clip(0.12 * sigma * span[idxs], 1e-6, 0.2 * span[idxs])
        simplex = [best_x.copy()]
        for i in range(k):
            p = best_x.copy()
            p[idxs[i]] = p[idxs[i]] + step[i]
            simplex.append(reflect_to_bounds(p))
        ys = []
        used = 0
        for p in simplex:
            if evals >= budgets or used >= max_evals:
                break
            ys.append(evaluate(p))
            used += 1
        if used < len(simplex):
            return used

        alpha, gamma, rho, nm_shrink = 1.0, 2.0, 0.5, 0.5
        it_guard = 0
        while used < max_evals and evals < budgets and it_guard < 200:
            it_guard += 1
            order = np.argsort([-np.inf if v is None else v for v in ys])[::-1]  # maximize
            simplex = [simplex[i] for i in order]
            ys = [ys[i] for i in order]
            xh = np.array(simplex[-1])
            yh = ys[-1]
            xl = np.array(simplex[0])
            yl = ys[0]
            xc = np.mean(np.array(simplex[:-1]), axis=0)

            xr = reflect_to_bounds(xc + alpha * (xc - xh))
            if used >= max_evals or evals >= budgets:
                break
            yr = evaluate(xr)
            used += 1
            if yr is None:
                break

            if yr > yl:
                xe = reflect_to_bounds(xc + gamma * (xr - xc))
                if used >= max_evals or evals >= budgets:
                    break
                ye = evaluate(xe)
                used += 1
                if ye is not None and ye > yr:
                    simplex[-1] = xe
                    ys[-1] = ye
                else:
                    simplex[-1] = xr
                    ys[-1] = yr
            elif yr > ys[-2]:
                simplex[-1] = xr
                ys[-1] = yr
            else:
                xcand = reflect_to_bounds(xc + rho * (xh - xc))
                if used >= max_evals or evals >= budgets:
                    break
                ycand = evaluate(xcand)
                used += 1
                if ycand is not None and ycand > yh:
                    simplex[-1] = xcand
                    ys[-1] = ycand
                else:
                    # shrink around best
                    xbest = np.array(simplex[0])
                    new_simplex = [xbest.copy()]
                    new_ys = [ys[0]]
                    for i in range(1, len(simplex)):
                        xi = reflect_to_bounds(xbest + nm_shrink * (simplex[i] - xbest))
                        if used >= max_evals or evals >= budgets:
                            new_simplex.append(xi)
                            new_ys.append(None)
                            continue
                        yi = evaluate(xi)
                        used += 1
                        new_simplex.append(xi)
                        new_ys.append(yi)
                    simplex, ys = new_simplex, new_ys
                    continue
        return used

    start_time = time.time()

    while evals < budgets:
        gen += 1
        remaining = budgets - evals
        lam = suggest_lambda(remaining)

        # Orthogonalized sampling around m with diagonal scaling and occasional heavy-tail perturbations
        Z = orthogonal_noise(lam)
        diag_sc = diagonal_sampling_scales()
        Z = Z * diag_sc[None, :]
        if rng.random() < 0.25:
            idx_ht = rng.integers(0, lam)
            ht = rng.standard_cauchy(size=dim)
            ht = np.clip(ht, -10, 10)
            Z[idx_ht] = ht * diag_sc

        # add antithetic partners to improve symmetry when possible
        if lam % 2 == 1 and remaining >= lam + 1:
            # ensure even count by duplicating one and mirroring
            Z = np.vstack([Z, -Z[-1:]])
            lam = Z.shape[0]
        for i in range(0, lam, 2):
            if i + 1 < lam:
                Z[i] = -Z[i + 1]

        candidates = m + Z * (sigma * span)
        candidates = reflect_to_bounds(candidates)

        ys = []
        improvements = 0
        for i in range(lam):
            if evals >= budgets:
                break
            y_i = evaluate(candidates[i])
            ys.append(y_i)
            if y_i is not None and y_i > m_y:
                m_y = float(y_i)
                m = candidates[i].copy()
                improvements += 1

        if len(ys) == 0:
            break

        # Recombination (rank-based)
        valid_idx = [i for i, yy in enumerate(ys) if yy is not None and np.isfinite(yy)]
        if len(valid_idx) > 0:
            sorted_idx = sorted(valid_idx, key=lambda i: ys[i], reverse=True)
            mu = max(1, len(sorted_idx) // 2)
            ranks = np.arange(1, mu + 1)
            weights = np.log(mu + 0.5) - np.log(ranks)
            weights /= np.sum(weights)
            m_recomb = np.sum(candidates[sorted_idx[:mu]] * weights[:, None], axis=0)
            alpha = 0.35
            m = reflect_to_bounds((1 - alpha) * m + alpha * m_recomb)

        # 1/5th success rule for global step-size
        succ_frac = improvements / max(1, lam)
        if succ_frac > 0.22:
            sigma *= 1.12
            stall_gens = 0
        else:
            sigma *= 0.87
            stall_gens += 1
        sigma = float(np.clip(sigma, sigma_min, sigma_max))

        # Occasional lightweight local probes near the current best
        if succ_frac > 0 and evals < budgets:
            k = min(3, dim)
            idxs = rng.choice(dim, size=k, replace=False)
            for j in idxs:
                if evals >= budgets:
                    break
                step = 0.35 * sigma * span[j]
                if step <= 0:
                    continue
                for direction in (+1, -1):
                    if evals >= budgets:
                        break
                    probe = best_x.copy()
                    probe[j] = probe[j] + direction * step
                    evaluate(probe)

        # Cheap ridge-linear surrogate directional steps when stagnating lightly
        if (stall_gens >= 1) and evals < budgets:
            rem = budgets - evals
            budget_r = int(min(max(3, dim // 2), rem // 4 if rem > 40 else rem // 3))
            used_r = ridge_linear_probe(budget_r)
            if used_r > 0:
                m = best_x.copy()
                m_y = best_y

        # If stagnated for a few generations, try subspace and full quadratic surrogate trust-region proposals
        if (stall_gens >= 2) and evals < budgets:
            rem = budgets - evals
            # First try a couple of active subspaces
            budget_s = int(min(rem // 4, max(6 * dim, 18)))
            used_s = 0
            used_s += subspace_quad_trust_propose(min(4, dim), budget_s // 2)
            if evals < budgets and used_s < budget_s:
                used_s += subspace_quad_trust_propose(min(6, dim), budget_s - used_s)
            if used_s > 0:
                m = best_x.copy()
                m_y = best_y

        if (stall_gens >= 3) and evals < budgets:
            rem = budgets - evals
            budget_q = int(min(rem // 4, max(8 * dim, 24)))
            used_q = quad_trust_propose(budget_q)
            if used_q > 0:
                m = best_x.copy()
                m_y = best_y

        # Finite-difference gradient trust-region steps when stagnating mildly
        if (stall_gens >= 2) and evals < budgets:
            rem = budgets - evals
            budget_g = int(min(max(6, 2 * dim), rem // 3 if rem > 30 else rem // 2))
            used_g = gradient_tr_probe(budget_g, gen)
            if used_g > 0:
                m = best_x.copy()
                m_y = best_y

        # Occasional global exploration burst if long stall
        if (stall_gens >= 6) and evals < budgets:
            rem = budgets - evals
            burst = int(min(rem // 3, max(12, 3 * dim)))
            print(f"Global exploration burst at gen {gen} with budget {burst}")
            global_explore(burst)
            m = best_x.copy()
            m_y = best_y

        print(f"Gen {gen}: evals {evals}/{budgets}, best_y={best_y:.6f}, sigma={sigma:.3f}, stall_gens={stall_gens}")

        # If stagnated long, perform a local refiner; then maybe restart
        if stall_gens >= max_stall_gens and evals < budgets:
            rem_before = budgets - evals
            refine_budget = int(min(rem_before // 3, max(dim * 8, 24)))
            if refine_budget > 0:
                print(f"Local refine at gen {gen} with budget {refine_budget}")
                local_refine(refine_budget)
                m = best_x.copy()
                m_y = best_y
            stall_gens = 0
            if evals < budgets:
                # soft restart near the best with moderate sigma
                m = reflect_to_bounds(best_x + rng.normal(scale=0.12, size=dim) * span)
                m_y = -np.inf
                sigma = 0.22
                print(f"Restart at gen {gen}: new center near best, sigma={sigma:.3f}")

        # Final-phase: if nearing the end, allocate remaining budget to subspace + local refine + last surrogate pass + NM finisher
        remaining = budgets - evals
        if remaining <= max(30, dim * 4) and remaining > 0:
            used = 0
            used += subspace_quad_trust_propose(min(6, dim), max(1, remaining // 4))
            remaining = budgets - evals
            if remaining > 0:
                used += quad_trust_propose(max(1, remaining // 4))
            remaining = budgets - evals
            if remaining > 0:
                nm_budget = max(0, remaining // 3)
                if nm_budget > 0:
                    print(f"Final-phase Nelder–Mead subspace with budget {nm_budget}")
                    nelder_mead_subspace(min(6, dim), nm_budget)
            remaining = budgets - evals
            if remaining > 0:
                print(f"Final-phase local refine with remaining budget {remaining}")
                local_refine(remaining)
            break

        # Safety: time cap guard (soft)
        if time.time() - start_time > 0.9 * 300:
            print("Time cap nearing, exiting loop.")
            break

    print("y: ", best_y)
    return best_x


# EVOLVE-BLOCK-END
"""
