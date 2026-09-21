"""
Bradley-Terry competition simulation.

N firms compete on K dimensions. Each firm i allocates effort x_{i,k} across
dimensions to maximize win-rate against other firms under a budget constraint.

    max_{x_i}  sum_{i'≠i} sum_k  w_k * σ(f(x_{i,k}) - f(x_{i',k}))
    s.t.        sum_k x_{i,k} <= B_i,  x_{i,k} >= 0

Social welfare for user distribution v:
    SW = sum_k  v_k * max_i  f(x_{i,k})
"""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class CompetitionSimulation:
    def __init__(
        self,
        N: int,
        K: int,
        B: np.ndarray,
        w: np.ndarray,
        v: np.ndarray,
        f: Callable[[float], float],
        f_prime: Optional[Callable[[float], float]] = None,
    ):
        """
        Parameters
        ----------
        N        : number of firms
        K        : number of dimensions / topic categories
        B        : budget vector, shape (N,)
        w        : evaluator topic weights, shape (K,)  — will be normalized
        v        : user topic weights, shape (K,)       — will be normalized
        f        : production function  θ = f(x)
        f_prime  : derivative f'(x); if None, estimated via finite differences
        """
        assert len(B) == N and len(w) == K and len(v) == K
        self.N = N
        self.K = K
        self.B = np.asarray(B, float)
        self.w = np.asarray(w, float) / np.sum(w)
        self.v = np.asarray(v, float) / np.sum(v)
        self.f = np.vectorize(f)
        self._f_scalar = f
        self.f_prime = np.vectorize(f_prime) if f_prime else None

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def theta(self, X: np.ndarray) -> np.ndarray:
        """Output matrix: theta[i, k] = f(X[i, k]).  Shape (N, K)."""
        return self.f(X)

    def firm_objective(self, x_i: np.ndarray, i: int, X: np.ndarray) -> float:
        """Win-rate of firm i given full allocation matrix X (row i may differ)."""
        th_i = self.f(x_i)
        th = self.theta(X)
        total = 0.0
        for ip in range(self.N):
            if ip != i:
                total += float(np.sum(self.w * expit(th_i - th[ip])))
        return total

    def all_objectives(self, X: np.ndarray) -> np.ndarray:
        """Win-rates for all N firms.  Shape (N,)."""
        th = self.theta(X)
        objs = np.zeros(self.N)
        for i in range(self.N):
            for ip in range(self.N):
                if ip != i:
                    objs[i] += float(np.sum(self.w * expit(th[i] - th[ip])))
        return objs

    def social_welfare(self, X: np.ndarray) -> float:
        """SW = sum_k  v_k * max_i  f(X[i,k])."""
        th = self.theta(X)
        return float(np.dot(self.v, np.max(th, axis=0)))

    # ------------------------------------------------------------------
    # Single-firm best response
    # ------------------------------------------------------------------

    def best_response(self, i: int, X: np.ndarray, n_restarts: int = 4) -> np.ndarray:
        """
        Solve firm i's optimization problem given others' allocations.

        Uses SLSQP with multiple random restarts to escape local optima.
        The problem is a separable NLP on a budget simplex; concavity of
        the objective (and thus global optimality of the local solution)
        holds when  d/dx [σ'(f(x)-c) f'(x)]  <= 0 for all c — see notes.
        """
        B_i = self.B[i]
        bounds = [(0.0, B_i)] * self.K
        constraints = [{"type": "ineq", "fun": lambda x: B_i - np.sum(x)}]

        candidates = [X[i].copy()]
        rng = np.random.default_rng(0)
        for _ in range(n_restarts - 1):
            alpha = rng.dirichlet(np.ones(self.K))
            candidates.append(alpha * B_i)

        best_x, best_val = X[i].copy(), -np.inf
        for x0 in candidates:
            res = minimize(
                lambda x: -self.firm_objective(x, i, X),
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500},
            )
            val = -res.fun
            if res.success and val > best_val:
                best_val, best_x = val, np.clip(res.x, 0, None)

        return best_x

    # ------------------------------------------------------------------
    # Nash equilibrium finders
    # ------------------------------------------------------------------

    def find_nash_br(
        self,
        X0: Optional[np.ndarray] = None,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Best-response dynamics (sequential / Gauss-Seidel).

        Each firm takes turns computing its best response to the current
        strategies of all others.  Convergence is guaranteed for concave games
        where the best-response map is a contraction; otherwise may cycle.
        """
        X = self._init_X(X0)
        sw_history = [self.social_welfare(X)]

        for t in range(max_iter):
            X_prev = X.copy()
            for i in range(self.N):
                X[i] = self.best_response(i, X)
            sw_history.append(self.social_welfare(X))
            delta = np.max(np.abs(X - X_prev))
            if verbose:
                print(f"  BR iter {t+1:3d}  |ΔX|={delta:.2e}  SW={sw_history[-1]:.6f}")
            if delta < tol:
                if verbose:
                    print(f"  Converged at iteration {t+1}.")
                break

        return X, sw_history

    def find_nash_gradient(
        self,
        X0: Optional[np.ndarray] = None,
        lr: float = 1e-2,
        max_iter: int = 5000,
        tol: float = 1e-8,
        verbose: bool = False,
    ) -> tuple[np.ndarray, list[float]]:
        """
        Simultaneous projected gradient ascent.

        All firms move along their own gradient simultaneously, then project
        onto the budget constraint.  Faster per step than BR dynamics but may
        not converge for non-potential games; use a small learning rate or
        combine with line search.
        """
        X = self._init_X(X0)
        sw_history = [self.social_welfare(X)]

        for t in range(max_iter):
            grads = self._gradients(X)
            X_new = self._project_budgets(X + lr * grads)
            delta = np.max(np.abs(X_new - X))
            X = X_new
            sw_history.append(self.social_welfare(X))
            if verbose and t % 500 == 0:
                print(f"  Grad iter {t:5d}  |ΔX|={delta:.2e}  SW={sw_history[-1]:.6f}")
            if delta < tol:
                if verbose:
                    print(f"  Converged at iteration {t+1}.")
                break

        return X, sw_history

    # ------------------------------------------------------------------
    # Social optimum
    # ------------------------------------------------------------------

    def social_optimum(
        self,
        X0: Optional[np.ndarray] = None,
        n_restarts: int = 6,
    ) -> tuple[np.ndarray, float]:
        """
        Find the socially optimal allocation maximizing SW = sum_k v_k max_i f(x_{i,k}).

        Non-convex in general (max_i is a pointwise maximum; SW is a sum of
        pointwise maxima which is convex in x, so we are maximizing a convex
        function — NP-hard in the worst case).  SLSQP with random restarts
        finds good solutions for moderate N, K.
        """
        NK = self.N * self.K

        def neg_sw(x_flat):
            return -self.social_welfare(x_flat.reshape(self.N, self.K))

        bounds = [(0.0, None)] * NK
        constraints = [
            {"type": "ineq", "fun": lambda x, i=i: self.B[i] - np.sum(x.reshape(self.N, self.K)[i])}
            for i in range(self.N)
        ]

        rng = np.random.default_rng(42)
        best_X, best_sw = None, -np.inf

        x0_list = [self._init_X(X0).flatten()]
        for _ in range(n_restarts - 1):
            rows = [rng.dirichlet(np.ones(self.K)) * self.B[i] for i in range(self.N)]
            x0_list.append(np.concatenate(rows))

        for x0 in x0_list:
            res = minimize(neg_sw, x0, method="SLSQP", bounds=bounds,
                           constraints=constraints, options={"ftol": 1e-10, "maxiter": 1000})
            sw = -res.fun
            if sw > best_sw:
                best_sw, best_X = sw, res.x.reshape(self.N, self.K)

        return best_X, best_sw

    # ------------------------------------------------------------------
    # KKT marginals (for analysis)
    # ------------------------------------------------------------------

    def kkt_marginals(self, X: np.ndarray) -> np.ndarray:
        """
        Return the marginal win-rate gain ∂U_i/∂x_{i,k} for each (i, k).

        At a Nash equilibrium with a binding budget, these should be equal
        across all k where x_{i,k} > 0 (equimarginal principle / water-filling).

        ∂U_i/∂x_{i,k} = f'(x_{i,k}) * sum_{i'≠i} w_k * σ'(f(x_{i,k}) - f(x_{i',k}))
        """
        th = self.theta(X)
        fp = self._f_prime_vals(X)
        marginals = np.zeros((self.N, self.K))
        for i in range(self.N):
            for ip in range(self.N):
                if ip != i:
                    diff = th[i] - th[ip]
                    sig_prime = expit(diff) * (1 - expit(diff))
                    marginals[i] += self.w * sig_prime * fp[i]
        return marginals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gradients(self, X: np.ndarray) -> np.ndarray:
        """Gradient of U_i w.r.t. x_i for all i.  Equivalent to kkt_marginals."""
        return self.kkt_marginals(X)

    def _f_prime_vals(self, X: np.ndarray) -> np.ndarray:
        if self.f_prime is not None:
            return self.f_prime(X)
        eps = 1e-6
        return (self.f(X + eps) - self.f(X)) / eps

    def _project_budgets(self, X: np.ndarray) -> np.ndarray:
        return np.stack([_project_budget(X[i], self.B[i]) for i in range(self.N)])

    def _init_X(self, X0: Optional[np.ndarray]) -> np.ndarray:
        if X0 is not None:
            return X0.astype(float).copy()
        return np.stack([np.full(self.K, self.B[i] / self.K) for i in range(self.N)])

    def summarize(self, X: np.ndarray, label: str = ""):
        th = self.theta(X)
        objs = self.all_objectives(X)
        sw = self.social_welfare(X)
        marg = self.kkt_marginals(X)
        print(f"\n{'='*55}")
        if label:
            print(f"  {label}")
        print(f"{'='*55}")
        print(f"Allocations X:\n{np.round(X, 4)}")
        print(f"Outputs θ:\n{np.round(th, 4)}")
        print(f"Win-rates:  {np.round(objs, 4)}")
        print(f"Social welfare (v):  {sw:.6f}")
        print(f"KKT marginals:\n{np.round(marg, 4)}")
        budget_use = np.sum(X, axis=1)
        print(f"Budget usage:  {np.round(budget_use, 4)}  /  {np.round(self.B, 4)}")


# ---------------------------------------------------------------------------
# Utility: simplex projection
# ---------------------------------------------------------------------------

def _project_budget(x: np.ndarray, B: float) -> np.ndarray:
    """Project x onto {x >= 0, sum(x) <= B} via isotonic regression."""
    x = np.clip(x, 0.0, None)
    if np.sum(x) <= B:
        return x
    # Project onto scaled simplex {x >= 0, sum(x) = B}
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = np.where(u * np.arange(1, len(u) + 1) > cssv - B)[0]
    rho = rho_candidates[-1]
    tau = (cssv[rho] - B) / (rho + 1.0)
    return np.maximum(x - tau, 0.0)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    # ---- problem setup ----
    N, K = 3, 4
    B = np.array([5.0, 4.0, 3.0])
    w = np.array([0.4, 0.3, 0.2, 0.1])   # evaluator cares most about dim 0
    v = np.array([0.1, 0.2, 0.3, 0.4])   # users care most about dim 3

    # Production function: concave square-root
    def f(x):
        return math.sqrt(max(x, 0.0))

    def f_prime(x):
        return 0.5 / math.sqrt(max(x, 1e-12))

    sim = CompetitionSimulation(N, K, B, w, v, f, f_prime)

    # ---- initial uniform allocation ----
    X0 = sim._init_X(None)
    sim.summarize(X0, label="Initial (uniform budget split)")

    # ---- Nash equilibrium via best-response dynamics ----
    print("\nRunning best-response dynamics...")
    X_nash, sw_hist_br = sim.find_nash_br(X0=X0, max_iter=50, verbose=True)
    sim.summarize(X_nash, label="Nash equilibrium (BR dynamics)")

    # ---- social optimum ----
    print("\nSolving social optimum...")
    X_opt, sw_opt = sim.social_optimum()
    sim.summarize(X_opt, label=f"Social optimum  (SW = {sw_opt:.6f})")

    # ---- price of anarchy ----
    sw_nash = sim.social_welfare(X_nash)
    poa = sw_opt / sw_nash if sw_nash > 0 else float("inf")
    print(f"\nPrice of Anarchy:  SW_opt / SW_Nash = {sw_opt:.4f} / {sw_nash:.4f} = {poa:.4f}")
