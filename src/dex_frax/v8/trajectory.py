"""Sparse constrained trajectory fitting, with fixed timestamps and named speeds.

Angle bounds and adjacent-sample velocity bounds are linear constraints. Solve
them in the pose update itself; projecting an unconstrained Gauss-Newton step
afterward can reject useful coordinated shoulder/elbow/wrist compensation.
"""
import numpy as np
from scipy.sparse import diags, eye, kron, triu, vstack
import osqp


class ConstrainedQPError(ValueError):
    """Keep a finite iterate for independent feasibility recovery/diagnosis."""
    def __init__(self,message,candidate=None):
        super().__init__(message)
        self.candidate=None if candidate is None else np.asarray(candidate).copy()


def _constraints(shape, step):
    n, dofs = shape
    difference = kron(diags([-np.ones(n-1), np.ones(n-1)], [0, 1],
                           shape=(n-1, n)), eye(dofs), format='csc')
    difference = diags(np.tile(1./np.broadcast_to(step, (dofs,)), n-1)) @ difference
    return vstack([eye(n*dofs, format='csc'), difference], format='csc')


def _bounds(shape, step, lower, upper):
    n, dofs = shape
    step = np.broadcast_to(np.asarray(step, dtype=float), (dofs,))
    lower = np.broadcast_to(np.asarray(lower, dtype=float), (dofs,))
    upper = np.broadcast_to(np.asarray(upper, dtype=float), (dofs,))
    if (not np.isfinite(step).all() or np.any(step <= 0)
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper)):
        raise ValueError('Invalid arm trajectory bounds')
    # Tiny inward margin absorbs sparse-solver tolerance and float32 export.
    # It tightens the supplied speed limits, never relaxes them.
    safe_step = step * (1. - 1e-5)
    return (np.r_[np.tile(lower, n), -np.full(dofs*(n-1), 1.-1e-5)],
            np.r_[np.tile(upper, n), np.full(dofs*(n-1), 1.-1e-5)], safe_step)


def _qp(P, gradient, A, lower, upper, initial, *, scaling=0, require_optimal=True,
        feasible_tolerance=1e-7):
    solver = osqp.OSQP()
    solver.setup(P=triu(P, format='csc'), q=np.asarray(gradient), A=A,
                 l=lower, u=upper, verbose=False, eps_abs=2e-8, eps_rel=2e-8,
                 max_iter=30000, polishing=True, adaptive_rho_interval=50,
                 check_termination=25, scaling=scaling)
    solver.warm_start(x=np.asarray(initial))
    result = solver.solve(raise_error=False)
    if result.x is None or not np.isfinite(result.x).all():
        raise ValueError('Arm constrained QP failed: '+result.info.status)
    actual = A @ result.x
    excess = max(float(np.max(lower-actual)), float(np.max(actual-upper)), 0.)
    status_ok=result.info.status_val in (1,2) or (not require_optimal and result.info.status_val==7 and excess<=feasible_tolerance)
    if not status_ok or excess > 1e-6:
        raise ConstrainedQPError(f'Constrained QP failed: {result.info.status}; excess={excess:.3g}',result.x)
    return result.x, dict(status=result.info.status, iterations=int(result.info.iter),
                         primal_residual=float(result.info.prim_res),
                         dual_residual=float(result.info.dual_res), constraint_excess=excess,
                         optimal_status=result.info.status_val in (1,2))


def _roundoff_cleanup(q, step, lower, upper):
    """Remove only QP roundoff; the optimizer has already enforced velocity."""
    out = np.clip(q, lower, upper)
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], np.maximum(lower, out[i-1]-step),
                         np.minimum(upper, out[i-1]+step))
    return out


def project_velocity(q, step, lower, upper):
    """Minimum squared joint change under angle and per-axis speed bounds."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('Expected finite arm trajectory with at least two frames')
    low, high, safe_step = _bounds(q.shape, step, lower, upper)
    if (np.all(q >= lower) and np.all(q <= upper)
            and np.all(abs(np.diff(q, axis=0)) <= safe_step)):
        return q.copy()
    A = _constraints(q.shape, step)
    fitted, _ = _qp(eye(q.size, format='csc'), -q.ravel(), A, low, high,
                    np.clip(q, lower, upper).ravel())
    return _roundoff_cleanup(fitted.reshape(q.shape), safe_step, lower, upper)


def constrained_refine(q, fun, jac, lower, upper, step, iterations=24):
    """Sequential quadratic pose fitting with all seven axes jointly bounded.

    Every QP update is constrained before its nonlinear pose cost is evaluated.
    A line search between feasible poses preserves both angle and speed limits.
    No timestamps change, and no infeasible candidate is exported as a success.
    """
    q = np.asarray(q, dtype=float)
    shape = q.shape
    A = _constraints(shape, step)
    low, high, safe_step = _bounds(shape, step, lower, upper)
    z = project_velocity(q, step, lower, upper).ravel()
    res = fun(z); cost = float(res @ res); initial_cost = cost
    history = []; trust = .4; damping = .01
    for k in range(iterations):
        J = jac(z)
        P = (J.T @ J).tocsc() + eye(z.size, format='csc') * damping
        gradient = np.asarray(J.T @ res).ravel()
        at = A @ z
        dl, du = low-at, high-at
        dl[:z.size] = np.maximum(dl[:z.size], -trust)
        du[:z.size] = np.minimum(du[:z.size], trust)
        delta, info = _qp(P, gradient, A, dl, du, np.zeros(z.size))
        candidate = _roundoff_cleanup((z+delta).reshape(shape), safe_step, lower, upper).ravel()
        delta = candidate-z
        accepted = False
        for alpha in (1., .5, .25, .125, .0625):
            proposal = z+alpha*delta
            new_res = fun(proposal); new_cost = float(new_res @ new_res)
            if new_cost < cost-1e-8:
                improvement = cost-new_cost
                z, res, cost = proposal, new_res, new_cost
                history.append(dict(iteration=k+1, cost=cost, alpha=alpha, qp=info))
                accepted = True
                if alpha == 1.: trust = min(.7, trust*1.25)
                break
        if not accepted:
            trust *= .5; damping *= 10.
            if trust < .025 or np.max(abs(delta)) < 1e-5: break
            continue
        damping = max(.001, damping*.5)
        if improvement < 1e-6 * max(cost, 1.): break
    out = _roundoff_cleanup(z.reshape(shape), safe_step, lower, upper)
    if np.any(abs(np.diff(out, axis=0)) > np.asarray(step)+1e-9):
        raise ValueError('Arm constrained fit failed final per-axis speed check')
    final_res = fun(out.ravel())
    return out, dict(method='sparse_sqp_pose_and_velocity_v2', iterations=len(history),
                     initial_cost=initial_cost, cost=float(final_res @ final_res),
                     max_step_deg=float(np.rad2deg(np.max(abs(np.diff(out, axis=0))))),
                     history=history)
