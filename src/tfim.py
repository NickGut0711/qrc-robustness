from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import List, Optional
from qutip import Qobj, qeye, tensor, sigmax, sigmaz, basis, expect, propagator, mesolve, ket2dm

def chain_adjacency(N: int) -> NDArray[np.float64]:
    A = np.zeros((N, N), dtype=float)
    for i in range(N - 1):
        A[i, i + 1] = A[i + 1, i] = 1.0
    return A

class TFIMReservoir:
    """
    QuTiP TFIM reservoir with a trainable classical input layer.
    H_res = sum_{i<j} JZZ[i,j] Z_i Z_j + h sum_i X_i
    H_in(t) = sum_{m,i in input_nodes} W_in[m,i] u_m(t) Z_i
    """
    
    def __init__(self, N: int, J: float, h: float, adj: NDArray[np.float64], W_in: NDArray[np.float64], input_nodes: list[int], dt: float, c_ops: Optional[List[Qobj]] = None,
    ):
        self.N, self.h = int(N), float(h)
        self.dt = float(dt)
        self.c_ops = c_ops or []
        self.W_in = np.array(W_in, dtype=float)           # (M, N)
        self.input_nodes = sorted(set(int(i) for i in input_nodes))

        # --- build JZZ from adjacency + global J (supports weighted adj too) ---
        adj = np.array(adj, dtype=float)
        if adj.shape != (N, N):
            raise ValueError("adj must be (N,N)")
        adj = np.triu(adj, 1)  # keep upper, zero diag
        self.JZZ = J * (adj + adj.T)                      # symmetric, zero diag

        if self.W_in.shape[1] != N:
            raise ValueError("W_in must be (M,N)")

        # mask columns not in input_nodes
        mask = np.zeros(N, dtype=bool); mask[self.input_nodes] = True
        self._W_masked = self.W_in.copy()
        self._W_masked[:, ~mask] = 0.0

        # local ops
        self.Zi, self.Xi = self._local_ops(N)

        # H_res from JZZ
        self.H_res = self._build_H_res()

    # ---- alternate constructors ----
    @classmethod
    def from_adjacency(
        cls, adj: NDArray[np.float64], J: float, h: float,
        W_in: NDArray[np.float64], input_nodes: list[int], dt: float,
        c_ops: Optional[List[Qobj]] = None,
    ):
        N = int(adj.shape[0])
        return cls(N, J, h, adj, W_in, input_nodes, dt, c_ops)

    @classmethod
    def from_couplings(cls, JZZ: NDArray[np.float64], h: float, W_in: NDArray[np.float64], input_nodes: list[int], dt: float, c_ops: Optional[List[Qobj]] = None,
    ):
        JZZ = np.array(JZZ, dtype=float)
        if JZZ.shape[0] != JZZ.shape[1]:
            raise ValueError("JZZ must be square")
        N = int(JZZ.shape[0])

        # start with a dummy adj/J, then overwrite JZZ cleanly
        obj = cls(N, J=0.0, h=h, adj=np.zeros_like(JZZ), W_in=W_in, input_nodes=input_nodes, dt=dt, c_ops=c_ops)
        if not np.allclose(JZZ, JZZ.T): raise ValueError("JZZ must be symmetric")
        if not np.allclose(np.diag(JZZ), 0): raise ValueError("JZZ diagonal must be zero")
        obj.JZZ = JZZ
        obj.H_res = obj._build_H_res()
        return obj

    # ---- helper functions ----
    @staticmethod
    def _site_op(op: Qobj, i: int, N: int) -> Qobj:
        ops = [qeye(2)] * N
        ops[i] = op
        return tensor(ops)

    def _local_ops(self, N: int):
        Zi = [self._site_op(sigmaz(), i, N) for i in range(N)]
        Xi = [self._site_op(sigmax(), i, N) for i in range(N)]
        return Zi, Xi

    def _build_H_res(self) -> Qobj:
        Hzz = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                Jij = float(self.JZZ[i, j])
                if Jij != 0.0:
                    Hzz = Hzz + Jij * (self.Zi[i] * self.Zi[j])
        Hx = sum(self.Xi) if self.N > 0 else 0
        return Hzz + self.h * Hx

    # ---- input layer & evolution functions ----
    def H_in(self, u_t: NDArray[np.float64]) -> Qobj:
        """H_in(t)=sum_i [(W_in^T u_t)_i * Z_i], masked to input_nodes."""
        M, N = self._W_masked.shape
        if u_t.shape != (M,): raise ValueError(f"u_t must be shape ({M},)")
        coeffs = self._W_masked.T @ u_t
        Hin = 0
        for i in range(N):
            c = float(coeffs[i])
            if c != 0.0:
                Hin = Hin + c * self.Zi[i]
        return Hin if isinstance(Hin, Qobj) else 0 * self.Zi[0]

    def init_state(self, kind: str = "plusx") -> Qobj:
        v = (basis(2,0) + basis(2,1)).unit() if kind == "plusx" else basis(2,0)
        psi = v
        for _ in range(self.N - 1): psi = tensor(psi, v)
        return psi

    def features_Z(self, state: Qobj) -> NDArray[np.float64]:
        return np.array([np.real(expect(Z, state)) for Z in self.Zi], dtype=float)

    def step(self, state: Qobj, u_t: NDArray[np.float64]) -> Qobj:
        Ht = self.H_res + self.H_in(u_t)
        if len(self.c_ops) == 0:  # unitary
            U_dt = propagator(Ht, self.dt)
            return U_dt * state
        rho0 = state if state.isoper else ket2dm(state)  # Lindblad
        return mesolve(Ht, rho0, tlist=[0.0, self.dt], c_ops=self.c_ops).states[-1]

    def evolve(self, U_seq: NDArray[np.float64], state0: Optional[Qobj] = None, collect_features: bool = True):
        T, M = U_seq.shape
        if M != self._W_masked.shape[0]:
            raise ValueError("U_seq.shape[1] must equal number of input channels M.")
        st = self.init_state() if state0 is None else state0
        Phi = np.zeros((T, self.N), dtype=float) if collect_features else None
        for t in range(T):
            st = self.step(st, U_seq[t])
            if collect_features: Phi[t] = self.features_Z(st)
        return st, Phi