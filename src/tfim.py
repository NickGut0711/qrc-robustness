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

