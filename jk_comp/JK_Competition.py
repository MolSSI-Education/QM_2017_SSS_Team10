import numpy as np 
import psi4
import build.basic_mod as bm
import timeit

# Make sure we get the same random array
np.random.seed(0)

# A hydrogen molecule
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")

mol.update_geometry()

# Build a ERI tensor
basis = psi4.core.BasisSet.build(mol, target="cc-pVTZ")
mints = psi4.core.MintsHelper(basis)
I = np.array(mints.ao_eri())


# Symmetric random density
nbf = I.shape[0]
D = np.random.rand(nbf, nbf)
D = (D + D.T) / 2

# Reference
J_ref = np.einsum("pqrs,rs->pq", I, D)
K_ref = np.einsum("prqs,rs->pq", I, D)

# Your implementation
np.array(I, copy=False)
np.array(D, copy=False)
J = np.empty([I.shape[0], I.shape[1]])
bm.einJ(I, D, J)
#J = np.random.rand(nbf, nbf)
K = np.empty([I.shape[0], I.shape[1]])
bm.einK(I, D, K)
#K = np.random.rand(nbf, nbf)

# Make sure your implementation is correct
print("J is correct: %s" % np.allclose(J, J_ref))
print("K is correct: %s" % np.allclose(K, K_ref))

