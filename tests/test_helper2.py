"""
Test for functions in the helper2.py
"""
import numpy as np
import qm10
import pytest
import psi4

#### Variables for testing
mol = psi4.geometry("""
O
H 1 1.1
H 1 1.1 2 104
""")
bas = psi4.core.BasisSet.build(mol, target="aug-cc-pVDZ")
mints = psi4.core.MintsHelper(bas)
A = np.array([[5.0,5.0], [5.0, 5.0]])
F = np.array([[5.0,5.0], [5.0, 5.0]])
F_old = np.array([[5.0,5.0], [5.0, 5.0]])
F_new = np.array([[6.0,6.0], [6.0, 6.0]])
damp_start = 5
iteration = 1
damp_value = 2.0
nel = 2
H = np.array([[6.0,6.0], [6.0, 6.0]])
E_old =np.float64(50)
D = np.array([[7.0,6.0], [6.0, 6.0]])
S = np.array([[6.0,6.0], [7.0, 6.0]])
######

expected1 = (np.array([[ 50.,50. ],[ 50.,50. ]]), np.array([ 0.,  1000. ]))
def test_update_D():
    assert np.allclose(qm10.helper2.update_D(F, A, nel)[0], expected1[0])
    assert np.allclose(qm10.helper2.update_D(F, A, nel)[1], expected1[1])

expected2 = np.array([[ 6.,  6.], [ 6.,  6.]])
def test_damping_func():
    assert np.allclose(qm10.helper2.damping_func
    (iteration,damp_start, F_old, F_new, damp_value),
    expected2)



expected3 = (np.array([[ 60., 0.],[ -5., -65.]]), 44.300112866673373)
def test_gradient():
    assert np.allclose(qm10.helper2.gradient(F, D, S)[0], expected3[0])
    assert np.allclose(qm10.helper2.gradient(F, D, S)[1], expected3[1])



expected4 = (283.00236645071908, 233.00236645071908)
def test_energy_conv():
    assert np.allclose(qm10.helper2.energy_conv(F, H, D, E_old, mol)[0],
    expected4[0])
    assert np.allclose(qm10.helper2.energy_conv(F, H, D, E_old, mol)[1],
    expected4[1])
