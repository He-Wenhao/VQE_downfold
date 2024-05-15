# %%
import openfermion as of

# %%
from scipy.optimize import minimize
from sympy import symbols
from pytket.extensions.qiskit import AerBackend
from pytket.circuit import Circuit, Qubit
from pytket.partition import PauliPartitionStrat
from pytket.passes import GuidedPauliSimp, FullPeepholeOptimise
from pytket.pauli import Pauli, QubitPauliString
from pytket.utils import get_operator_expectation_value, gen_term_sequence_circuit
from pytket.utils.operators import QubitPauliOperator
from numpy.random import random_sample
#from pytket.extensions.quantinuum import QuantinuumBackend
from scipy.linalg import eig
from pytket.partition import measurement_reduction
from pytket.circuit.display import render_circuit_jupyter
from typing import Dict, Tuple
from pytket.partition import MeasurementBitMap
from typing import List
from pytket.partition import MeasurementSetup
from pytket.backends.backendresult import BackendResult
from sympy import *
from typing import Callable
from numpy import ndarray
from pytket.backends.resulthandle import ResultHandle
import numpy as np
from scipy.optimize import approx_fprime
from pytket import OpType
from openfermion.linalg import get_sparse_operator


# %%
#quantinuum_backend = QuantinuumBackend(device_name="H1-1E")
#quantinuum_backend.login()

# %%
aer_backend = AerBackend()

# %%
from qiskit_aer.noise import NoiseModel
from qiskit_aer import noise

# %%
def VQE_solver(hamiltonian,noise=False):
    if noise:
        noiseModel = noise.NoiseModel()
        noisy_aer_backend = AerBackend(noise_model = noiseModel)
    # ED solve the qubit Hamiltonian
    def ED_solve_JW(new_jw_hamiltonian):
        new_jw_matrix = get_sparse_operator(new_jw_hamiltonian)
        new_eigenenergies, new_eigenvecs = np.linalg.eigh(new_jw_matrix.toarray())
        return new_eigenenergies[0]
    print('dbg_input energy:',ED_solve_JW(hamiltonian))
    print('dbg_Hamiltonian:',hamiltonian)

    # %%
    def qps_from_openfermion(term):
        """Convert OpenFermion term of Paulis to pytket QubitPauliString, ensuring all qubits are included."""
        # Initialize with identity operations for all qubits
        qubit_pauli_map = {Qubit(i): Pauli.I for i in range(4)}
        
        if term:  # If the term is not empty
            for qubit, pauli in term:
                qubit_pauli_map[Qubit(qubit)] = pauli_sym[pauli]

        return QubitPauliString(qubit_pauli_map)

    def qpo_from_openfermion(openf_op):
        """Convert OpenFermion QubitOperator to pytket QubitPauliOperator."""
        tk_op = dict()
        for term, coeff in openf_op.terms.items():
            qps = qps_from_openfermion(term)
            tk_op[qps] = coeff
        return QubitPauliOperator(tk_op)

    # %%
    pauli_sym = {"I": Pauli.I, "X": Pauli.X, "Y": Pauli.Y, "Z": Pauli.Z}

    # %%
    hamiltonian_op = qpo_from_openfermion(hamiltonian)

    # %%
    print(hamiltonian_op)

    # %%
    sm = hamiltonian_op.to_sparse_matrix().toarray()
    ground_state_energy = eig(sm)[0]
    ground_state_energy = np.sort(ground_state_energy)
    ground_state_energy = ground_state_energy[0].real
    print(f"{ground_state_energy} Ha")
    #return ground_state_energy
    #return ground_state_energy # dbg
    # %%
    strat = PauliPartitionStrat.NonConflictingSets
    pauli_strings = [term for term in hamiltonian_op._dict.keys()]
    measurement_setup = measurement_reduction(pauli_strings, strat)

    # %%
    for measurement_subcircuit in measurement_setup.measurement_circs:
        render_circuit_jupyter(measurement_subcircuit)

    # %%
    # Define the symbolic parameter for the rotation angle
    sym = symbols("theta")

    def generate_circuit(pauli_string):
        c = Circuit(4)
        
        # Initial sequence based on Pauli operators
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                c.H(i)
            elif pauli == 'Y':
                c.Rx(-0.5, i)
            # Assuming no operation is added for 'Z' in the initial part

        # Apply CNOT gates
        for i in range(3):
            c.CX(i, i+1)
        
        # Apply the Rz gate to the last qubit with sign adjustment
        if pauli_string in ['XXYX', 'YXXX', 'YYYX', 'YXYY']:
            c.Rz(-1 * sym, 3)  # Negative theta for these specific strings
        else:
            c.Rz(sym, 3)
            
        # Apply CNOT gates
        for i in range(3):
            c.CX(i, i+1)
        
        # Final sequence (mirror the initial gates)
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                c.H(i)
            elif pauli == 'Y':
                c.Rx(0.5, i)
            # Assuming no operation is added for 'Z' in the final part

        return c

    # Pauli strings for your problem
    pauli_strings = ['XXXY', 'XXYX', 'XYXX', 'YXXX', 'YYYX', 'YYXY', 'YXYY', 'XYYY']

    # Generate and store circuits for each Pauli string
    circuits = {pauli: generate_circuit(pauli) for pauli in pauli_strings}
    # Assuming 'circuits' is your dictionary of circuits from previous steps
    ucc = Circuit(4).X(0).X(1)

    # Add each circuit to the main circuit
    for pauli, circuit in circuits.items():
        ucc.append(circuit)

    #render_circuit_jupyter(ucc)
    GuidedPauliSimp().apply(ucc)
    FullPeepholeOptimise().apply(ucc)
    #render_circuit_jupyter(ucc)
        

    # %%
    print("CX depth after PS+FPO", ucc.depth_by_type(OpType.CX))

    # %%
    def objective_test(params):
        circ = ucc.copy()
        # Assuming we only have one parameter, you should create a symbolic variable if not already done
        theta = symbols('theta')  # can replace 'theta' with the name of your actual symbol if different
        # Create a symbol map using the single symbol and the passed parameter value
        sym_map = {theta: params[0]}
        circ.symbol_substitution(sym_map)
        return (
            get_operator_expectation_value(
                circ,
                hamiltonian_op,
                aer_backend, #change backend between quantinuum_backend and aer_backend for benchmarks
                n_shots=200,
                partition_strat=PauliPartitionStrat.NonConflictingSets,
            )
        ).real

    initial_params = np.array([1.3])
    result_test = minimize(objective_test, initial_params,
        method='Nelder-Mead',
        #jac=scaled_parameter_shift_gradient, #if you want to use para shift rule, implement it in the objective function
        options={"disp": True, "return_all": True},
        tol=1e-3  # Lower tolerance for finer precision
    )
    print("Final parameter values", result_test.x)
    print("Final energy value", result_test.fun)
        
    return result_test.fun
    # %%
    print('error is:', result_test.fun - ground_state_energy)

    # %% [markdown]
    # # 


if __name__ == '__main__':
    ham = (
        (1.3007238601106832+0j) *of.QubitOperator('') +
        (-0.04020462980098227+0j) *of.QubitOperator('X0 X1 Y2 Y3') +
        (0.04020462980098227+0j) *of.QubitOperator('X0 Y1 Y2 X3') +
        (0.04020462980098227+0j) *of.QubitOperator('Y0 X1 X2 Y3') +
        (-0.04020462980098227+0j) *of.QubitOperator('Y0 Y1 X2 X3') +
        (0.25869154301451336+0j) *of.QubitOperator('Z0') +
        (0.18800463899413097+0j) *of.QubitOperator('Z0 Z1') +
        (0.1452708879592182+0j) *of.QubitOperator('Z0 Z2') +
        (0.18547551776020055+0j) *of.QubitOperator('Z0 Z3') +
        (0.25869154301451336+0j) *of.QubitOperator('Z1') +
        (0.18547551776020055+0j) *of.QubitOperator('Z1 Z2') +
        (0.1452708879592182+0j) *of.QubitOperator('Z1 Z3') +
        (-0.5499573668944435+0j) *of.QubitOperator('Z2') +
        (0.19623437361620866+0j) *of.QubitOperator('Z2 Z3') +
        (-0.5499573668944435+0j) *of.QubitOperator('Z3')
    )
    VQE_solver(ham)

'''(1.3007238601106832+0j) [] +
(-0.04020462980098227+0j) [X0 X1 Y2 Y3] +
(0.04020462980098227+0j) [X0 Y1 Y2 X3] +
(0.04020462980098227+0j) [Y0 X1 X2 Y3] +
(-0.04020462980098227+0j) [Y0 Y1 X2 X3] +
(0.25869154301451336+0j) [Z0] +
(0.18800463899413097+0j) [Z0 Z1] +
(0.1452708879592182+0j) [Z0 Z2] +
(0.18547551776020055+0j) [Z0 Z3] +
(0.25869154301451336+0j) [Z1] +
(0.18547551776020055+0j) [Z1 Z2] +
(0.1452708879592182+0j) [Z1 Z3] +
(-0.5499573668944435+0j) [Z2] +
(0.19623437361620866+0j) [Z2 Z3] +
(-0.5499573668944435+0j) [Z3]'''