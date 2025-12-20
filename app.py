# -*- coding: utf-8 -*-
"""
A Streamlit web application for visualizing quantum circuits from .qasm files,
with a modern light theme and Bloch sphere visualization.
"""

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit_aer import Aer
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError
)

# -------------------------------------------------
# Example circuits
# -------------------------------------------------
EXAMPLES = {
    "Single Qubit Superposition": {
        "qasm": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q -> c;
""",
        "note": "Single qubit in |+⟩ state (pure state)."
    },
    "Bell State": {
        "qasm": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q -> c;
""",
        "note": "Two-qubit entangled Bell state."
    }
}

# -------------------------------------------------
# Pauli matrices
# -------------------------------------------------
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# -------------------------------------------------
# Quantum helpers
# -------------------------------------------------
def remove_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    qc2 = QuantumCircuit(qc.num_qubits)
    for instr, qargs, _ in qc.data:
        if instr.name != "measure":
            qc2.append(instr, qargs)
    return qc2


def statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    return Statevector.from_instruction(remove_final_measurements(qc))


def reduced_density_for_qubit(state: Statevector, idx: int):
    return partial_trace(
        state,
        [q for q in range(state.num_qubits) if q != idx]
    ).data


def bloch_vector_from_rho(rho):
    x = np.real(np.trace(rho @ SX))
    y = np.real(np.trace(rho @ SY))
    z = np.real(np.trace(rho @ SZ))
    return x, y, z


def purity_from_rho(rho):
    return np.real(np.trace(rho @ rho))

# -------------------------------------------------
# Noise model
# -------------------------------------------------
def build_noise_model(depol, t1, t2, ro01, ro10):
    noise = NoiseModel()

    if depol > 0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(depol, 1),
            ['h', 'x']
        )

    if t1 > 0:
        noise.add_all_qubit_quantum_error(
            amplitude_damping_error(t1),
            ['h', 'x']
        )

    if t2 > 0:
        noise.add_all_qubit_quantum_error(
            phase_damping_error(t2),
            ['h', 'x']
        )

    if ro01 > 0 or ro10 > 0:
        noise.add_all_qubit_readout_error(
            ReadoutError([
                [1 - ro01, ro01],
                [ro10, 1 - ro10]
            ])
        )

    return noise

# -------------------------------------------------
# Bloch sphere plot
# -------------------------------------------------
def plot_bloch_sphere(x, y, z, title):
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)

    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(x=sx, y=sy, z=sz, opacity=0.15, showscale=False)

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines+markers',
        line=dict(color='deeppink', width=6),
        marker=dict(size=4),
        showlegend=False
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Quantum Circuit Visualizer", layout="wide")
st.title("⚛️ Quantum Circuit Visualizer")

st.sidebar.header("Circuit Source")
choice = st.sidebar.selectbox(
    "Choose example",
    ["Select"] + list(EXAMPLES.keys())
)

qasm_text = None
note = None

if choice in EXAMPLES:
    qasm_text = EXAMPLES[choice]["qasm"]
    note = EXAMPLES[choice]["note"]

st.sidebar.header("Noise")
enable_noise = st.sidebar.checkbox("Enable Noise")
depol = st.sidebar.slider("Depolarization", 0.0, 0.3, 0.0)
t1 = st.sidebar.slider("Amplitude Damping", 0.0, 0.3, 0.0)
t2 = st.sidebar.slider("Phase Damping", 0.0, 0.3, 0.0)
ro01 = st.sidebar.slider("|0⟩ → |1⟩", 0.0, 0.3, 0.0)
ro10 = st.sidebar.slider("|1⟩ → |0⟩", 0.0, 0.3, 0.0)

if qasm_text:
    qc = QuantumCircuit.from_qasm_str(qasm_text)

    st.subheader("Circuit")
    fig, ax = plt.subplots()
    qc.draw("mpl", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

    state = statevector_from_circuit(qc)

    st.subheader("Bloch Spheres")
    cols = st.columns(qc.num_qubits)

    for i in range(qc.num_qubits):
        with cols[i]:
            rho = reduced_density_for_qubit(state, i)
            x, y, z = bloch_vector_from_rho(rho)
            p = purity_from_rho(rho)

            st.plotly_chart(plot_bloch_sphere(x, y, z, f"Qubit {i}"))
            st.metric("Purity", f"{p:.4f}")

    if note:
        st.info(note)
