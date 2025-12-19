# -*- coding: utf-8 -*-
"""
Quantum Circuit Visualizer
- Peach UI
- Default example circuits
- Simple Bloch sphere
- Optional noise model
- Multi-language algorithm code (safe)
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
from qiskit.qasm3 import dumps

# ==================================================
# UI THEME (PEACH)
# ==================================================
st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #FFFFFF;  /* White */
}

/* Sidebar background */
[data-testid="stSidebar"] {
    background-color: #E6F2FF;  /* Light blue */
}

/* Improve text contrast */
[data-testid="stSidebar"] * {
    color: #000000;
}
</style>
""", unsafe_allow_html=True)


# ==================================================
# DEFAULT EXAMPLES
# ==================================================
EXAMPLES = {
    "Single Qubit Superposition": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q -> c;
""",

    "Bell State": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
""",

    "GHZ State": """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[0], q[2];
measure q -> c;
"""
}

# ==================================================
# SIDEBAR
# ==================================================
st.sidebar.header("Circuit Source")

choice = st.sidebar.selectbox(
    "Choose a circuit",
    ["Select example..."] + list(EXAMPLES.keys()) + ["Upload my own"]
)

qasm_text = None

if choice in EXAMPLES:
    qasm_text = EXAMPLES[choice]
elif choice == "Upload my own":
    uploaded = st.sidebar.file_uploader("Upload QASM file", type="qasm")
    if uploaded:
        qasm_text = uploaded.read().decode()

shots = st.sidebar.slider("Shots", 100, 4096, 1024)

# ------------------ Noise ------------------
st.sidebar.header("Noise (Optional)")

use_noise = st.sidebar.checkbox("Enable Noise", value=False)

depol = st.sidebar.slider("Depolarization", 0.0, 0.3, 0.0)
decay = st.sidebar.slider("Amplitude Damping", 0.0, 0.3, 0.0)
phase = st.sidebar.slider("Phase Damping", 0.0, 0.3, 0.0)
tsp01 = st.sidebar.slider("|0⟩ → |1⟩", 0.0, 0.3, 0.0)
tsp10 = st.sidebar.slider("|1⟩ → |0⟩", 0.0, 0.3, 0.0)

# ==================================================
# HELPER FUNCTIONS
# ==================================================
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

def strip_measurements(qc):
    new = QuantumCircuit(qc.num_qubits)
    for inst, q, _ in qc.data:
        if inst.name != "measure":
            new.append(inst, q)
    return new

def reduced_density(state, idx):
    return partial_trace(state, [q for q in range(state.num_qubits) if q != idx]).data

def bloch_vector(rho):
    x = np.real(np.trace(rho @ SX))
    y = np.real(np.trace(rho @ SY))
    z = np.real(np.trace(rho @ SZ))
    return x, y, z

def purity(rho):
    return np.real(np.trace(rho @ rho))

def plot_bloch(x, y, z, title):
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 40)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_surface(x=xs, y=ys, z=zs, opacity=0.15, showscale=False)

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines",
        line=dict(color="crimson", width=6)
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube"
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

def build_noise():
    noise = NoiseModel()

    if depol > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(depol, 1),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if decay > 0:
        noise.add_all_qubit_quantum_error(amplitude_damping_error(decay),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if phase > 0:
        noise.add_all_qubit_quantum_error(phase_damping_error(phase),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if tsp01 > 0 or tsp10 > 0:
        noise.add_all_qubit_readout_error(ReadoutError([
            [1 - tsp01, tsp01],
            [tsp10, 1 - tsp10]
        ]))

    return noise

# ==================================================
# MAIN LOGIC
# ==================================================
if qasm_text:
    qc = QuantumCircuit.from_qasm_str(qasm_text)

    st.header("Quantum Circuit")

    fig, ax = plt.subplots(
        figsize=(7, max(2, qc.num_qubits * 0.5)),
        dpi=200
    )
    qc.draw("mpl", ax=ax, scale=0.8)
    st.pyplot(fig)

    # ---------------- Measurement ----------------
    backend = Aer.get_backend("qasm_simulator")
    noise_model = build_noise() if use_noise else None

    job = backend.run(qc, shots=shots, noise_model=noise_model)
    counts = job.result().get_counts()

    st.header("Measurement Results")
    st.bar_chart(counts)

    # ---------------- Bloch ----------------
    st.header("Bloch Sphere Analysis")

    state = Statevector.from_instruction(strip_measurements(qc))
    cols = st.columns(qc.num_qubits)

    for i in range(qc.num_qubits):
        with cols[i]:
            rho = reduced_density(state, i)
            x, y, z = bloch_vector(rho)
            p = purity(rho)

            st.plotly_chart(plot_bloch(x, y, z, f"Qubit {i}"), use_container_width=True)
            st.metric("Purity", f"{p:.3f}")

    # ---------------- Algorithm Code ----------------
    st.header("Algorithm Code")

    t1, t2, t3 = st.tabs(["Qiskit (QASM)", "Cirq", "CUDA-Q"])

    with t1:
        st.code(dumps(qc), language="qasm")

    with t2:
        st.code("# Cirq example\nimport cirq\ncircuit = cirq.Circuit()", language="python")

    with t3:
        st.code("# CUDA-Q example\n@cudaq.kernel\ndef kernel(): pass", language="python")

else:
    st.info("Select an example circuit or upload a QASM file from the sidebar.")

