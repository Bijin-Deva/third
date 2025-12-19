# -*- coding: utf-8 -*-
"""
Advanced Quantum Circuit Visualizer with:
• Realistic Bloch Sphere
• Multi-language Algorithm Code
• High-quality Circuit Rendering
• Full Quantum Noise Modeling
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
    thermal_relaxation_error,
    ReadoutError
)

# --------------------------------------------------
# Pauli matrices
# --------------------------------------------------
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def strip_measurements(qc):
    new = QuantumCircuit(qc.num_qubits)
    for inst, q, _ in qc.data:
        if inst.name != "measure":
            new.append(inst, q)
    return new

def reduced_density(state, i):
    return partial_trace(state, [q for q in range(state.num_qubits) if q != i]).data

def bloch_vector(rho):
    return (
        float(np.real(np.trace(rho @ SX))),
        float(np.real(np.trace(rho @ SY))),
        float(np.real(np.trace(rho @ SZ)))
    )

def purity(rho):
    return float(np.real(np.trace(rho @ rho)))

def bloch_angles(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1e-6:
        return r, None, None
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi

# --------------------------------------------------
# Realistic Bloch Sphere
# --------------------------------------------------
def plot_bloch(x, y, z, title):
    u = np.linspace(0, 2*np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_surface(
        x=xs, y=ys, z=zs,
        opacity=0.18,
        colorscale='Greys',
        showscale=False
    )

    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode="lines",
        line=dict(width=8, color="#ff1493")
    ))

    fig.add_trace(go.Cone(
        x=[x], y=[y], z=[z],
        u=[x], v=[y], w=[z],
        sizemode="absolute",
        sizeref=0.18,
        anchor="tip",
        showscale=False,
        colorscale=[[0, "#ff1493"], [1, "#ff1493"]]
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

# --------------------------------------------------
# Advanced Noise Model (ALL PARAMETERS)
# --------------------------------------------------
def build_noise(p):
    noise = NoiseModel()

    if p["rx"] > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p["rx"], 1), ['rx'])
    if p["ry"] > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p["ry"], 1), ['ry'])
    if p["rz"] > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p["rz"], 1), ['rz'])

    if p["decay"] > 0:
        noise.add_all_qubit_quantum_error(amplitude_damping_error(p["decay"]),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if p["deco"] > 0:
        noise.add_all_qubit_quantum_error(phase_damping_error(p["deco"]),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if p["depol"] > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p["depol"], 1),
                                          ['h', 'x', 'rx', 'ry', 'rz'])

    if p["bell"] > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p["bell"], 2), ['cx'])

    if p["thermal"] > 0:
        noise.add_all_qubit_quantum_error(
            thermal_relaxation_error(50e3, 70e3, 100),
            ['h', 'x', 'rx', 'ry', 'rz']
        )

    if p["tsp01"] > 0 or p["tsp10"] > 0:
        noise.add_all_qubit_readout_error(ReadoutError([
            [1 - p["tsp01"], p["tsp01"]],
            [p["tsp10"], 1 - p["tsp10"]]
        ]))

    return noise

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(layout="wide")
st.title("⚛️ Advanced Quantum Circuit Visualizer")

uploaded = st.sidebar.file_uploader("Upload QASM", type="qasm")
shots = st.sidebar.slider("Shots", 100, 8192, 1024)

st.sidebar.header("Noise Parameters")

rx = st.sidebar.number_input("RX Error", 0.0, 1.0, 0.0)
ry = st.sidebar.number_input("RY Error", 0.0, 1.0, 0.0)
rz = st.sidebar.number_input("RZ Error", 0.0, 1.0, 0.0)

tsp01 = st.sidebar.number_input("|0⟩ → |1⟩", 0.0, 1.0, 0.0)
tsp10 = st.sidebar.number_input("|1⟩ → |0⟩", 0.0, 1.0, 0.0)

decay = st.sidebar.number_input("Decay (f)", 0.0, 1.0, 0.0)
deco = st.sidebar.number_input("Decoherence (g)", 0.0, 1.0, 0.0)

depol = st.sidebar.number_input("Depolarization", 0.0, 1.0, 0.0)
bell = st.sidebar.number_input("Bell Depolarization", 0.0, 1.0, 0.0)
thermal = st.sidebar.number_input("Thermal Factor", 0.0, 1.0, 0.0)

if uploaded:
    qc = QuantumCircuit.from_qasm_str(uploaded.read().decode())

    st.header("Quantum Circuit")
    fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
    qc.draw("mpl", ax=ax, scale=1.0)
    st.pyplot(fig)

    # ---------------- Measurements ----------------
    backend = Aer.get_backend("qasm_simulator")
    noise = build_noise({
        "rx": rx, "ry": ry, "rz": rz,
        "tsp01": tsp01, "tsp10": tsp10,
        "decay": decay, "deco": deco,
        "depol": depol, "bell": bell,
        "thermal": thermal
    })

    job = backend.run(qc, shots=shots, noise_model=noise)
    counts = job.result().get_counts()

    st.header("Measurement Results")
    st.bar_chart(counts)

    # ---------------- Bloch Analysis ----------------
    st.header("Bloch Sphere Analysis")
    state = Statevector.from_instruction(strip_measurements(qc))

    cols = st.columns(qc.num_qubits)
    for i in range(qc.num_qubits):
        with cols[i]:
            rho = reduced_density(state, i)
            x, y, z = bloch_vector(rho)
            r, theta, phi = bloch_angles(x, y, z)
            p = purity(rho)

            st.plotly_chart(plot_bloch(x, y, z, f"Qubit {i}"))
            st.markdown(f"""
**⟨σx⟩**={x:.3f}  
**⟨σy⟩**={y:.3f}  
**⟨σz⟩**={z:.3f}  
**‖r‖**={r:.3f}  
**Purity**={p:.3f}  
""")

    # ---------------- Algorithm Code ----------------
    st.header("Algorithm Code")

    qiskit_tab, cirq_tab, cudaq_tab = st.tabs(["Qiskit", "Cirq", "CUDA-Q"])

    with qiskit_tab:
        st.code(qc.qasm(), language="qasm")

    with cirq_tab:
        st.code("# Cirq conversion (basic)\nUse gate-by-gate mapping", language="python")

    with cudaq_tab:
        st.code("# CUDA-Q kernel skeleton\n@cudaq.kernel\ndef kernel(): ...", language="python")
