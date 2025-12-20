# app.py — FINAL CLEAN VERSION
# Noise correctly affects Bloch spheres and measurement

import streamlit as st
import numpy as np
import io
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit_aer import Aer, AerSimulator
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    amplitude_damping_error,
    phase_damping_error,
    ReadoutError
)

# -------------------------------------------------
# Pauli matrices
# -------------------------------------------------
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)

# -------------------------------------------------
# Helper functions
# -------------------------------------------------
def remove_final_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    qc2 = QuantumCircuit(qc.num_qubits)
    for instr, qargs, cargs in qc.data:
        if instr.name != "measure":
            qc2.append(instr, qargs)
    return qc2


def statevector_from_circuit(qc: QuantumCircuit) -> Statevector:
    qc2 = remove_final_measurements(qc)
    return Statevector.from_instruction(qc2)


def noisy_density_matrix(qc: QuantumCircuit, noise_model: NoiseModel) -> DensityMatrix:
    qc2 = remove_final_measurements(qc)
    qc2.save_density_matrix()

    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    result = sim.run(qc2).result()
    return DensityMatrix(result.data(0)["density_matrix"])


def bloch_vector_from_rho(rho: np.ndarray):
    x = np.real(np.trace(rho @ SX))
    y = np.real(np.trace(rho @ SY))
    z = np.real(np.trace(rho @ SZ))
    return x, y, z


def purity_from_rho(rho: np.ndarray):
    return np.real(np.trace(rho @ rho))


# -------------------------------------------------
# Noise model
# -------------------------------------------------
def build_noise_model(depol, t1, t2, ro_01, ro_10):
    noise = NoiseModel()

    if depol > 0:
        noise.add_all_qubit_quantum_error(
            depolarizing_error(depol, 1),
            ['h', 'x', 'y', 'z']
        )
        noise.add_all_qubit_quantum_error(
            depolarizing_error(depol, 2),
            ['cx']
        )

    if t1 > 0:
        noise.add_all_qubit_quantum_error(
            amplitude_damping_error(t1),
            ['h', 'x', 'y', 'z']
        )

    if t2 > 0:
        noise.add_all_qubit_quantum_error(
            phase_damping_error(t2),
            ['h', 'x', 'y', 'z']
        )

    if ro_01 > 0 or ro_10 > 0:
        noise.add_all_qubit_readout_error(
            ReadoutError([
                [1 - ro_01, ro_01],
                [ro_10, 1 - ro_10]
            ])
        )

    return noise


# -------------------------------------------------
# Bloch sphere plot
# -------------------------------------------------
def plot_bloch_sphere(x, y, z, title):
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    sx = np.outer(np.cos(u), np.sin(v))
    sy = np.outer(np.sin(u), np.sin(v))
    sz = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()

    fig.add_surface(
        x=sx, y=sy, z=sz,
        opacity=0.15,
        colorscale='Greys',
        showscale=False
    )

    # Axes
    for axis, label in [([1,0,0],'X'), ([0,1,0],'Y'), ([0,0,1],'|0⟩')]:
        fig.add_trace(go.Scatter3d(
            x=[-1.2*axis[0], 1.2*axis[0]],
            y=[-1.2*axis[1], 1.2*axis[1]],
            z=[-1.2*axis[2], 1.2*axis[2]],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False
        ))

    r = np.sqrt(x*x + y*y + z*z)
    if r > 1e-3:
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines',
            line=dict(color='#FF1493', width=8),
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers',
            marker=dict(size=6, color='#FF1493'),
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
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Quantum Noise Visualizer", layout="wide")
st.title("⚛️ Quantum Circuit Visualizer")

qasm_text = st.text_area(
    "Paste OpenQASM code here",
    value="""OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
h q[0];
measure q -> c;
""",
    height=200
)

st.sidebar.header("Quantum Noise")
enable_noise = st.sidebar.checkbox("Enable Noise")

depol = st.sidebar.slider("Depolarization", 0.0, 0.3, 0.0)
t1 = st.sidebar.slider("Amplitude Damping (T1)", 0.0, 0.3, 0.0)
t2 = st.sidebar.slider("Phase Damping (T2)", 0.0, 0.3, 0.0)
ro_01 = st.sidebar.slider("|0⟩ → |1⟩ Readout", 0.0, 0.3, 0.0)
ro_10 = st.sidebar.slider("|1⟩ → |0⟩ Readout", 0.0, 0.3, 0.0)

if qasm_text.strip():
    try:
        qc = QuantumCircuit.from_qasm_str(qasm_text)

        st.subheader("Circuit Diagram")
        fig, ax = plt.subplots()
        qc.draw("mpl", ax=ax)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Measurement Outcomes")
        qc_meas = qc.copy()
        if qc_meas.num_clbits == 0:
            qc_meas.measure_all()

        noise_model = build_noise_model(depol, t1, t2, ro_01, ro_10) if enable_noise else None
        backend = Aer.get_backend("qasm_simulator")
        job = backend.run(qc_meas, shots=1024, noise_model=noise_model)
        counts = job.result().get_counts()

        st.bar_chart(counts)

        st.subheader("Bloch Sphere Analysis")

        if enable_noise:
            full_dm = noisy_density_matrix(qc, noise_model)
            st.info("Noisy state: Bloch vector shrinks and purity < 1")
        else:
            state = statevector_from_circuit(qc)
            st.info("Ideal state: Bloch vector lies on sphere surface")

        cols = st.columns(qc.num_qubits)
        for i in range(qc.num_qubits):
            with cols[i]:
                if enable_noise:
                    rho = partial_trace(
                        full_dm, [q for q in range(qc.num_qubits) if q != i]
                    ).data
                else:
                    rho = partial_trace(
                        DensityMatrix(state), [q for q in range(qc.num_qubits) if q != i]
                    ).data

                x, y, z = bloch_vector_from_rho(rho)
                purity = purity_from_rho(rho)

                st.plotly_chart(plot_bloch_sphere(x, y, z, f"Qubit {i}"))
                st.metric("Purity", f"{purity:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")
