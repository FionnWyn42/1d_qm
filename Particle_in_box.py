import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import streamlit.components.v1 as components

# --- Physics Constants ---
L = 1.0  # Box length
hbar = 1.0
m = 1.0

# --- Energy eigenfunction ---
def psi_n(n, x, t):
    En = (n**2 * np.pi**2 * hbar**2) / (2 * m * L**2)
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L) * np.exp(-1j * En * t / hbar)

# --- Superposition wavefunction ---
def psi_superposition(coeffs, x, t):
    psi_total = np.zeros_like(x, dtype=np.complex128)
    for n, c in enumerate(coeffs, start=1):
        psi_total += c * psi_n(n, x, t)
    return psi_total

# --- Normalize coefficients ---
def normalize(coeffs):
    norm = np.sqrt(np.sum(np.abs(coeffs)**2))
    return coeffs / norm if norm != 0 else coeffs

# --- Streamlit UI ---
st.title("🔊 Particle in a 1D Box: Live Animation")
mode = st.radio("Select Mode", ["Single Energy Level", "Superposition"])

if mode == "Single Energy Level":
    n = st.slider("Select Energy Level (n)", 1, 10, 1)
    coeffs = np.zeros(n, dtype=complex)
    coeffs[n - 1] = 1.0
else:
    max_n = st.slider("Max Energy Level (n)", 2, 10, 3)
    coeffs = []
    for i in range(1, max_n + 1):
        real = st.slider(f"Re(c{i})", -1.0, 1.0, 0.0, key=f"r{i}")
        imag = st.slider(f"Im(c{i})", -1.0, 1.0, 0.0, key=f"i{i}")
        coeffs.append(complex(real, imag))
    coeffs = normalize(np.array(coeffs))

# --- Animation Setup ---
x = np.linspace(0, L, 500)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

line_re, = ax1.plot([], [], 'b-', label='Re(ψ)')
line_im, = ax1.plot([], [], 'r-', label='Im(ψ)')
ax1.set_xlim(0, L)
ax1.set_ylim(-2, 2)
ax1.set_ylabel('ψ(x, t)')
ax1.legend()
ax1.grid()

line_prob, = ax2.plot([], [], 'g-', label='|ψ|²')
ax2.set_xlim(0, L)
ax2.set_ylim(0, 4)
ax2.set_xlabel('x')
ax2.set_ylabel('|ψ(x, t)|²')
ax2.legend()
ax2.grid()

# --- Animation update function ---
def update(frame):
    t = frame * 0.05
    psi_t = psi_superposition(coeffs, x, t)
    line_re.set_data(x, np.real(psi_t))
    line_im.set_data(x, np.imag(psi_t))
    line_prob.set_data(x, np.abs(psi_t)**2)
    return line_re, line_im, line_prob

ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

# --- Embed animation in Streamlit using HTML ---
st.write("### Animation (JS-based, interactive)")
components.html(ani.to_jshtml(), height=800)
    
