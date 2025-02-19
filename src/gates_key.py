import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

theta = Parameter("θ")
fig, axs = plt.subplots(1, 6, figsize=(14, 2))

qc_line = QuantumCircuit(1)
qc_line.id(0)
qc_line.draw('mpl', ax=axs[0], idle_wires=False)
axs[0].set_title("Qubit line")

qc_h = QuantumCircuit(1)
qc_h.h(0)
qc_h.draw('mpl', ax=axs[1], idle_wires=False)
axs[1].set_title("H Gate")

qc_p = QuantumCircuit(1)
qc_p.p(theta, 0)
qc_p.draw('mpl', ax=axs[2], idle_wires=False)
axs[2].set_title("P(θ)")

qc_sx = QuantumCircuit(1)
qc_sx.sx(0)
qc_sx.draw('mpl', ax=axs[3], idle_wires=False)
axs[3].set_title("SX")

qc_sxdg = QuantumCircuit(1)
qc_sxdg.sxdg(0)
qc_sxdg.draw('mpl', ax=axs[4], idle_wires=False)
axs[4].set_title("SX†")

qc_cnot = QuantumCircuit(2)
qc_cnot.cx(0, 1)
qc_cnot.draw('mpl', ax=axs[5], idle_wires=False)
axs[5].set_title("CNOT")

plt.tight_layout()
plt.savefig("qiskit_circuit_key.png", dpi=300)
plt.show()
