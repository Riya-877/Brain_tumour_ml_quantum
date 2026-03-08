import pennylane as qml
import tensorflow as tf
from pennylane.qnn import KerasLayer

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="tf", diff_method="backprop")
def quantum_circuit(inputs, weights):

    # Remove batch dimension manually
    inputs = tf.reshape(inputs, (-1,))

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    for i in range(n_qubits):
        qml.RZ(weights[i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]



weight_shapes = {"weights": (n_qubits,)}

QuantumLayer = KerasLayer(
    quantum_circuit,
    weight_shapes,
    output_dim=n_qubits
)
