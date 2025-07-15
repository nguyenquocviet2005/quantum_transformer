#!/usr/bin/env python
# coding: utf-8

# ## Experiment
# import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu"
import os
import json
import jax
import jax.numpy as jnp
import optax
import pennylane as qml
# from sklearn.datasets import fetch_openml # No longer used directly
# from sklearn.model_selection import train_test_split # No longer used directly
import time
import pandas as pd
# from filelock import FileLock # No longer used directly
import numpy as np
# from jax.experimental import host_callback # No longer used directly
# import tensorflow as tf  # For loading CIFAR-10 # Will be removed
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-GUI)
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score # Added

from jax import config
config.update("jax_enable_x64", True)

# --- Model Configuration Constants (Adapted for CIFAR-10) ---
D_PATCH_VALUE = 96  # From qvt_cifar_nodistill.py
NUM_LAYERS = 2      # From qvt_cifar_nodistill.py
PATCH_SIZE_CIFAR = 4 # CIFAR images are 32x32, 4x4 patches -> 8x8 = 64 patches
S_VALUE_CIFAR = (32 // PATCH_SIZE_CIFAR)**2 # Sequence length (number of patches)
INPUT_PATCH_DIM_CIFAR = (PATCH_SIZE_CIFAR**2) * 3 # For CIFAR-10 (3 channels)
N_QUBITS = 5            # Number of qubits for QSAL (example value, can be tuned)
DENC_QSAL = 2           # Depth of encoding ansatz for QSAL (example value, can be tuned)
D_QSAL = 1              # Depth of Q, K, V ansatzes for QSAL (example value, can be tuned)
NUM_CLASSES = 10        # CIFAR-10 has 10 classes
# --- End Model Configuration ---

# Check JAX backend (e.g., CPU or GPU)
print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Helper Functions for Transformer Components ---
@jax.jit
def layer_norm(x, gamma, beta, eps=1e-5):
    """Applies Layer Normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

@jax.jit
def feed_forward(x, w1, b1, w2, b2):
    """Position-wise Feed-Forward Network with ReLU."""
    x = jnp.dot(x, w1) + b1
    x = jax.nn.relu(x)
    x = jnp.dot(x, w2) + b2
    return x
# --- End Helper Functions ---

# QViT Model Classes (Adapted for JAX)
class QSAL_pennylane:
    def __init__(self, S, n, D, d_patch_config):
        self.seq_num = S  # Number of sequence positions (16 for 4x4 patches)
        self.num_q = n    # Number of qubits
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = d_patch_config # Dimension of input/output vectors (now configurable)
        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit using the new diversified method
        # self.observables = QSAL_pennylane._generate_observables(self.num_q, self.d)
        # # Define observables for value circuit, now based on d_patch_config
        self.observables = []
        for i in range(self.d):
            qubit = i % self.num_q
            pauli_idx = (i // self.num_q) % 3
            if pauli_idx == 0:
                obs = qml.PauliZ(qubit)
            elif pauli_idx == 1:
                obs = qml.PauliX(qubit)
            else:
                obs = qml.PauliY(qubit)
            self.observables.append(obs)


        # Define quantum nodes with JAX interface
        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="jax")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="jax")

    def circuit_v(self, inputs, weights):
        """Value circuit returning a d-dimensional vector of observable expectations."""
        # Amplitude encoding with padding or truncation to match 2^num_qubits length
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        # Normalize input vector, handling zero norm case
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / norm, jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
    
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j)
            qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[idx], wires=j)
                idx += 1
        return [qml.expval(obs) for obs in self.observables]

    def circuit_qk(self, inputs, weights):
        """Query/Key circuit returning Pauli-Z expectation on qubit 0."""
        # Amplitude encoding with padding or truncation to match 2^num_qubits length
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
        # Normalize input vector, handling zero norm case
        norm = jnp.linalg.norm(inputs)
        normalized_inputs = jnp.where(norm > 0, inputs / (norm + 1e-9), jnp.ones_like(inputs) / jnp.sqrt(2 ** self.num_q))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=True)
        
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j)
            qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for i in range(self.D):
            for j in range(self.num_q):
                qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q):
                qml.RY(weights[idx], wires=j)
                idx += 1
        return [qml.expval(qml.PauliZ(0))]

    # @staticmethod
    # def _generate_observables(num_qubits, num_observables_needed):
    #     candidate_observables = []
    #     paulis = [qml.PauliX, qml.PauliY, qml.PauliZ]

    #     # 1. Single-qubit observables
    #     if len(candidate_observables) < num_observables_needed:
    #         for i in range(num_qubits):
    #             for P_constructor in paulis:
    #                 if len(candidate_observables) < num_observables_needed:
    #                     candidate_observables.append(P_constructor(wires=i))
    #                 else:
    #                     break
    #             if len(candidate_observables) >= num_observables_needed:
    #                 break
        
    #     # 2. Two-qubit observables
    #     if len(candidate_observables) < num_observables_needed:
    #         for i in range(num_qubits):
    #             for j in range(i + 1, num_qubits): # Ensure i < j for unique pairs of qubits
    #                 for P1_constructor in paulis:
    #                     for P2_constructor in paulis:
    #                         if len(candidate_observables) < num_observables_needed:
    #                             candidate_observables.append(P1_constructor(wires=i) @ P2_constructor(wires=j))
    #                         else:
    #                             break
    #                     if len(candidate_observables) >= num_observables_needed:
    #                         break
    #                 if len(candidate_observables) >= num_observables_needed:
    #                     break
    #             if len(candidate_observables) >= num_observables_needed:
    #                 break
        
    #     # 3. Three-qubit observables
    #     if len(candidate_observables) < num_observables_needed:
    #         for i in range(num_qubits):
    #             for j in range(i + 1, num_qubits):
    #                 for k in range(j + 1, num_qubits): # Ensure i < j < k
    #                     for P1_constructor in paulis:
    #                         for P2_constructor in paulis:
    #                             for P3_constructor in paulis:
    #                                 if len(candidate_observables) < num_observables_needed:
    #                                     candidate_observables.append(
    #                                         P1_constructor(wires=i) @ P2_constructor(wires=j) @ P3_constructor(wires=k)
    #                                     )
    #                                 else:
    #                                     break
    #                             if len(candidate_observables) >= num_observables_needed:
    #                                 break
    #                         if len(candidate_observables) >= num_observables_needed:
    #                             break
    #                     if len(candidate_observables) >= num_observables_needed:
    #                         break
    #                 if len(candidate_observables) >= num_observables_needed:
    #                     break
    #             if len(candidate_observables) >= num_observables_needed:
    #                 break

    #     if len(candidate_observables) >= num_observables_needed:
    #         return candidate_observables[:num_observables_needed]
    #     else:
    #         # Fallback: Not enough unique observables generated with up to 3-qubit Paulis
    #         print(f"Warning: Generated only {len(candidate_observables)} unique Pauli string observables (up to 3 qubits) "
    #               f"for {num_qubits} qubits, but needed {num_observables_needed}. "
    #               "Repeating from the generated set to fill the remainder.")
            
    #         final_observables = list(candidate_observables) # Start with what we have
    #         if not final_observables: # e.g. num_qubits = 0
    #             if num_observables_needed > 0:
    #                 raise ValueError(f"Cannot generate {num_observables_needed} observables with {num_qubits} qubits.")
    #             else:
    #                 return [] # Return empty list if 0 observables needed and 0 generated

    #         # Cycle through the generated unique observables to fill the remainder
    #         # This ensures the list has 'num_observables_needed' items.
    #         idx = 0
    #         while len(final_observables) < num_observables_needed:
    #             final_observables.append(candidate_observables[idx % len(candidate_observables)])
    #             idx += 1
    #         return final_observables

    def __call__(self, input_sequence, layer_params):
        # layer_params contains: Q, K, V circuit weights, ln1_gamma, ln1_beta, ffn_w1, ffn_b1, ffn_w2, ffn_b2, ln2_gamma, ln2_beta
        batch_size = input_sequence.shape[0]
        S = self.seq_num
        d = self.d # This is d_patch, should be 48

        # 1. Layer Norm before QSA
        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        
        # Reshape input for QSA (as before)
        # Assuming x_norm1 is (batch_size, S, d)
        input_flat = jnp.reshape(x_norm1, (-1, d))  # Flatten batch and sequence dimensions together

        # Compute Q, K, V using vectorized operations (Quantum Self-Attention part)
        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        # Reshape back to include sequence dimension
        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, d)

        # Compute Gaussian self-attention coefficients
        Q_expanded = Q_output[:, :, None, :]
        K_expanded = K_output[:, None, :, :]
        alpha = jnp.exp(-(Q_expanded - K_expanded) ** 2)
        Sum_a = jnp.sum(alpha, axis=2, keepdims=True)
        alpha_normalized = alpha / (Sum_a + 1e-9) # Add epsilon for stability

        # Compute weighted sum of values
        V_output_expanded = V_output[:, None, :, :] # V_output is (batch, S, d), need (batch, 1, S, d) for broadcasting with alpha_normalized (batch, S, S, 1)
                                                # So V_output_expanded should be V_output[:, None, :, :] -> (batch_size, 1, S, d_patch)
                                                # alpha_normalized is (batch_size, S, S, 1)
        weighted_V = alpha_normalized * V_output_expanded # (B,S,S,1) * (B,1,S,d) -> (B,S,S,d) via broadcasting
        qsa_out = jnp.sum(weighted_V, axis=2) # Sum over the K dimension (axis=2) -> (B,S,d)

        # 2. First Residual Connection
        x_after_qsa_res = input_sequence + qsa_out

        # 3. Layer Norm before FFN
        x_norm2 = layer_norm(x_after_qsa_res, layer_params['ln2_gamma'], layer_params['ln2_beta'])

        # 4. Feed-Forward Network
        ffn_out = feed_forward(x_norm2, 
                               layer_params['ffn_w1'], layer_params['ffn_b1'], 
                               layer_params['ffn_w2'], layer_params['ffn_b2'])

        # 5. Second Residual Connection
        output = x_after_qsa_res + ffn_out
        return output

class QSANN_pennylane:
    def __init__(self, S, n, D, num_layers, d_patch_config):
        self.qsal_lst = [QSAL_pennylane(S, n, D, d_patch_config) for _ in range(num_layers)]

    def __call__(self, input_sequence, qnn_params_dict):
        # qnn_params_dict contains 'pos_encoding' and 'layers' (list of layer_param dicts)
        x = input_sequence + qnn_params_dict['pos_encoding'] # Add positional encoding
        
        for i, qsal_layer in enumerate(self.qsal_lst):
            layer_specific_params = qnn_params_dict['layers'][i]
            x = qsal_layer(x, layer_specific_params)
        return x

class QSANN_image_classifier:
    def __init__(self, S, n, D, num_layers, d_patch_config):
        self.Qnn = QSANN_pennylane(S, n, D, num_layers, d_patch_config)
        self.d_patch = d_patch_config # Store the configured patch dimension
        self.S = S
        self.num_layers = num_layers
        # final_ln_gamma and final_ln_beta are initialized directly in params, not as class members here

    def __call__(self, x, params):
        # x is initially (batch_size, S, input_patch_dim) e.g. (B, 64, 48)
        
        # 1. Patch Embedding Projection
        # Project from input_patch_dim (48) to d_patch_config (e.g., 96)
        # x needs to be (batch_size * S, input_patch_dim) for matmul
        batch_size, S, input_patch_dim_actual = x.shape
        x_flat = x.reshape(batch_size * S, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat, params['patch_embed_w']) + params['patch_embed_b']
        x_projected = projected_x_flat.reshape(batch_size, S, self.d_patch) # self.d_patch is d_patch_config

        # 2. QNN (Transformer blocks)
        qnn_params_dict = params['qnn'] # This now contains 'pos_encoding' and 'layers'
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)
        
        # 3. Final Layer Norm before classification head
        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        
        # 4. Flatten
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1) # (batch_size, S * d_patch)
        # x_pooled = jnp.mean(x_final_norm, axis=1) # Shape: (batch_size, d_patch)

        # 5. Final layer (Classifier Head)
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x_flat_for_head, w) + b
        return logits # Return raw logits for cross-entropy

# Loss and Metrics
@jax.jit
def softmax_cross_entropy_with_integer_labels(logits, labels):
    """Computes softmax cross entropy between logits and labels (integers)."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

@jax.jit
def accuracy_multiclass(logits, labels):
    """Computes accuracy for multi-class classification."""
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze()) #.squeeze() to match shape if labels are (N,1)

# Evaluation Function
def evaluate(model, params, dataloader):
    """Evaluate the model on the given dataloader (multi-class)."""
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    cifar_mean_np = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    cifar_std_np = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    all_logits_list = []
    all_labels_list = []

    # JIT-compiled function for batch evaluation
    @jax.jit
    def evaluate_batch(params, x_batch_jax, y_batch_jax):
        logits = model(x_batch_jax, params)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch_jax))
        acc = accuracy_multiclass(logits, y_batch_jax)
        return loss, acc, logits

    for x_batch_torch, y_batch_torch in dataloader:
        # Convert PyTorch tensors to NumPy arrays
        x_batch_np = x_batch_torch.numpy()
        y_batch_np = y_batch_torch.numpy().reshape(-1, 1) # Ensure labels are (N, 1)
        current_batch_size = x_batch_np.shape[0]

        # Denormalize: (normalized_image * std) + mean
        x_batch_np = (x_batch_np * cifar_std_np) + cifar_mean_np
        x_batch_np = np.clip(x_batch_np, 0., 1.) # Ensure values are in [0,1]
        x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)

        # Create patches
        x_batch_patches_np = create_patches(x_batch_np)

        # Convert to JAX arrays
        x_batch_jax = jnp.array(x_batch_patches_np)
        y_batch_jax = jnp.array(y_batch_np) # Labels are integers 0-9

        # Use JIT-compiled evaluation function
        loss, acc, logits = evaluate_batch(params, x_batch_jax, y_batch_jax)

        total_loss += loss * current_batch_size
        total_acc += acc * current_batch_size
        num_samples += current_batch_size

        all_logits_list.append(logits)
        all_labels_list.append(y_batch_jax)

    avg_loss = total_loss / num_samples
    avg_acc = total_acc / num_samples
    
    all_logits_jnp = jnp.concatenate(all_logits_list, axis=0)
    all_labels_jnp = jnp.concatenate(all_labels_list, axis=0)
    
    return avg_loss, avg_acc, all_logits_jnp, all_labels_jnp

def save_model_params(params, model_name, project_path="."):
    """
    Save trained model parameters to disk.
    
    Args:
        params: JAX parameters to save
        model_name: Name for the saved files
        project_path: Directory to save files in
    """
    import pickle
    params_save_path = os.path.join(project_path, f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Trained parameters saved to: {params_save_path}")
    return params_save_path

def load_model_params(model_name, project_path="."):
    """
    Load trained model parameters from disk.
    
    Args:
        model_name: Name of the saved model
        project_path: Directory containing the saved files
        
    Returns:
        params: Loaded JAX parameters
    """
    import pickle
    params_save_path = os.path.join(project_path, f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'rb') as f:
        params = pickle.load(f)
    print(f"Trained parameters loaded from: {params_save_path}")
    return params

def save_model_config(config, model_name, project_path="."):
    """
    Save model configuration to disk.
    
    Args:
        config: Dictionary containing model configuration
        model_name: Name for the saved files
        project_path: Directory to save files in
    """
    config_save_path = os.path.join(project_path, f"{model_name}_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to: {config_save_path}")
    return config_save_path

def load_model_config(model_name, project_path="."):
    """
    Load model configuration from disk.
    
    Args:
        model_name: Name of the saved model
        project_path: Directory containing the saved files
        
    Returns:
        config: Loaded model configuration dictionary
    """
    config_save_path = os.path.join(project_path, f"{model_name}_config.json")
    with open(config_save_path, 'r') as f:
        config = json.load(f)
    print(f"Model configuration loaded from: {config_save_path}")
    return config

def create_patches(images, patch_size=4):
    """Convert CIFAR images into patches.
    
    Args:
        images: Array of shape (batch_size, 32, 32, 3)
        patch_size: Size of each square patch
    
    Returns:
        patches: Array of shape (batch_size, num_patches, patch_size*patch_size*3)
    """
    batch_size = images.shape[0]
    img_size = 32
    num_patches_per_dim = img_size // patch_size
    
    # Convert to JAX array for better performance
    images_jax = jnp.array(images)
    
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            # Extract patch (including all color channels)
            patch = images_jax[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :]
            # Flatten patch (patch_size*patch_size*3 dimensions)
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)
    
    # Stack patches
    patches = jnp.stack(patches, axis=1)  # Shape: (batch_size, num_patches, patch_size*patch_size*3)
    return patches

def plot_confusion_matrix(cm, class_names, epoch, n_train, rep_num, d_patch, num_layers, batch_s):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'CM - Epoch {epoch} (N_train={n_train}, Rep={rep_num}, D_patch={d_patch}, L={num_layers}, Batch={batch_s})', fontsize=16)
    filename = f'confusion_matrix_cifar_E{epoch}_N{n_train}_Rep{rep_num}_D{d_patch}_L{num_layers}_B{batch_s}.png'
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def load_cifar_data(n_train, n_test, batch_size, augment=True):
    """Load and preprocess CIFAR-10 dataset (10 classes) with optional data augmentation.
    Returns PyTorch DataLoaders for training and testing.
    """
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2470, 0.2435, 0.2616) # More common std for CIFAR10

    transform_train_list = [transforms.ToTensor()]
    if augment:
        transform_train_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ])
    transform_train_list.append(transforms.Normalize(cifar_mean, cifar_std))
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar_mean, cifar_std)
    ])

    # Load CIFAR-10 (10 classes)
    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                 download=True, transform=transform_train)
    testset_full = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)

    # Subsample if n_train or n_test is smaller than the full dataset
    if n_train < len(trainset_full):
        train_indices = np.random.choice(len(trainset_full), n_train, replace=False)
        trainset = torch.utils.data.Subset(trainset_full, train_indices)
    else:
        trainset = trainset_full
    
    if n_test < len(testset_full):
        test_indices = np.random.choice(len(testset_full), n_test, replace=False)
        testset = torch.utils.data.Subset(testset_full, test_indices)
    else:
        testset = testset_full
        
    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True) # num_workers=0 for JAX compatibility
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, # Can use a larger batch for testing if memory allows
                                             shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# --- Parameter Initialization (Adapted for CIFAR-10) ---
def init_params(S, n, D, num_layers, d_patch_config, input_patch_dim, num_classes_out, key_seed=42):
    key = jax.random.PRNGKey(key_seed)
    d_ffn = d_patch_config * 4 

    num_random_keys = 2 + 1 + num_layers * 5 + 1 
    keys = jax.random.split(key, num_random_keys)
    key_idx = 0

    # Patch embedding projection from input_patch_dim_actual (e.g. 4*4*3=48) to d_patch_config (e.g. 96)
    patch_embed_w = jax.random.normal(keys[key_idx], (input_patch_dim, d_patch_config), dtype=jnp.float64) * jnp.sqrt(1.0 / input_patch_dim); key_idx+=1
    patch_embed_b = jax.random.normal(keys[key_idx], (d_patch_config,), dtype=jnp.float64) * 0.01; key_idx+=1 # Small random bias
    
    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch_config), dtype=jnp.float64) * 0.02; key_idx+=1

    qnn_layers_params = []
    for _ in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float64) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float64) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float64) - 1),
            'ln1_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 'ln1_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64),
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch_config, d_ffn), dtype=jnp.float64) * jnp.sqrt(1.0 / d_patch_config), # Kaiming for linear before relu often sqrt(2/fan_in)
            'ffn_b1': jnp.zeros((d_ffn,), dtype=jnp.float64),
            'ffn_w2': jax.random.normal(keys[key_idx+4], (d_ffn, d_patch_config), dtype=jnp.float64) * jnp.sqrt(1.0 / d_ffn),
            'ffn_b2': jnp.zeros((d_patch_config,), dtype=jnp.float64),
            'ln2_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 'ln2_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64)
        }
        qnn_layers_params.append(layer_params)
        key_idx += 5

    params = {
        'patch_embed_w': patch_embed_w, 'patch_embed_b': patch_embed_b,
        'qnn': {'pos_encoding': pos_encoding_params, 'layers': qnn_layers_params},
        'final_ln_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 
        'final_ln_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64),
        'final': {
            'weight': jax.random.normal(keys[key_idx], (d_patch_config * S, num_classes_out), dtype=jnp.float64) * 0.01, # 10 classes for CIFAR-10
            'bias': jnp.zeros((num_classes_out,), dtype=jnp.float64) # 10 classes for CIFAR-10
        }
    }
    return params
# --- End Parameter Initialization ---

def count_parameters(params):
    """Counts the total number of parameters in a PyTree."""
    count = 0
    for leaf in jax.tree_util.tree_leaves(params):
        count += leaf.size
    return count

# Training Function
def train_qvit(n_train, n_test, n_epochs, batch_size=64, rep_num=0): # Added rep_num
    # Load data
    train_loader, test_loader = load_cifar_data(n_train, n_test, batch_size, augment=False)  # Disable augmentation
    cifar_mean_np = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    cifar_std_np = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 3, 1, 1)

    # Initialize model and parameters
    model = QSANN_image_classifier(S=S_VALUE_CIFAR, n=N_QUBITS, D=D_QSAL, num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE)
    params = init_params(
        S=S_VALUE_CIFAR, n=N_QUBITS, D=D_QSAL, num_layers=NUM_LAYERS, 
        d_patch_config=D_PATCH_VALUE, input_patch_dim=INPUT_PATCH_DIM_CIFAR, num_classes_out=NUM_CLASSES
    )    
    # Count and print total parameters
    total_params = count_parameters(params)
    print(f"\nTotal number of parameters in the QViT model: {total_params:,}")

    # Define optimizer with cosine annealing learning rate schedule
    initial_lr = 0.001
    num_batches_per_epoch = len(train_loader)
    total_train_steps = n_epochs * num_batches_per_epoch
    warmup_steps = int(total_train_steps * 0.05)  # 5% of total steps for warmup

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, # Start LR for warmup
        peak_value=initial_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_train_steps - warmup_steps, # Steps for cosine decay after warmup
        end_value=initial_lr * 0.01 # Optionally, end at a small fraction of LR
    )

    # Chain Adam with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),  # Clip gradients to a max global norm of 1.0
        optax.adamw(learning_rate=lr_schedule) # You reverted to adam, keeping it here
    )

    opt_state = optimizer.init(params)

    # Create arrays to store metrics
    train_costs = []
    test_costs = []
    train_accs = []
    test_accs = []
    steps = []
    # New lists for additional metrics
    test_precision_macros = []
    test_recall_macros = []
    test_f1_macros = []
    test_top1_errors = []
    test_top5_errors = []

    # Loss function (batch context)
    def loss_fn_batch(p, x_batch, y_batch):
        logits = model(x_batch, p)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch))
        return loss, logits

    # JIT-compiled update step for a single batch
    @jax.jit
    def update_batch(params, opt_state, x_batch_jax, y_batch_jax):
        (loss_val, logits), grads = jax.value_and_grad(loss_fn_batch, has_aux=True)(params, x_batch_jax, y_batch_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params) # Pass params for AdamW style updates if needed
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    # Training loop
    current_params = params
    current_opt_state = opt_state
    start_time = time.time()
    
    for epoch in range(n_epochs):
        start_time_epoch = time.time()
        
        # Track batch losses only for monitoring (not for final metrics)
        epoch_batch_losses = []
        num_train_batches = 0

        for x_batch_torch, y_batch_torch in train_loader:
            # Convert PyTorch tensors to NumPy arrays
            x_batch_np = x_batch_torch.numpy()
            y_batch_np = y_batch_torch.numpy().reshape(-1, 1) # Ensure labels are (N, 1)

            # Denormalize, transpose, and patch
            x_batch_np = (x_batch_np * cifar_std_np) + cifar_mean_np
            x_batch_np = np.clip(x_batch_np, 0., 1.)
            x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
            x_batch_patches_np = create_patches(x_batch_np)
            
            # Convert to JAX arrays
            x_batch_jax = jnp.array(x_batch_patches_np)
            y_batch_jax = jnp.array(y_batch_np)
            current_params, current_opt_state, batch_loss = update_batch(
                current_params, current_opt_state, x_batch_jax, y_batch_jax
            )
            epoch_batch_losses.append(float(batch_loss))
            num_train_batches += 1
        
        # Evaluate on entire training set using final epoch parameters
        train_loss, train_acc, train_all_logits, train_all_labels = evaluate(model, current_params, train_loader)
        
        # Evaluate on test set using final epoch parameters
        test_loss, test_acc, test_all_logits, test_all_labels = evaluate(model, current_params, test_loader)
        
        # Calculate additional metrics for the test set
        test_y_true_np = np.array(test_all_labels.squeeze())
        test_y_pred_proba_np = np.array(jax.nn.softmax(test_all_logits, axis=-1))
        test_y_pred_np = np.array(jnp.argmax(test_all_logits, axis=-1))
        
        num_classes = test_all_logits.shape[-1]
        # Define CIFAR-10 class names for report and confusion matrix
        cifar10_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        if num_classes == 10:
            class_labels_to_use = cifar10_class_names
        else:
            # Fallback for safety, though CIFAR-10 should always have 10 classes
            print(f"Warning: Expected 10 CIFAR-10 classes, got {num_classes}. Using numeric labels.")
            class_labels_to_use = [str(i) for i in range(num_classes)]
        
        # Classification report (precision, recall, F1)
        # Ensure labels parameter is correctly used if not all classes are present, though for CIFAR10 it's usually fine.
        report_dict = classification_report(test_y_true_np, test_y_pred_np, output_dict=True, zero_division=0, labels=np.arange(num_classes), target_names=class_labels_to_use)
        precision_macro = report_dict['macro avg']['precision']
        recall_macro = report_dict['macro avg']['recall']
        f1_macro = report_dict['macro avg']['f1-score']
        
        current_top1_error = 1.0 - test_acc # test_acc is top-1 accuracy
        # For top_k_accuracy_score, ensure labels are 0 to num_classes-1
        current_top5_acc = top_k_accuracy_score(test_y_true_np, test_y_pred_proba_np, k=5, labels=np.arange(num_classes))
        current_top5_error = 1.0 - current_top5_acc

        # Store metrics
        train_costs.append(float(train_loss))
        train_accs.append(float(train_acc))
        test_costs.append(float(test_loss))
        test_accs.append(float(test_acc))
        # Store new metrics
        test_precision_macros.append(float(precision_macro))
        test_recall_macros.append(float(recall_macro))
        test_f1_macros.append(float(f1_macro))
        test_top1_errors.append(float(current_top1_error))
        test_top5_errors.append(float(current_top5_error))
        steps.append(epoch + 1)
        
        # Calculate average batch loss for monitoring
        avg_batch_loss = np.mean(epoch_batch_losses)
        
        # Print progress (reduced frequency)
        # if (epoch + 1) % 10 == 0: # Print every 10 epochs
        print(f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f} | "
                f"Time: {time.time() - start_time_epoch:.2f}s")

        if epoch == n_epochs - 1: # For the last epoch of this training run
            final_cm = confusion_matrix(test_y_true_np, test_y_pred_np, labels=np.arange(num_classes))
            # D_PATCH_VALUE and NUM_LAYERS are global in your script
            plot_confusion_matrix(final_cm, class_labels_to_use, epoch + 1, n_train, rep_num, D_PATCH_VALUE, NUM_LAYERS, batch_size)
    print("--- Saving Trained Parameters ---")

    MODEL_NAME=f"CIFAR_{n_train}_rep{rep_num}"
    save_model_params(current_params, MODEL_NAME)

    # Also save model configuration for future loading
    model_config = {
        'S': S_VALUE_CIFAR,
        'n_qubits': N_QUBITS,
        'D': D_QSAL,
        'num_layers': NUM_LAYERS,
        'd_patch': D_PATCH_VALUE,
        'input_patch_dim': INPUT_PATCH_DIM_CIFAR,
        'num_classes': NUM_CLASSES,
        'patch_size': PATCH_SIZE_CIFAR
    }
    save_model_config(model_config, MODEL_NAME)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds for n_train={n_train}, rep {rep_num}")

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'step': steps,
        'train_cost': train_costs,
        'train_acc': train_accs,
        'test_cost': test_costs,
        'test_acc': test_accs,
        'test_precision_macro': test_precision_macros,
        'test_recall_macro': test_recall_macros,
        'test_f1_macro': test_f1_macros,
        'test_top1_error': test_top1_errors,
        'test_top5_error': test_top5_errors,
        'n_train': [n_train] * len(steps),
        'batch_size': [batch_size] * len(steps), # Add batch_size to results
        'repetition': [rep_num] * len(steps) # Add repetition number
    })
    
    return results_df

# Constants
# n_test = 4000
n_epochs = 100
n_reps = 1 # Consider reducing for faster testing of 10-class setup
train_sizes = [1000,2000,5000,10000,20000,50000] # Consider smaller sizes first for 10-class
BATCH_SIZE = 64 # Define a global batch size or pass it around

def run_iterations(n_train, current_batch_size):
    """Run multiple training iterations for a given training size and print progress."""
    all_results = []
    for rep in range(n_reps):
        print(f"\nStarting repetition {rep + 1}/{n_reps} for train size {n_train}, batch size {current_batch_size}")
        results_df = train_qvit(n_train, n_train//5, n_epochs, batch_size=current_batch_size, rep_num=rep + 1) # Pass rep_num
        all_results.append(results_df)
    
    return pd.concat(all_results, ignore_index=True)

# Run experiments and collect results
all_results_collected = [] # Renamed to avoid conflict with all_results in run_iterations
for n_train_current in train_sizes:
    print(f"\n=== Starting training for train size {n_train_current}, batch size {BATCH_SIZE} ===")
    results = run_iterations(n_train_current, BATCH_SIZE)
    all_results_collected.append(results)

# Combine all results
results_df_combined = pd.concat(all_results_collected, ignore_index=True) # Renamed

# Aggregate results
df_agg = results_df_combined.groupby(["n_train", "batch_size", "step"]).agg({ # Added batch_size to groupby
    "train_cost": ["mean", "std"],
    "test_cost": ["mean", "std"],
    "train_acc": ["mean", "std"],
    "test_acc": ["mean", "std"],
    "test_precision_macro": ["mean", "std"], # Added
    "test_recall_macro": ["mean", "std"],   # Added
    "test_f1_macro": ["mean", "std"],       # Added
    "test_top1_error": ["mean", "std"],     # Added
    "test_top5_error": ["mean", "std"]      # Added
}).reset_index()

# Plotting
sns.set_style('whitegrid')
colors = sns.color_palette()
fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

generalization_errors = []

# Plot losses and accuracies
for i, n_train in enumerate(train_sizes):
    df = df_agg[df_agg.n_train == n_train]
    dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
    lines = ["o-", "x--", "o-", "x--"]
    labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
    axs = [0, 0, 2, 2]

    for k in range(4):
        ax = axes[axs[k]]
        ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

    # Compute generalization error
    dif = df[df.step == n_epochs].test_cost["mean"].values[0] - df[df.step == n_epochs].train_cost["mean"].values[0]
    generalization_errors.append(dif)

# Format plots
axes[0].set_title('Train and Test Losses (CIFAR-10)', fontsize=14)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')

axes[1].plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
axes[1].set_xscale('log')
axes[1].set_xticks(train_sizes)
axes[1].set_xticklabels(train_sizes)
axes[1].set_title(r'Generalization Error $gen(\alpha) = R(\alpha) - \hat{R}_N(\alpha)$', fontsize=14)
axes[1].set_xlabel('Training Set Size')
axes[1].set_yscale('log', base=2)

axes[2].set_title('Train and Test Accuracies (CIFAR-10)', fontsize=14)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Accuracy')
axes[2].set_ylim(0.1, 1.05)

legend_elements = (
    [mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)] +
    [
        mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
        mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
    ]
)

axes[0].legend(handles=legend_elements, ncol=3)
axes[2].legend(handles=legend_elements, ncol=3)

plt.tight_layout()
plt.savefig('qvit_cifar_learning_curves.png')
plt.close()

# Save results to CSV
results_df_combined.to_csv('qvit_cifar10_results.csv', index=False) # Updated filename
print("Results saved to qvit_cifar10_results.csv")
print("Plots saved to qvit_cifar_learning_curves.png") # Consider updating plot filename if it reflects 10-class