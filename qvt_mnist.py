#!/usr/bin/env python
# coding: utf-8

# ## Experiment
import os
import json
# os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import optax
import pennylane as qml
import time
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg (non-GUI)
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from functools import partial
# Create organized folder structure
def create_experiment_dirs(dataset_name="mnist"):
    """Create organized directory structure for experiments."""
    base_dir = f"experiments/{dataset_name}"
    dirs = [
        f"{base_dir}/models",
        f"{base_dir}/metrics", 
        f"{base_dir}/plots",
        f"{base_dir}/configs",
        f"{base_dir}/adversarial"
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    return base_dir


from jax import config
from jax.nn import one_hot
config.update("jax_enable_x64", True)

# --- Model Configuration Constants ---
D_PATCH_VALUE = 32  # From qvt_MNIST_nodistill.py
NUM_LAYERS = 1      # From qvt_MNIST_nodistill.py
PATCH_SIZE_MNIST = 4 # MNIST images are 32x32, 4x4 patches -> 8x8 = 64 patches
S_VALUE_MNIST = (28 // PATCH_SIZE_MNIST)**2 # Sequence length (number of patches)
INPUT_PATCH_DIM_MNIST = (PATCH_SIZE_MNIST**2) * 1 # For MNIST-10 (3 channels)
N_QUBITS = 5            # Number of qubits for QSAL (example value, can be tuned)
D_QSAL = 1              # Depth of Q, K, V ansatzes for QSAL (example value, can be tuned)
NUM_CLASSES = 10        # MNIST-10 has 10 classes
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

# --- Lipschitz Constant Estimation ---
@partial(jax.jit, static_argnames=('model_fn', 'norm_type'))
def lipschitz_bound_qvt(model_fn, params, x_batch, norm_type="l2"):
    """
    Compute empirical Lipschitz constant for QVT model based on gradient norms.
    
    This function computes the maximum gradient norm across a batch of inputs,
    which serves as an empirical estimate of the Lipschitz constant.
    
    Args:
        model_fn: The QVT model function
        params: Model parameters
        x_batch: Input batch for gradient computation (shape: [batch_size, seq_len, patch_dim])
        norm_type: "l2" or "inf" for gradient norm type
    
    Returns:
        Maximum gradient norm across the batch (empirical Lipschitz constant)
    """
    def single_grad_norm(x):
        # Add batch dimension if needed (x should be [seq_len, patch_dim])
        if x.ndim == 2:
            x_batched = x[None, :, :]  # Add batch dimension: [1, seq_len, patch_dim]
        else:
            x_batched = x
        
        # Compute gradient of model output w.r.t. input
        # Use the norm of the output logits to get a scalar value for gradient computation
        grad_fn = jax.grad(lambda inp: jnp.linalg.norm(model_fn(inp, params), ord=2))
        g = grad_fn(x_batched)
        
        # Remove batch dimension from gradient if it was added
        if x.ndim == 2:
            g = g[0]  # Remove batch dimension: [seq_len, patch_dim]
        
        # Compute norm of gradient
        if norm_type == "l2":
            return jnp.linalg.norm(g, ord=2)
        else:  # inf norm
            return jnp.linalg.norm(g, ord=jnp.inf)
    
    # Compute gradient norm for each sample in batch
    return jnp.max(jax.vmap(single_grad_norm)(x_batch))

# --- End Helper Functions ---

# QViT Model Classes (Adapted for JAX)
class QSAL_pennylane:
    def __init__(self, S, n, D, d_patch_config):
        self.seq_num = S  # Number of sequence positions
        self.num_q = n    # Number of qubits
        self.D = D        # Depth of Q, K, V ansatzes
        self.d = d_patch_config # Dimension of input/output vectors (embedding dimension)
        self.dev = qml.device("default.qubit", wires=self.num_q)

        # Define observables for value circuit, now based on d_patch_config
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
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
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
        expected_length = 2 ** self.num_q
        if len(inputs) > expected_length:
            inputs = inputs[:expected_length]
        elif len(inputs) < expected_length:
            inputs = jnp.pad(inputs, (0, expected_length - len(inputs)), mode='constant', constant_values=0)
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

    def __call__(self, input_sequence, layer_params):
        batch_size = input_sequence.shape[0]
        S = self.seq_num 
        d = self.d # This is d_patch_config (embedding dimension)

        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        input_flat = jnp.reshape(x_norm1, (-1, d))

        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, d)

        Q_expanded = Q_output[:, :, None, :]
        K_expanded = K_output[:, None, :, :]
        alpha = jnp.exp(-(Q_expanded - K_expanded) ** 2)
        Sum_a = jnp.sum(alpha, axis=2, keepdims=True)
        alpha_normalized = alpha / (Sum_a + 1e-9)

        V_output_expanded = V_output[:, None, :, :]
        weighted_V = alpha_normalized * V_output_expanded
        qsa_out = jnp.sum(weighted_V, axis=2)

        x_after_qsa_res = input_sequence + qsa_out
        x_norm2 = layer_norm(x_after_qsa_res, layer_params['ln2_gamma'], layer_params['ln2_beta'])
        ffn_out = feed_forward(x_norm2, 
                               layer_params['ffn_w1'], layer_params['ffn_b1'], 
                               layer_params['ffn_w2'], layer_params['ffn_b2'])
        output = x_after_qsa_res + ffn_out
        return output

class QSANN_pennylane:
    def __init__(self, S, n, D, num_layers, d_patch_config):
        self.qsal_lst = [QSAL_pennylane(S, n, D, d_patch_config) for _ in range(num_layers)]

    def __call__(self, input_sequence, qnn_params_dict):
        x = input_sequence + qnn_params_dict['pos_encoding']
        for i, qsal_layer in enumerate(self.qsal_lst):
            layer_specific_params = qnn_params_dict['layers'][i]
            x = qsal_layer(x, layer_specific_params)
        return x

class QSANN_image_classifier:
    def __init__(self, S, n, D, num_layers, d_patch_config):
        self.Qnn = QSANN_pennylane(S, n, D, num_layers, d_patch_config)
        self.d_patch = d_patch_config # Store the configured patch dimension (embedding dim)
        self.S = S # Number of patches
        self.num_layers = num_layers

    def __call__(self, x, params):
        # x is initially (batch_size, S, input_patch_dim_actual) e.g. (B, 49, 16) for MNIST
        batch_size, S_actual, input_patch_dim_actual = x.shape
        x_flat = x.reshape(batch_size * S_actual, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat, params['patch_embed_w']) + params['patch_embed_b']
        # x_projected is (batch_size, S, self.d_patch) where self.d_patch is d_patch_config (embedding dim)
        x_projected = projected_x_flat.reshape(batch_size, S_actual, self.d_patch)

        qnn_params_dict = params['qnn']
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)
        
        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1) # (batch_size, S * d_patch)
        
        w = params['final']['weight']
        b = params['final']['bias']
        logits = jnp.dot(x_flat_for_head, w) + b
        return logits

# Loss and Metrics
@jax.jit
def softmax_cross_entropy_with_integer_labels(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

@jax.jit
def accuracy_multiclass(logits, labels):
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze())

# Evaluation Function
def evaluate(model, params, dataloader):
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    mnist_mean_np = np.array([0.1307]).reshape(1, 1, 1, 1) # For (B,C,H,W)
    mnist_std_np = np.array([0.3081]).reshape(1, 1, 1, 1)  # For (B,C,H,W)

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
        x_batch_np = x_batch_torch.numpy()
        y_batch_np = y_batch_torch.numpy().reshape(-1, 1)
        current_batch_size = x_batch_np.shape[0]

        # Denormalize: (normalized_image * std) + mean
        x_batch_np = (x_batch_np * mnist_std_np) + mnist_mean_np # MNIST is 1 channel
        x_batch_np = np.clip(x_batch_np, 0., 1.)
        x_batch_np = x_batch_np.transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C) -> (B,28,28,1) for MNIST

        x_batch_patches_np = create_patches(x_batch_np) # MNIST patches

        x_batch_jax = jnp.array(x_batch_patches_np)
        y_batch_jax = jnp.array(y_batch_np)

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


def save_fidelity_metrics(fidelity_data, model_name, dataset_name="mnist"):
    """Save fidelity metrics to organized directory structure."""
    base_dir = create_experiment_dirs(dataset_name)
    fidelity_path = os.path.join(base_dir, "metrics", f"{model_name}_fidelity.json")
    with open(fidelity_path, 'w') as f:
        json.dump(fidelity_data, f, indent=4)
    print(f"Fidelity metrics saved to: {fidelity_path}")
    return fidelity_path

def save_model_params(params, model_name, dataset_name="mnist"):
    """Save trained model parameters to organized directory structure."""
    import pickle
    base_dir = create_experiment_dirs(dataset_name)
    params_save_path = os.path.join(base_dir, "models", f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Trained parameters saved to: {params_save_path}")
    return params_save_path

def load_model_params(model_name, dataset_name="mnist"):
    """Load trained model parameters from organized directory structure."""
    import pickle
    base_dir = f"experiments/{dataset_name}"
    params_save_path = os.path.join(base_dir, "models", f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'rb') as f:
        params = pickle.load(f)
    print(f"Trained parameters loaded from: {params_save_path}")
    return params

def save_model_config(config, model_name, dataset_name="mnist"):
    """
    Save model configuration to organized directory structure.
    
    Args:
        config: Dictionary containing model configuration
        model_name: Name for the saved files
        dataset_name: Dataset name for directory organization
    """
    base_dir = create_experiment_dirs(dataset_name)
    config_save_path = os.path.join(base_dir, "configs", f"{model_name}_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to: {config_save_path}")
    return config_save_path

def load_model_config(model_name, dataset_name="mnist"):
    """
    Load model configuration from organized directory structure.
    
    Args:
        model_name: Name of the saved model
        dataset_name: Dataset name for directory organization
        
    Returns:
        config: Loaded model configuration dictionary
    """
    base_dir = f"experiments/{dataset_name}"
    config_save_path = os.path.join(base_dir, "configs", f"{model_name}_config.json")
    with open(config_save_path, 'r') as f:
        config = json.load(f)
    print(f"Model configuration loaded from: {config_save_path}")
    return config

def create_patches(images, patch_size=4):
    """Convert MNIST images into patches.
    
    Args:
        images: Array of shape (batch_size, 28, 28, 1)
        patch_size: Size of each square patch (e.g., 4)
    
    Returns:
        patches: Array of shape (batch_size, num_patches, patch_size*patch_size*1)
                 e.g. (batch_size, 49, 16) for 28x28 images, 4x4 patches
    """
    batch_size = images.shape[0]
    img_size = 28 # MNIST image size
    num_channels = images.shape[-1] # Should be 1 for MNIST
    
    num_patches_per_dim = img_size // patch_size
    # num_patches = num_patches_per_dim * num_patches_per_dim # This is S_VALUE_MNIST (49)
    
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size, 
                            j*patch_size:(j+1)*patch_size, :]
            patch = patch.reshape(batch_size, -1) # Flatten to (batch_size, patch_size*patch_size*num_channels)
            patches.append(patch)
    
    patches = jnp.stack(patches, axis=1)
    return patches

def plot_confusion_matrix(cm, class_names, epoch, n_train, rep_num, d_patch, num_layers, batch_s, dataset_name="MNIST"):
    """Plots and saves the confusion matrix."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'CM ({dataset_name}) - Epoch {epoch} (N_train={n_train}, Rep={rep_num}, D_patch={d_patch}, L={num_layers}, Batch={batch_s})', fontsize=16)
    filename = f'confusion_matrix_{dataset_name.lower()}_E{epoch}_N{n_train}_Rep{rep_num}_D{d_patch}_L{num_layers}_B{batch_s}.png'
    plt.tight_layout()
    # Save to organized directory structure
    base_dir = create_experiment_dirs("mnist")
    plot_path = os.path.join(base_dir, "plots", filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix saved to {filename}")

def load_mnist_data(n_train, n_test, batch_size, augment=True):
    """Load and preprocess MNIST dataset with optional data augmentation.
    Returns PyTorch DataLoaders for training and testing.
    """
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    transform_train_list = []
    transform_train_list.append(transforms.ToTensor())
    if augment:
        transform_train_list.append(transforms.RandomCrop(28, padding=2)) # Adjusted padding for 28x28
        transform_train_list.append(transforms.RandomHorizontalFlip())
        # ColorJitter removed as MNIST is grayscale
    transform_train_list.append(transforms.Normalize(mnist_mean, mnist_std))
    transform_train = transforms.Compose(transform_train_list)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mnist_mean, mnist_std)
    ])

    trainset_full = torchvision.datasets.MNIST(root='./data', train=True,
                                               download=True, transform=transform_train)
    testset_full = torchvision.datasets.MNIST(root='./data', train=False,
                                              download=True, transform=transform_test)

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
        
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

# Parameter Initialization
def init_params(S, n, D, num_layers, d_patch_config, input_patch_dim):
    key = jax.random.PRNGKey(42)
    # S: number of patches (e.g., 49 for MNIST)
    # input_patch_dim: dimension of each patch (e.g., 16 for MNIST 4x4x1)
    # d_patch_config: target embedding dimension (e.g., D_PATCH_VALUE = 64)
    d_ffn = d_patch_config * 4

    num_random_keys = 2 + 1 + num_layers * 5 + 1
    keys = jax.random.split(key, num_random_keys)
    key_idx = 0

    patch_embed_w = jax.random.normal(keys[key_idx], (input_patch_dim, d_patch_config), dtype=jnp.float32) * jnp.sqrt(1.0 / input_patch_dim)
    key_idx += 1
    patch_embed_b = jax.random.normal(keys[key_idx], (d_patch_config,), dtype=jnp.float32) * 0.01
    key_idx += 1

    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch_config), dtype=jnp.float32) * 0.02
    key_idx += 1

    qnn_layers_params = []
    for i in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float32) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float32) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float32) - 1),
            'ln1_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
            'ln1_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32),
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch_config, d_ffn), dtype=jnp.float32) * jnp.sqrt(1.0 / d_patch_config),
            'ffn_b1': jnp.zeros((d_ffn,), dtype=jnp.float32),
            'ffn_w2': jax.random.normal(keys[key_idx+4], (d_ffn, d_patch_config), dtype=jnp.float32) * jnp.sqrt(1.0 / d_ffn),
            'ffn_b2': jnp.zeros((d_patch_config,), dtype=jnp.float32),
            'ln2_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
            'ln2_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32)
        }
        qnn_layers_params.append(layer_params)
        key_idx += 5

    params = {
        'patch_embed_w': patch_embed_w,
        'patch_embed_b': patch_embed_b,
        'qnn': {
            'pos_encoding': pos_encoding_params,
            'layers': qnn_layers_params
        },
        'final_ln_gamma': jnp.ones((d_patch_config,), dtype=jnp.float32),
        'final_ln_beta': jnp.zeros((d_patch_config,), dtype=jnp.float32),
        'final': { # Classifier head for 10 classes (MNIST)
            'weight': jax.random.normal(keys[key_idx], (d_patch_config * S, 10), dtype=jnp.float32) * 0.01,
            'bias': jnp.zeros((10,), dtype=jnp.float32)
        }
    }
    return params

def count_parameters(params):
    count = 0
    for leaf in jax.tree_util.tree_leaves(params):
        count += leaf.size
    return count

# === Adversarial Attack Evaluation Functions ===
import math
from functools import partial
from jax.nn import softmax, one_hot
from jax import lax

@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def evaluate_adv(model_fn, params, x, y, batch_size=64):
    n_samples = x.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    pad_amount = n_batches * batch_size - n_samples
    x_padded = jnp.pad(x, ((0, pad_amount), (0, 0), (0, 0)), 'constant')
    y_padded = jnp.pad(y, ((0, pad_amount), (0, 0)), 'constant')
    x_batched = x_padded.reshape(n_batches, batch_size, *x.shape[1:])
    y_batched = y_padded.reshape(n_batches, batch_size, *y.shape[1:])
    mask = jnp.arange(n_batches * batch_size) < n_samples
    mask_batched = mask.reshape(n_batches, batch_size)
    def eval_step(carry, batch):
        x_b, y_b, mask_b = batch
        logits = model_fn(x_b, params)
        preds = jnp.argmax(logits, axis=1)
        labels = jnp.argmax(y_b, axis=1)
        correct_in_batch = jnp.sum((preds == labels) * mask_b)
        carry += correct_in_batch
        return carry, None
    total_correct, _ = lax.scan(eval_step, 0, (x_batched, y_batched, mask_batched))
    return total_correct / n_samples

@partial(jax.jit, static_argnames=("model_fn", "batch_size"))
def evaluate_batched_for_adversarial(model_fn, params, x, batch_size=64):
    n_samples = x.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    pad_amount = n_batches * batch_size - n_samples
    x_padded = jnp.pad(x, ((0, pad_amount), (0, 0), (0, 0)), 'constant')
    x_batched = x_padded.reshape(n_batches, batch_size, *x.shape[1:])
    def map_fn(x_b):
        logits = model_fn(x_b, params)
        probs = softmax(logits, axis=-1)
        preds = jnp.argmax(logits, axis=1)
        return preds, probs
    all_preds_b, all_probs_b = lax.map(map_fn, x_batched)
    all_preds = all_preds_b.reshape(-1)[:n_samples]
    all_probs = all_probs_b.reshape(-1, all_probs_b.shape[-1])[:n_samples]
    return all_preds, all_probs

class Adversarial:
    def __init__(self, model_fn, params, x, y, batch_size=64):
        self.model_fn = model_fn
        self.params = params
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.acc_clean = float(evaluate_adv(model_fn, params, x, y, batch_size=batch_size))
        preds_clean, probs_clean = evaluate_batched_for_adversarial(model_fn, params, x, batch_size=batch_size)
        labels = jnp.argmax(y, axis=1)
        self.correct_clean = preds_clean == labels
        self.fidel_clean = probs_clean

    @staticmethod
    @partial(jax.jit, static_argnames='model_fn')
    def _fgsm_attack(x, y, params, model_fn, eps):
        loss_fn_scalar = lambda x_in, y_in: optax.softmax_cross_entropy(model_fn(x_in, params), y_in).mean()
        grad = jax.grad(loss_fn_scalar)(x, y)
        adv_x = x + eps * jnp.sign(grad)
        return jnp.clip(adv_x, 0.0, 1.0)

    @staticmethod
    @partial(jax.jit, static_argnames=("model_fn", "steps"))
    def _pgd_attack(x, y, params, model_fn, eps, steps):
        alpha = eps / steps
        loss_fn_scalar = lambda x_in, y_in: optax.softmax_cross_entropy(model_fn(x_in, params), y_in).mean()
        def pgd_step(x_adv, _):
            grads = jax.grad(loss_fn_scalar)(x_adv, y)
            x_adv = x_adv + alpha * jnp.sign(grads)
            x_adv = jnp.clip(x_adv, x - eps, x + eps)
            x_adv = jnp.clip(x_adv, 0.0, 1.0)
            return x_adv, None
        x_adv, _ = lax.scan(pgd_step, x, None, length=steps)
        return x_adv

    @staticmethod
    @partial(jax.jit, static_argnames=("model_fn", "steps"))
    def _apgd_attack(x, y, params, model_fn, eps, steps, decay):
        alpha = eps / steps
        loss_fn_one_hot = lambda x_in, y_in: optax.softmax_cross_entropy(model_fn(x_in, params), y_in)
        loss_fn_scalar = lambda x_in, y_in: loss_fn_one_hot(x_in, y_in).mean()
        def attack_step(misc, _):
            x_adv, best_adv, best_loss, step_size, best_loss_mean = misc
            grads = jax.grad(loss_fn_scalar)(x_adv, y)
            x_adv = x_adv + step_size * jnp.sign(grads)
            x_adv = jnp.clip(x_adv, x - eps, x + eps)
            x_adv = jnp.clip(x_adv, 0.0, 1.0)
            cur_loss = loss_fn_one_hot(x_adv, y)
            cur_loss_mean = cur_loss.mean()
            step_size = jnp.where(cur_loss_mean < best_loss_mean, step_size * decay, step_size)
            update_mask = cur_loss > best_loss
            update_mask = update_mask[:, None, None]
            best_adv = jnp.where(update_mask, x_adv, best_adv)
            best_loss = jnp.maximum(best_loss, cur_loss)
            best_loss_mean = best_loss.mean()
            return (x_adv, best_adv, best_loss, step_size, best_loss_mean), None
        x_adv = jnp.array(x)
        step_size = alpha
        cur_loss = loss_fn_one_hot(x_adv, y)
        best_loss = cur_loss
        best_loss_mean = cur_loss.mean()
        best_adv = x_adv
        init_misc = (x_adv, best_adv, best_loss, step_size, best_loss_mean)
        (_, best_adv, _, _, _), _ = lax.scan(attack_step, init_misc, None, length=steps)
        return best_adv

    @staticmethod
    @partial(jax.jit, static_argnames=("model_fn", "steps"))
    def _mim_attack(x, y, params, model_fn, eps, steps, decay):
        alpha = eps / steps
        loss_fn_scalar = lambda x_in, y_in: optax.softmax_cross_entropy(model_fn(x_in, params), y_in).mean()
        def attack_step(misc, _):
            x_adv, momentum = misc
            grad = jax.grad(loss_fn_scalar)(x_adv, y)
            grad = grad / (jnp.mean(jnp.abs(grad)) + 1e-8)
            momentum = decay * momentum + grad
            x_adv = x_adv + alpha * jnp.sign(momentum)
            x_adv = jnp.clip(x_adv, x - eps, x + eps)
            x_adv = jnp.clip(x_adv, 0.0, 1.0)
            return (x_adv, momentum), None
        x_adv = jnp.array(x)
        momentum = jnp.zeros_like(x)
        (final_adv, _), _ = lax.scan(attack_step, (x_adv, momentum), None, length=steps)
        return final_adv

    @staticmethod
    @partial(jax.jit, static_argnames='model_fn')
    def _jitted_predict(model_fn, params, x):
        return model_fn(x, params)

    def run(self, tpe="FGSM", metric=False, fidelity_only=False, **kwargs):
        attack_fn_map = {
            "FGSM": self._fgsm_attack,
            "PGD": self._pgd_attack,
            "MIM": self._mim_attack,
            "APGD": self._apgd_attack
        }
        attack_fn = attack_fn_map.get(tpe)
        if attack_fn is None:
            raise ValueError(f"Unsupported attack type: {tpe}")
        eps_list = [kwargs.get("eps", 8/255)]
        if metric:
            eps_list = [x * 1/255 for x in range(1, 11)]
        acc_adv_list, asr_list, robustness_gap_list, fidel_val_list = [], [], [], []
        for eps in eps_list:
            total_correct_adv, total_successful_attack, fidel_val_total = 0, 0, 0
            num_batches = (len(self.x) + self.batch_size - 1) // self.batch_size
            for i in range(num_batches):
                start, end = i * self.batch_size, (i + 1) * self.batch_size
                x_batch, y_batch = self.x[start:end], self.y[start:end]
                correct_clean_batch = self.correct_clean[start:end]
                fidel_clean_batch = self.fidel_clean[start:end]
                labels = jnp.argmax(y_batch, axis=1)
                if tpe == "FGSM":
                    x_adv_batch = attack_fn(x_batch, y_batch, self.params, self.model_fn, eps)
                elif tpe == "PGD":
                    steps = kwargs.get("steps", 100)
                    x_adv_batch = attack_fn(x_batch, y_batch, self.params, self.model_fn, eps, steps)
                elif tpe == "MIM":
                    steps = kwargs.get("steps", 100)
                    decay = kwargs.get("decay", 1.0)
                    x_adv_batch = attack_fn(x_batch, y_batch, self.params, self.model_fn, eps, steps, decay)
                elif tpe == "APGD":
                    steps = kwargs.get("steps", 100)
                    decay = kwargs.get("decay", 0.75)
                    x_adv_batch = attack_fn(x_batch, y_batch, self.params, self.model_fn, eps, steps, decay)
                logits_adv = self._jitted_predict(self.model_fn, self.params, x_adv_batch)
                preds_adv = jnp.argmax(logits_adv, axis=1)
                fidel_adv_batch = softmax(logits_adv, axis=-1)

                if not fidelity_only:
                    correct_adv = (preds_adv == labels).sum()
                    successful_attack = (correct_clean_batch & (preds_adv != labels)).sum()
                    total_correct_adv += correct_adv
                    total_successful_attack += successful_attack
                fidel_val_batch = jnp.sum(jnp.sqrt(fidel_clean_batch * fidel_adv_batch), axis=-1).sum()
                fidel_val_total += fidel_val_batch
            total_samples = len(self.x)
            if not fidelity_only:
                num_correct_clean = jnp.sum(self.correct_clean)
                acc_adv = float(total_correct_adv / total_samples)
                asr = float(total_successful_attack / num_correct_clean) if num_correct_clean > 0 else 0.0
                robustness_gap = float(self.acc_clean - acc_adv)
                fidel_val = float(fidel_val_total / total_samples)
                acc_adv_list.append(acc_adv)
                asr_list.append(asr)
                robustness_gap_list.append(robustness_gap)
                fidel_val_list.append(fidel_val)
            else:
                fidel_val = float(fidel_val_total / total_samples)
                fidel_val_list.append(fidel_val)
        if metric:
            data = {
                "acc_adv": acc_adv_list, "asr": asr_list,
                "robustness_gap": robustness_gap_list, "fidel_val": fidel_val_list
            }
            file_name = f"QVT_MNIST_{tpe}_on_perturbation_magnitude.json"
            save_path = os.path.join(".", file_name)
            with open(save_path, "w") as f:
                json.dump(data, f, indent=4)
        else:
            if fidelity_only:
                return fidel_val_list[0]
            else:
                return acc_adv_list[0], asr_list[0], robustness_gap_list[0], fidel_val_list[0]

# Training Function
def train_qvit(n_train, n_test, n_epochs, batch_size=64, rep_num=0, lipschitz_frequency=1, measure_lipschitz=True, 
               adversarial_frequency=1, measure_adversarial=True):
    train_loader, test_loader = load_mnist_data(n_train, n_test, batch_size, augment=False)  # Disable augmentation
    # MNIST specific mean and std for denormalization during training batch processing
    mnist_mean_np_train = np.array([0.1307]).reshape(1, 1, 1, 1) # For (B,C,H,W)
    mnist_std_np_train = np.array([0.3081]).reshape(1, 1, 1, 1)  # For (B,C,H,W)

    model = QSANN_image_classifier(S=S_VALUE_MNIST, n=N_QUBITS, D=D_QSAL, num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE)
    params = init_params(S=S_VALUE_MNIST, n=N_QUBITS, D=D_QSAL, num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE, input_patch_dim=INPUT_PATCH_DIM_MNIST)
    
    total_params = count_parameters(params)
    print(f"Total number of parameters in the QViT model for MNIST: {total_params:,}")

    initial_lr = 0.0005
    num_batches_per_epoch = len(train_loader)
    decay_steps = n_epochs * num_batches_per_epoch
    lr_schedule = optax.cosine_decay_schedule(init_value=initial_lr, decay_steps=decay_steps)
    optimizer = optax.adam(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    train_costs, test_costs, train_accs, test_accs, steps = [], [], [], [], []
    # New lists for additional metrics
    test_precision_macros = []
    test_recall_macros = []
    test_f1_macros = []
    test_top1_errors = []
    test_top5_errors = []
    epoch_fidelities = []  # Track fidelity per epoch

    def loss_fn_batch(p, x_batch, y_batch):
        logits = model(x_batch, p)
        loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_batch))
        return loss, logits

    @jax.jit
    def update_batch(params, opt_state, x_batch_jax, y_batch_jax):
        (loss_val, logits), grads = jax.value_and_grad(loss_fn_batch, has_aux=True)(params, x_batch_jax, y_batch_jax)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val

    current_params = params
    current_opt_state = opt_state
    start_time = time.time()

    # Track Lipschitz constants across epochs
    lipschitz_constants = {"l2": [], "inf": []}
    last_batch_x = None
    
    # Prepare fidelity evaluation data if enabled (only for largest training size)
    fidelity_data = None
    if measure_adversarial and n_train == max(train_sizes_mnist):
        print(f"Fidelity evaluation enabled for largest training size - will measure every {adversarial_frequency} epoch(s)")
        # Load test data for fidelity evaluation (load once, reuse)
        transform_raw_test = transforms.Compose([transforms.ToTensor()])
        testset_full_raw = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_raw_test)
        indices = np.random.choice(len(testset_full_raw), 50, replace=False)
        testset = [testset_full_raw[i] for i in indices]
        test_loader_adv_raw = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
        x_test_torch_adv, y_test_torch_adv_int = next(iter(test_loader_adv_raw))
        x_test_np_images_adv = x_test_torch_adv.numpy()
        x_test_np_images_hwc_adv = x_test_np_images_adv.transpose(0, 2, 3, 1)
        x_test_fid_patched = create_patches(jnp.array(x_test_np_images_hwc_adv))
        y_test_fid_int_labels = jnp.array(y_test_torch_adv_int.numpy())
        y_test_fid_one_hot = one_hot(y_test_fid_int_labels, num_classes=NUM_CLASSES)
        fidelity_data = {
            'x_test_fid_patched': x_test_fid_patched,
            'y_test_fid_one_hot': y_test_fid_one_hot
        }
    elif measure_adversarial:
        print("Fidelity evaluation disabled for smaller training sizes")
    else:
        print("Fidelity evaluation disabled for faster training")
    
    if measure_lipschitz and n_train == max(train_sizes_mnist):
        print(f"Lipschitz constant measurement enabled for largest training size - will measure every {lipschitz_frequency} epoch(s)")
    elif measure_lipschitz:
        print("Lipschitz constant measurement disabled for smaller training sizes")
    else:
        print("Lipschitz constant measurement disabled for faster training")
    
    for epoch in range(n_epochs):
        time_start_epoch = time.time()
        
        # Track batch losses only for monitoring (not for final metrics)
        epoch_batch_losses = []
        num_train_batches = 0

        for x_batch_torch, y_batch_torch in train_loader:
            x_batch_np = x_batch_torch.numpy()
            y_batch_np = y_batch_torch.numpy().reshape(-1, 1)

            x_batch_np = (x_batch_np * mnist_std_np_train) + mnist_mean_np_train
            x_batch_np = np.clip(x_batch_np, 0., 1.)
            x_batch_np = x_batch_np.transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C) -> (B,28,28,1)
            x_batch_patches_np = create_patches(x_batch_np) # MNIST patches
            
            x_batch_jax = jnp.array(x_batch_patches_np)
            y_batch_jax = jnp.array(y_batch_np)
            current_params, current_opt_state, batch_loss = update_batch(
                current_params, current_opt_state, x_batch_jax, y_batch_jax
            )
            epoch_batch_losses.append(float(batch_loss))
            num_train_batches += 1
            last_batch_x = x_batch_jax  # Store last batch for LC computation
        
        # Evaluate on entire training set using final epoch parameters
        train_loss, train_acc, _, _ = evaluate(model, current_params, train_loader)
        
        # Evaluate on test set using final epoch parameters
        test_loss, test_acc, test_all_logits, test_all_labels = evaluate(model, current_params, test_loader)
        
        # Calculate additional metrics for the test set
        test_y_true_np = np.array(test_all_labels.squeeze())
        test_y_pred_proba_np = np.array(jax.nn.softmax(test_all_logits, axis=-1))
        test_y_pred_np = np.array(jnp.argmax(test_all_logits, axis=-1))
        
        num_classes = test_all_logits.shape[-1] # Should be 10 for MNIST
        class_labels_for_report = [str(i) for i in range(num_classes)]
        
        # Calculate metrics manually for multiclass classification
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision_macro = precision_score(test_y_true_np, test_y_pred_np, average='macro', zero_division='warn')
        recall_macro = recall_score(test_y_true_np, test_y_pred_np, average='macro', zero_division='warn')
        f1_macro = f1_score(test_y_true_np, test_y_pred_np, average='macro', zero_division='warn')
        
        current_top1_error = 1.0 - test_acc 
        current_top5_acc = top_k_accuracy_score(test_y_true_np, test_y_pred_proba_np, k=5, labels=np.arange(num_classes))
        current_top5_error = 1.0 - current_top5_acc

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
        # avg_batch_loss = np.mean(epoch_batch_losses)
        
        # Compute Lipschitz constant using the last batch (only for largest training size)
        lipschitz_info = ""
        if last_batch_x is not None and measure_lipschitz and n_train == max(train_sizes_mnist) and (epoch + 1) % lipschitz_frequency == 0:
            lipschitz_l2 = float(lipschitz_bound_qvt(model, current_params, last_batch_x, "l2"))
            lipschitz_inf = float(lipschitz_bound_qvt(model, current_params, last_batch_x, "inf"))
            lipschitz_constants["l2"].append(lipschitz_l2)
            lipschitz_constants["inf"].append(lipschitz_inf)
            lipschitz_info = f" | Lipschitz L2: {lipschitz_l2:.6f} | Lipschitz Inf: {lipschitz_inf:.6f}"
        
        # Real-time fidelity evaluation (only for largest training size)
        fidelity_info = ""
        if measure_adversarial and fidelity_data is not None and n_train == max(train_sizes_mnist) and (epoch + 1) % adversarial_frequency == 0:
            adv_runner = Adversarial(
                model_fn=model,
                params=current_params,
                x=fidelity_data['x_test_fid_patched'],
                y=fidelity_data['y_test_fid_one_hot'],
                batch_size=64
            )
            # Run APGD attack
            attack_type = "APGD"
            attack_params_dict = {"eps": 8/255, "steps": 100, "decay": 0.75}
            fidel_val = adv_runner.run(
                tpe=attack_type, metric=False, fidelity_only = True, **attack_params_dict
            )
            epoch_fidelities.append({
                "epoch": epoch + 1,
                "fidelity": float(fidel_val),
                "attack_type": attack_type,
                "eps": attack_params_dict["eps"]
            })
            fidelity_info = f" | Fidelity: {fidel_val:.4f}"     
        
        print(f"Epoch {epoch+1}/{n_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | "
                f"Test Acc: {test_acc:.4f}"
                f"{lipschitz_info}{fidelity_info} | "
                f"Time: {time.time() - time_start_epoch:.2f}s")

        # Save model parameters for each epoch
        epoch_model_name = f"MNIST_{n_train}_rep{rep_num}_epoch{epoch+1}"
        save_model_params(current_params, epoch_model_name, "mnist")
        print(f"Epoch {epoch+1} parameters saved as: {epoch_model_name}")

        if epoch == n_epochs -1: # For the last epoch
            final_cm = confusion_matrix(test_y_true_np, test_y_pred_np, labels=np.arange(num_classes))
            plot_confusion_matrix(final_cm, class_labels_for_report, epoch + 1, n_train, rep_num, 
                                  D_PATCH_VALUE, NUM_LAYERS, batch_size, dataset_name="MNIST")

    print("--- Saving Trained Parameters ---")

    MODEL_NAME=f"MNIST_{n_train}_rep{rep_num}"
    save_model_params(current_params, MODEL_NAME, "mnist")

    # Also save model configuration for future loading
    model_config = {
        'S': S_VALUE_MNIST,
        'n_qubits': N_QUBITS,
        'D': D_QSAL,
        'num_layers': NUM_LAYERS,
        'd_patch': D_PATCH_VALUE,
        'input_patch_dim': INPUT_PATCH_DIM_MNIST,
        'num_classes': NUM_CLASSES,
        'patch_size': PATCH_SIZE_MNIST
    }
    save_model_config(model_config, MODEL_NAME, "mnist")
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds for n_train={n_train}, rep {rep_num} (MNIST)")

    if epoch_fidelities and n_train == max(train_sizes_mnist):
        fidelity_save_data = {
            "model_name": MODEL_NAME,
            "n_train": n_train,
            "rep_num": rep_num,
            "fidelity_per_epoch": epoch_fidelities
        }
        save_fidelity_metrics(fidelity_save_data, MODEL_NAME, "mnist")

    # Save Lipschitz constants data to organized directory structure
    if n_train == max(train_sizes_mnist):
        base_dir = create_experiment_dirs("mnist")
        lipschitz_file_name = f"qvit_mnist_{n_train}_rep{rep_num}_lipschitz.csv"
        lipschitz_path = os.path.join(base_dir, "metrics", lipschitz_file_name)
        lipschitz_df = pd.DataFrame({"epoch": list(range(1, n_epochs+1)), "l2": lipschitz_constants["l2"], "inf": lipschitz_constants["inf"]})
        lipschitz_df.to_csv(lipschitz_path, index=False)
        print(f"Lipschitz constants saved to {lipschitz_path}")

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
        'batch_size': [batch_size] * len(steps),
        'repetition': [rep_num] * len(steps)
    })
    return results_df

# Constants for MNIST experiment
# MNIST has 60,000 training images, 10,000 test images
# n_test_mnist = 10000 # Using full test set for MNIST
n_epochs_mnist = 2  # MNIST might train faster, adjust as needed
n_reps_mnist = 1
train_sizes_mnist = [50,100] # Example sizes for MNIST
BATCH_SIZE_MNIST = 64

def run_iterations(n_train, current_batch_size, num_epochs_current, lipschitz_frequency=1, measure_lipschitz=True, 
                  adversarial_frequency=1, measure_adversarial=True):
    all_results = []
    adv_results = []

    for rep in range(n_reps_mnist):
        print(f"\nStarting repetition {rep + 1}/{n_reps_mnist} for train size {n_train}, batch size {current_batch_size} (MNIST)")
        # Ensure n_test is an integer for np.random.choice
        n_test_int = min(int(n_train // 5), 10000)
        results_df = train_qvit(n_train, n_test_int, num_epochs_current, batch_size=current_batch_size, rep_num=rep + 1,
                               lipschitz_frequency=lipschitz_frequency, measure_lipschitz=measure_lipschitz,
                               adversarial_frequency=adversarial_frequency, measure_adversarial=measure_adversarial)
        all_results.append(results_df)

        if n_train == max(train_sizes_mnist):

            # --- Adversarial Evaluation: run after training, using in-memory model/params ---
            N_TEST_ADVERSARIAL = 50
            
            def load_mnist_data_raw(n_samples, train=False, batch_size=64):
                transform = transforms.Compose([transforms.ToTensor()])
                dataset_full = torchvision.datasets.MNIST(root='./data', train=train, download=True, transform=transform)
                if n_samples < len(dataset_full):
                    indices = np.random.choice(len(dataset_full), n_samples, replace=False)
                    dataset_full = torch.utils.data.Subset(dataset_full, indices)
                xs, ys = zip(*[(x, y) for x, y in dataset_full])
                x_all_np = np.stack([np.array(x) for x in xs], axis=0)
                x_all_np = np.clip(x_all_np, 0., 1.)
                x_all_np = x_all_np.transpose(0, 2, 3, 1)
                y_all_np = np.array(ys)
                return jnp.array(x_all_np), jax.nn.one_hot(jnp.array(y_all_np), NUM_CLASSES)
            
            x_test_fid_images, y_test_fid_one_hot = load_mnist_data_raw(N_TEST_ADVERSARIAL, train=False, batch_size=64)
            x_test_fid_patched = create_patches(x_test_fid_images, patch_size=PATCH_SIZE_MNIST)
            qvt_model_instance = QSANN_image_classifier(
                S=S_VALUE_MNIST, n=N_QUBITS, D=D_QSAL,
                num_layers=NUM_LAYERS, d_patch_config=D_PATCH_VALUE
            )
                # Load parameters before creating adversarial runner
            trained_params = None
            try:
                # Try to load from the organized directory structure first
                trained_params = load_model_params(f"MNIST_{n_train}_rep{rep + 1}", "mnist")
            except Exception:
                print("Warning: Could not load trained params for adversarial evaluation.")
                continue
            
            adv_runner = Adversarial(
                model_fn=qvt_model_instance,
                params=trained_params,
                x=x_test_fid_patched,
                y=y_test_fid_one_hot,
                batch_size=64
            )

            print(f"Clean accuracy on {N_TEST_ADVERSARIAL} adversarial test samples: {adv_runner.acc_clean:.4f}")
            # Only run APGD attack
            attack_type = "APGD"
            attack_params_dict = {"eps": 8/255, "steps": 100, "decay": 0.75}
            print(f"--- Running Attack: {attack_type} with params: {attack_params_dict} ---")
            acc_adv, asr, robustness_gap, fidel_val = adv_runner.run(
                tpe=attack_type, metric=False, **attack_params_dict
            )
            print(f"Results for {attack_type} ({attack_params_dict}):")
            print(f"  Adversarial Accuracy: {acc_adv:.4f}")
            print(f"  Attack Success Rate (ASR): {asr:.4f}")
            print(f"  Robustness Gap: {robustness_gap:.4f}")
            print(f"  Fidelity: {fidel_val:.4f}")
            
            adv_results.append({
                "n_train": n_train,
                "attack_type": attack_type,
                "attack_params": str(attack_params_dict),
                "clean_accuracy": float(adv_runner.acc_clean),
                "adv_accuracy": float(acc_adv),
                "asr": float(asr),
                "robustness_gap": float(robustness_gap),
                "fidelity": float(fidel_val)
            })
    if adv_results:
        base_dir = create_experiment_dirs("mnist")
        adv_df = pd.DataFrame(adv_results)
        adv_csv_name = f"qvt_mnist_adversarial_results.csv"
        adv_path = os.path.join(base_dir, "adversarial", adv_csv_name)
        adv_df.to_csv(adv_path, index=False)
        print(f"Adversarial results saved to {adv_path}")
    return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":

    all_results_collected_mnist = []
    for n_train_current in train_sizes_mnist:
        print(f"=== Starting training for MNIST: train size {n_train_current}, batch size {BATCH_SIZE_MNIST}, epochs {n_epochs_mnist} ===")
        results = run_iterations(n_train_current, BATCH_SIZE_MNIST, n_epochs_mnist, 
                            lipschitz_frequency=1, measure_lipschitz=True,
                            adversarial_frequency=1, measure_adversarial=True)
        all_results_collected_mnist.append(results)

    results_df_combined_mnist = pd.concat(all_results_collected_mnist, ignore_index=True)

    df_agg_mnist = results_df_combined_mnist.groupby(["n_train", "batch_size", "step"]).agg({
        "train_cost": ["mean", "std"],
        "test_cost": ["mean", "std"],
        "train_acc": ["mean", "std"],
        "test_acc": ["mean", "std"],
        "test_precision_macro": ["mean", "std"],
        "test_recall_macro": ["mean", "std"],
        "test_f1_macro": ["mean", "std"],
        "test_top1_error": ["mean", "std"],
        "test_top5_error": ["mean", "std"]
    }).reset_index()

    # Plotting
    sns.set_style('whitegrid')
    colors = sns.color_palette("viridis", n_colors=len(train_sizes_mnist)) # Different palette for clarity
    fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))
    generalization_errors_mnist = []

    for i, n_train in enumerate(train_sizes_mnist):
        df = df_agg_mnist[(df_agg_mnist.n_train == n_train) & (df_agg_mnist.batch_size == BATCH_SIZE_MNIST)]
        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
        axs = [0, 0, 2, 2]

        for k in range(4):
            ax = axes[axs[k]]
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=max(1, n_epochs_mnist // 10), color=colors[i], alpha=0.8)

        # Compute generalization error at the end of training
        if not df[df.step == n_epochs_mnist].empty:
            train_cost_final = df[df.step == n_epochs_mnist].train_cost["mean"].values[0]
            test_cost_final = df[df.step == n_epochs_mnist].test_cost["mean"].values[0]
            dif = test_cost_final - train_cost_final
            generalization_errors_mnist.append(dif)
        else: # Fallback if exact last epoch not found (should not happen with reset_index)
            generalization_errors_mnist.append(np.nan)


    axes[0].set_title('Train and Test Losses (MNIST)', fontsize=14)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(train_sizes_mnist, generalization_errors_mnist, "o-", label=r"$gen(\alpha)$")
    axes[1].set_xscale('log')
    axes[1].set_xticks(train_sizes_mnist)
    axes[1].set_xticklabels([str(ts) for ts in train_sizes_mnist]) # Ensure labels are strings
    axes[1].set_title(r'Generalization Error $gen(\alpha)$ (MNIST)', fontsize=14)
    axes[1].set_xlabel('Training Set Size')
    axes[1].set_yscale('log', base=2, nonpositive='clip') # clip nonpositive for log scale

    axes[2].set_title('Train and Test Accuracies (MNIST)', fontsize=14)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim(0.2, 1.05) # MNIST accuracies are typically higher

    legend_elements_mnist = (
        [mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes_mnist)] +
        [
            mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
            mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
        ]
    )

    axes[0].legend(handles=legend_elements_mnist, ncol=2) # Adjusted ncol for potentially fewer train sizes
    axes[2].legend(handles=legend_elements_mnist, ncol=2)

    plt.tight_layout()
    # Save to organized directory structure
    base_dir = create_experiment_dirs("mnist")
    plot_path = os.path.join(base_dir, "plots", 'qvit_mnist_learning_curves.png')
    plt.savefig(plot_path)
    plt.close()

    results_path = os.path.join(base_dir, "metrics", "qvit_mnist_results.csv")
    results_df_combined_mnist.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")
    print(f"Plots saved to {plot_path}")

    print("MNIST QViT script setup complete.")


