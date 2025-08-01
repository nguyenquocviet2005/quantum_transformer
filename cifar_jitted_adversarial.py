# qvt_cifar_adversarial_optimized.py

#!/usr/bin/env python
# coding: utf-8

import os
# os.environ["JAX_PLATFORM_NAME"] = "cpu" # Ensure CPU execution if no GPU specific setup
import jax
import jax.numpy as jnp
from jax.nn import softmax, one_hot # Added one_hot for consistency
from jax import lax
import optax
import pennylane as qml
import time
import pandas as pd # Added for results_df
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg') # Use non-GUI backend for matplotlib
import json # New import for the updated Adversarial class
from functools import partial # Add this import at the top of your script

from jax import config
config.update("jax_enable_x64", True)

# --- Model Configuration Constants (Adapted for CIFAR-10) ---
D_PATCH_VALUE = 96  # From qvt_cifar_nodistill.py
NUM_LAYERS = 2      # From qvt_cifar_nodistill.py
PATCH_SIZE_CIFAR = 4 # CIFAR images are 32x32, 4x4 patches -> 8x8 = 64 patches
S_VALUE_CIFAR = (32 // PATCH_SIZE_CIFAR)**2 # Sequence length (number of patches)
INPUT_PATCH_DIM_CIFAR = (PATCH_SIZE_CIFAR**2) * 3 # For CIFAR-10 (3 channels)
N_QUBITS = 5            # Number of qubits for QSAL (example value, can be tuned)
D_QSAL = 1              # Depth of Q, K, V ansatzes for QSAL (example value, can be tuned)
NUM_CLASSES = 10        # CIFAR-10 has 10 classes
# --- End Model Configuration ---

# Global variables for the new Adversarial class's 'run' method
MODEL_NAME = "QVT_CIFAR" #
project_path = "." # Current directory for saving results

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Helper Functions for Transformer Components ---
@jax.jit
def layer_norm(x, gamma, beta, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normalized_x = (x - mean) / jnp.sqrt(variance + eps)
    return gamma * normalized_x + beta

@jax.jit
def feed_forward(x, w1, b1, w2, b2):
    x = jnp.dot(x, w1) + b1
    x = jax.nn.relu(x)
    x = jnp.dot(x, w2) + b2
    return x
# --- End Helper Functions ---

# --- QViT Model Classes (from qvt_mnist_nodistill.py, minor adjustments might be needed) ---
class QSAL_pennylane:
    def __init__(self, S, n, D, d_patch_config):
        self.seq_num = S
        self.num_q = n
        self.D = D
        self.d = d_patch_config
        self.dev = qml.device("default.qubit", wires=self.num_q)

        self.observables = []
        for i in range(self.d):
            qubit = i % self.num_q
            pauli_idx = (i // self.num_q) % 3
            if pauli_idx == 0: obs = qml.PauliZ(qubit)
            elif pauli_idx == 1: obs = qml.PauliX(qubit)
            else: obs = qml.PauliY(qubit)
            self.observables.append(obs)

        self.vqnod = qml.QNode(self.circuit_v, self.dev, interface="jax")
        self.qnod = qml.QNode(self.circuit_qk, self.dev, interface="jax")

    def _amplitude_embed_robust(self, inputs):
        # Pad or truncate inputs to 2^num_q
        expected_length = 2 ** self.num_q
        if inputs.shape[-1] > expected_length:
            inputs_processed = inputs[..., :expected_length]
        elif inputs.shape[-1] < expected_length:
            pad_width = [(0, 0)] * (inputs.ndim - 1) + [(0, expected_length - inputs.shape[-1])]
            inputs_processed = jnp.pad(inputs, pad_width, mode='constant', constant_values=0)
        else:
            inputs_processed = inputs

        norm = jnp.linalg.norm(inputs_processed, axis=-1, keepdims=True)
        # Add epsilon to norm to prevent division by zero for zero vectors
        normalized_inputs = jnp.where(norm > 1e-9, inputs_processed / (norm + 1e-9),
                                      jnp.ones_like(inputs_processed) / jnp.sqrt(expected_length))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=False) # Normalization done manually

    def circuit_v(self, inputs, weights):
        self._amplitude_embed_robust(inputs)
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j); qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for _ in range(self.D):
            for j in range(self.num_q): qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q): qml.RY(weights[idx], wires=j); idx += 1
        return [qml.expval(obs) for obs in self.observables]

    def circuit_qk(self, inputs, weights):
        self._amplitude_embed_robust(inputs)
        idx = 0
        for j in range(self.num_q):
            qml.RX(weights[idx], wires=j); qml.RY(weights[idx + 1], wires=j)
            idx += 2
        for _ in range(self.D):
            for j in range(self.num_q): qml.CNOT(wires=(j, (j + 1) % self.num_q))
            for j in range(self.num_q): qml.RY(weights[idx], wires=j); idx += 1
        return [qml.expval(qml.PauliZ(0))]

    def __call__(self, input_sequence, layer_params):
        batch_size, S, d_in = input_sequence.shape # d_in should be self.d
        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        input_flat = jnp.reshape(x_norm1, (-1, self.d))

        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, self.d)

        alpha = jnp.exp(-(Q_output - K_output) ** 2) # Simplified attention
        Sum_a = jnp.sum(alpha, axis=1, keepdims=True) # Sum over K_output's S dimension
        alpha_normalized = alpha / (Sum_a + 1e-9)

        Q_exp = Q_output[:, :, None, :] # (B, S_q, 1, 1)
        K_exp = K_output[:, None, :, :] # (B, 1, S_kv, 1)
        alpha_matrix = jnp.exp(-(Q_exp - K_exp)**2) # (B, S_q, S_kv, 1)
        Sum_a_matrix = jnp.sum(alpha_matrix, axis=2, keepdims=True) # (B, S_q, 1, 1)
        alpha_norm_matrix = alpha_matrix / (Sum_a_matrix + 1e-9)

        V_output_exp = V_output[:, None, :, :] # (B, 1, S_kv, d)
        weighted_V = alpha_norm_matrix * V_output_exp # (B, S_q, S_kv, d)
        qsa_out = jnp.sum(weighted_V, axis=2) # (B, S_q, d)

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
    def __init__(self, S, n, D, num_layers, d_patch_config, num_classes):
        self.Qnn = QSANN_pennylane(S, n, D, num_layers, d_patch_config)
        self.d_patch = d_patch_config
        self.S = S
        self.num_layers = num_layers
        self.num_classes = num_classes

    def __call__(self, x_patches, params): # x_patches is (B, S, input_patch_dim_actual)
        batch_size, S_actual, input_patch_dim_actual = x_patches.shape

        # Linear projection for patches
        x_flat_patches = x_patches.reshape(batch_size * S_actual, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat_patches, params['patch_embed_w']) + params['patch_embed_b']
        x_projected = projected_x_flat.reshape(batch_size, S_actual, self.d_patch)

        qnn_params_dict = params['qnn']
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)

        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1)

        logits = jnp.dot(x_flat_for_head, params['final']['weight']) + params['final']['bias']
        return logits
# --- End QViT Model Classes ---

# --- Loss and Metrics ---
@jax.jit
def softmax_cross_entropy_with_integer_labels(logits, labels):
    """Computes softmax cross entropy between logits and labels (integers)."""
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

@jax.jit
def accuracy_multiclass(logits, labels):
    """Computes accuracy for multi-class classification."""
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze())

# --- Loss and Metrics (JIT-compiled versions) ---

@partial(jax.jit, static_argnames=('model_fn', 'batch_size'))
def evaluate(model_fn, params, x, y, batch_size):
    """
    JIT-compiled function to evaluate the model accuracy.
    Uses lax.scan to iterate over batches efficiently.
    """
    n_samples = x.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    pad_amount = n_batches * batch_size - n_samples

    # Pad the data to be perfectly divisible by batch_size
    x_padded = jnp.pad(x, ((0, pad_amount), (0, 0), (0, 0)), 'constant')
    y_padded = jnp.pad(y, ((0, pad_amount), (0, 0)), 'constant')

    # Reshape into batches
    x_batched = x_padded.reshape(n_batches, batch_size, *x.shape[1:])
    y_batched = y_padded.reshape(n_batches, batch_size, *y.shape[1:])

    # Create a mask to ignore padded samples
    mask = jnp.arange(n_batches * batch_size) < n_samples
    mask_batched = mask.reshape(n_batches, batch_size)

    def eval_step(carry, batch):
        x_b, y_b, mask_b = batch
        logits = model_fn(x_b, params)
        preds = jnp.argmax(logits, axis=1)
        labels = jnp.argmax(y_b, axis=1)
        
        # Count correct predictions only for non-padded, real samples
        correct_in_batch = jnp.sum((preds == labels) * mask_b)
        carry += correct_in_batch
        return carry, None

    # Scan over the data and the mask
    total_correct, _ = lax.scan(eval_step, 0, (x_batched, y_batched, mask_batched))

    return total_correct / n_samples


@partial(jax.jit, static_argnames=('model_fn', 'batch_size'))
def evaluate_batched_for_adversarial(model_fn, params, x, batch_size):
    """
    JIT-compiled function to get predictions and probabilities for all samples.
    Uses lax.map for efficient batch processing.
    """
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
    
    # Reshape back to a flat list of predictions/probabilities
    # and slice off the padded elements to return the original sample size.
    all_preds = all_preds_b.reshape(-1)[:n_samples]
    all_probs = all_probs_b.reshape(-1, all_probs_b.shape[-1])[:n_samples]

    return all_preds, all_probs

def count_parameters(params):
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))

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

# --- Patch Creation (Adapted for CIFAR-10) ---
def create_patches(images, patch_size=PATCH_SIZE_CIFAR):
    # images: (B, H, W, C) e.g. (B, 32, 32, 3) for CIFAR
    batch_size = images.shape[0]
    img_size = images.shape[1] # Assuming square images (H=W), 32 for CIFAR

    num_patches_per_dim = img_size // patch_size
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size,
                            j*patch_size:(j+1)*patch_size, :] # Include all channels
            patch = patch.reshape(batch_size, -1) # Flatten patch (e.g., 4*4*3 = 48)
            patches.append(patch)
    return jnp.stack(patches, axis=1) # (B, S, patch_dim_flat)
# --- End Patch Creation ---

# --- Data Loading (Adapted for CIFAR-10, similar to qvt_cifar_nodistill.py but for raw 0-1 images) ---
def load_cifar10_data_for_training(n_train, n_test_val, batch_size, augment=False):
    cifar_mean = (0.4914, 0.4822, 0.4465)
    cifar_std = (0.2023, 0.1994, 0.2010)

    transform_list = [transforms.ToTensor()]
    if augment:
        transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip()
        ])
    transform_list.append(transforms.Normalize(cifar_mean, cifar_std))
    transform_train = transforms.Compose(transform_list)
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])

    trainset_full = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset_full = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_indices = np.random.choice(len(trainset_full), n_train, replace=False) if n_train < len(trainset_full) else np.arange(len(trainset_full))
    trainset = torch.utils.data.Subset(trainset_full, train_indices)

    test_indices = np.random.choice(len(testset_full), n_test_val, replace=False) if n_test_val < len(testset_full) else np.arange(len(testset_full))
    testset_val = torch.utils.data.Subset(testset_full, test_indices)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valloader = torch.utils.data.DataLoader(testset_val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return trainloader, valloader

def load_cifar10_data_raw(n_samples, train=False, batch_size=64):
    # No normalization for raw data, attacks expect 0-1 range.
    transform = transforms.Compose([transforms.ToTensor()]) # Converts images to [0,1] range

    dataset_full = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                download=True, transform=transform)

    if n_samples < len(dataset_full):
        indices = np.random.choice(len(dataset_full), n_samples, replace=False)
        dataset = torch.utils.data.Subset(dataset_full, indices)
    else:
        dataset = dataset_full

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=0) # Shuffle false for consistent test set

    # Extract all data from loader
    all_x, all_y = [], []
    for x_batch_torch, y_batch_torch in dataloader:
        all_x.append(x_batch_torch)
        all_y.append(y_batch_torch)

    x_all_torch = torch.cat(all_x, dim=0)
    y_all_torch = torch.cat(all_y, dim=0)

    # Convert to numpy, expected format (N, H, W, C) for images
    x_all_np = x_all_torch.numpy().transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)
    y_all_np = y_all_torch.numpy() # (B,)

    # Clip to 0-1 just in case, ToTensor should handle this.
    x_all_np = np.clip(x_all_np, 0., 1.)

    return jnp.array(x_all_np), jax.nn.one_hot(jnp.array(y_all_np), NUM_CLASSES) # CIFAR-10 has 10 classes

# --- End Data Loading ---

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


# --- Adversarial Attack Class (Final Corrected Version) ---
class Adversarial:
    def __init__(self, model_fn, params, x, y, batch_size=64):
        self.model_fn = model_fn
        self.params = params
        self.x = x
        self.y = y
        self.batch_size = batch_size

        # Use the JIT-compiled evaluation functions
        self.acc_clean = evaluate(self.model_fn, params, x, y, batch_size=batch_size)
        preds_clean, probs_clean = evaluate_batched_for_adversarial(self.model_fn, params, x, batch_size=batch_size)
        
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
    @partial(jax.jit, static_argnames=('model_fn', 'steps'))
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
    @partial(jax.jit, static_argnames=('model_fn', 'steps'))
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
    @partial(jax.jit, static_argnames=('model_fn', 'steps'))
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
        """JIT-compiled prediction function."""
        return model_fn(x, params)

    def run(self, tpe="FGSM", metric=False, **kwargs):
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

                # Generate adversarial examples with proper argument handling
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
                
                # Get predictions with a separate JIT-compiled predict function
                logits_adv = self._jitted_predict(self.model_fn, self.params, x_adv_batch)

                preds_adv = jnp.argmax(logits_adv, axis=1)
                fidel_adv_batch = softmax(logits_adv, axis=-1)

                correct_adv = (preds_adv == labels).sum()
                successful_attack = (correct_clean_batch & (preds_adv != labels)).sum()
                fidel_val_batch = jnp.sum(jnp.sqrt(fidel_clean_batch * fidel_adv_batch), axis=-1).sum()

                total_correct_adv += correct_adv
                total_successful_attack += successful_attack
                fidel_val_total += fidel_val_batch

            total_samples = len(self.x)
            num_correct_clean = jnp.sum(self.correct_clean)

            acc_adv = float(total_correct_adv / total_samples)
            asr = float(total_successful_attack / num_correct_clean)
            robustness_gap = float(self.acc_clean - acc_adv)
            fidel_val = float(fidel_val_total / total_samples)

            acc_adv_list.append(acc_adv)
            asr_list.append(asr)
            robustness_gap_list.append(robustness_gap)
            fidel_val_list.append(fidel_val)

        if metric:
            data = {
                "acc_adv": acc_adv_list, "asr": asr_list,
                "robustness_gap": robustness_gap_list, "fidel_val": fidel_val_list
            }
            file_name = f"{MODEL_NAME}_{tpe}_on_perturbation_magnitude.json"
            with open(os.path.join(project_path, file_name), "w") as f:
                json.dump(data, f, indent=4)
        else:
            return acc_adv_list[0], asr_list[0], robustness_gap_list[0], fidel_val_list[0]

# --- Minimal Training and Evaluation (Adapted for CIFAR-10) ---
# train_model is a generic MLP training function from the new adversarial context, not QVT specific.
# I will NOT include it here as it's not directly for QVT model training, which is already handled by train_single_qvt_model.

def train_single_qvt_model(
    n_train, n_test_val, n_epochs, batch_size,
    s_val, num_q_n, d_val, num_layers_val, d_patch_val, input_patch_dim_val, num_classes_val,
    lr=0.001, augment_data=False, measure_fidelity=True, fidelity_frequency=1
):
    """
    Train a single QVT model for CIFAR-10 classification.
    
    Args:
        measure_fidelity (bool): Whether to measure fidelity across epochs. Default: True
        fidelity_frequency (int): How often to measure fidelity (every N epochs). Default: 1 (every epoch)
    """
    train_loader, val_loader = load_cifar10_data_for_training(n_train, n_test_val, batch_size, augment=augment_data)

    global qvt_model_trained_instance # Declare global to make it accessible by `evaluate`
    qvt_model_trained_instance = QSANN_image_classifier(
        S=s_val, n=num_q_n, D=d_val,
        num_layers=num_layers_val, d_patch_config=d_patch_val, num_classes=num_classes_val
    )
    params = init_params(
        S=s_val, n=num_q_n, D=d_val, num_layers=num_layers_val,
        d_patch_config=d_patch_val, input_patch_dim=input_patch_dim_val, num_classes_out=num_classes_val
    )

    print(f"QVT Model Parameters: {count_parameters(params):,}")

    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init(params)

    cifar_mean_np = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 1, 3)
    cifar_std_np = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 1, 3)

    @jax.jit
    def update_batch(p, opt_s, x_jax, y_jax):
        def loss_fn(params_model, x_b, y_b):
            logits = qvt_model_trained_instance(x_b, params_model) # Use the global instance
            loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_b))
            return loss, logits

        (loss_val, logits_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, x_jax, y_jax)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        batch_acc = accuracy_multiclass(logits_val, y_jax)
        return new_p, new_opt_s, loss_val, batch_acc

    current_params = params
    current_opt_state = opt_state

    # Initialize fidelity tracking similar to Adversarial_classfin.py
    attack_list = ["FGSM", "PGD", "MIM", "APGD"]
    fidelity = {
        "FGSM": [],
        "PGD": [],
        "MIM": [],
        "APGD": []
    }

    # Prepare full training data for fidelity calculation
    all_x_patches = []
    all_y_one_hot = []
    
    # Only prepare fidelity data if we're going to measure it
    if measure_fidelity:
    for x_batch_torch, y_batch_torch in train_loader:
        x_np = x_batch_torch.numpy()
        y_np = y_batch_torch.numpy().reshape(-1, 1)

        # Denormalize to [0,1] before patching for consistency with adversarial attacks
        x_np_denormalized = np.zeros_like(x_np)
        for i in range(x_np.shape[1]): # Iterate over channels
            x_np_denormalized[:, i, :, :] = x_np[:, i, :, :] * cifar_std_np[0,0,0,i] + cifar_mean_np[0,0,0,i]

        x_np = np.clip(x_np_denormalized, 0., 1.)
        x_np_hwc = x_np.transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)

        x_patches_np = create_patches(x_np_hwc, patch_size=PATCH_SIZE_CIFAR)
        y_one_hot = one_hot(jnp.array(y_np.squeeze()), num_classes=num_classes_val)
        
        all_x_patches.append(x_patches_np)
        all_y_one_hot.append(y_one_hot)
    
    # Concatenate all batches to get the full training set
    x_train_patches_full = jnp.concatenate(all_x_patches, axis=0)
    y_train_one_hot_full = jnp.concatenate(all_y_one_hot, axis=0)

    for epoch in range(n_epochs):
        epoch_train_loss, epoch_train_acc, num_train_batches = 0.0, 0.0, 0
        for x_batch_torch, y_batch_torch in train_loader:
            x_np = x_batch_torch.numpy()
            y_np = y_batch_torch.numpy().reshape(-1, 1)

            # Denormalize to [0,1] before patching for consistency with adversarial attacks
            x_np_denormalized = np.zeros_like(x_np)
            for i in range(x_np.shape[1]): # Iterate over channels
                x_np_denormalized[:, i, :, :] = x_np[:, i, :, :] * cifar_std_np[0,0,0,i] + cifar_mean_np[0,0,0,i]

            x_np = np.clip(x_np_denormalized, 0., 1.)
            x_np_hwc = x_np.transpose(0, 2, 3, 1) # (B,C,H,W) -> (B,H,W,C)

            x_patches_np = create_patches(x_np_hwc, patch_size=PATCH_SIZE_CIFAR)
            x_jax_batch, y_jax_batch = jnp.array(x_patches_np), jnp.array(y_np)

            current_params, current_opt_state, b_loss, b_acc = update_batch(
                current_params, current_opt_state, x_jax_batch, y_jax_batch
            )
            epoch_train_loss += b_loss; epoch_train_acc += b_acc; num_train_batches += 1

        avg_epoch_train_loss = epoch_train_loss / num_train_batches
        avg_epoch_train_acc = epoch_train_acc / num_train_batches

        # Calculate fidelity for each attack type (conditionally)
        if measure_fidelity and (epoch + 1) % fidelity_frequency == 0:
        attacker = Adversarial(qvt_model_trained_instance, current_params, x_train_patches_full, y_train_one_hot_full, batch_size=32)
        for atk in attack_list:
            if atk == "FGSM":
                _, _, _, fidel_val = attacker.run(tpe=atk, eps=2/255)
            elif atk == "PGD":
                _, _, _, fidel_val = attacker.run(tpe=atk, eps=2/255, steps=100)
            elif atk == "MIM":
                _, _, _, fidel_val = attacker.run(tpe=atk, eps=2/255, steps=100, decay=1.0)
            elif atk == "APGD":
                _, _, _, fidel_val = attacker.run(tpe=atk, eps=2/255, steps=100, decay=0.75)
            
            fidelity[atk].append(fidel_val)

        if val_loader and (epoch + 1) % 5 == 0: # Validate every 5 epochs
            val_losses, val_accs, val_batches = 0,0,0
            for x_val_torch, y_val_torch in val_loader:
                x_val_np = x_val_torch.numpy()
                y_val_np_int = y_val_torch.numpy().reshape(-1,1)

                # Denormalize for validation as well
                x_val_np_denormalized = np.zeros_like(x_val_np)
                for i in range(x_val_np.shape[1]):
                    x_val_np_denormalized[:, i, :, :] = x_val_np[:, i, :, :] * cifar_std_np[0,0,0,i] + cifar_mean_np[0,0,0,i]

                x_val_np = np.clip(x_val_np_denormalized, 0., 1.).transpose(0,2,3,1)

                x_val_patches = create_patches(x_val_np, patch_size=PATCH_SIZE_CIFAR)

                logits_val = qvt_model_trained_instance(jnp.array(x_val_patches), current_params) # Use the global instance
                val_losses += jnp.mean(softmax_cross_entropy_with_integer_labels(logits_val, jnp.array(y_val_np_int)))
                val_accs += accuracy_multiclass(logits_val, jnp.array(y_val_np_int))
                val_batches +=1
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {avg_epoch_train_acc:.4f} | Val Loss: {val_losses/val_batches:.4f} | Val Acc: {val_accs/val_batches:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {avg_epoch_train_acc:.4f}")

    # Note: fidelity dictionary will be empty if measure_fidelity=False, 
    # or will contain values only for epochs that match fidelity_frequency
    return current_params, qvt_model_trained_instance, fidelity

# --- Main Adversarial Evaluation Script (Adapted for CIFAR-10) ---
if __name__ == "__main__":
    # Configuration for training vs loading
    TRAIN_NEW_MODEL = True  # Set to False to load pre-trained model
    LOAD_MODEL_NAME = MODEL_NAME  # Name of the model to load (if TRAIN_NEW_MODEL = False)

    if TRAIN_NEW_MODEL:
        # 1. Train QVT model
        # qvt_model_trained_instance is set as a global variable within train_single_qvt_model
        # Fidelity measurement options:
        # - measure_fidelity=True: Enable fidelity tracking (default)
        # - measure_fidelity=False: Disable fidelity tracking for faster training
        # - fidelity_frequency=N: Measure fidelity every N epochs (default: 1)
        # 
        # Example usage scenarios:
        # 1. Full fidelity tracking: measure_fidelity=True, fidelity_frequency=1
        # 2. Periodic fidelity tracking: measure_fidelity=True, fidelity_frequency=10
        # 3. No fidelity tracking (fastest): measure_fidelity=False
        # 4. End-of-training only: measure_fidelity=True, fidelity_frequency=n_epochs
        trained_qvt_params, _, fidelity = train_single_qvt_model(
            n_train=50000, n_test_val=10000, n_epochs=100, batch_size=64, # Example training parameters
            s_val=S_VALUE_CIFAR, num_q_n=N_QUBITS, d_val=D_QSAL,
            num_layers_val=NUM_LAYERS, d_patch_val=D_PATCH_VALUE,
            input_patch_dim_val=INPUT_PATCH_DIM_CIFAR, num_classes_val=NUM_CLASSES,
            lr=0.001, augment_data=False
        )
        print("--- QVT Model Trained ---")

        # Save fidelity data during training (similar to Adversarial_classfin.py)
        file_name = f"{MODEL_NAME}_fidelity.json"
        save_path = os.path.join(project_path, file_name)
        with open(save_path, "w") as f:
            json.dump(fidelity, f, indent=4)
        print(f"Fidelity data saved to: {save_path}")

        # Save trained parameters
        print("--- Saving Trained Parameters ---")
        save_model_params(trained_qvt_params, MODEL_NAME)

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

    else:
        # Load pre-trained model
        print("--- Loading Pre-trained Model ---")
        trained_qvt_params = load_model_params(LOAD_MODEL_NAME)
        model_config = load_model_config(LOAD_MODEL_NAME)

        # Recreate the model instance with loaded configuration
        global qvt_model_trained_instance
        qvt_model_trained_instance = QSANN_image_classifier(
            S=model_config['S'],
            n=model_config['n_qubits'],
            D=model_config['D'],
            num_layers=model_config['num_layers'],
            d_patch_config=model_config['d_patch'],
            num_classes=model_config['num_classes']
        )
        print("--- Pre-trained Model Loaded ---")

    # 2. Prepare data for adversarial attacks
    print("--- Preparing Test Data for Attacks ---")
    N_TEST_ADVERSARIAL = 10000  # Reduced from 10000 to avoid memory issues
    x_test_adv_images, y_test_adv_one_hot = load_cifar10_data_raw(N_TEST_ADVERSARIAL, train=False, batch_size=N_TEST_ADVERSARIAL)
    x_test_adv_patched = create_patches(x_test_adv_images, patch_size=PATCH_SIZE_CIFAR)  # (N, S, D)

    # 3. Instantiate Adversarial runner
    print("--- Initializing Adversarial Runner ---")
    adv_runner = Adversarial(
        model_fn = qvt_model_trained_instance, # Pass the QVT model instance directly
        params = trained_qvt_params,
        x = x_test_adv_patched,
        y = y_test_adv_one_hot,
        batch_size = 64
    )
    print(f"Clean accuracy on {N_TEST_ADVERSARIAL} adversarial test samples: {adv_runner.acc_clean:.4f}")

    # 4. Define and run attacks (using the new 'run' method)
    attacks_config = [
        {"type": "FGSM", "params": {"eps": 2/255}},
        {"type": "PGD", "params": {"eps": 2/255, "steps": 100}},
        {"type": "MIM", "params": {"eps": 2/255, "steps": 100, "decay": 1.0}},
        {"type": "APGD", "params": {"eps": 2/255, "steps": 100, "decay": 0.75}},
    ]

    attack_results_summary = []
    for attack_info in attacks_config:
        attack_type = attack_info["type"]
        attack_params_dict = attack_info["params"]
        print(f"--- Running Attack: {attack_type} with params: {attack_params_dict} ---")

        acc_adv, asr, robustness_gap, fidel_val = adv_runner.run(
            tpe=attack_type, metric=False, **attack_params_dict # Ensure metric=False for single run
        )

        print(f"Results for {attack_type} ({attack_params_dict}):")
        print(f"  Adversarial Accuracy: {acc_adv:.4f}")
        print(f"  Attack Success Rate (ASR): {asr:.4f}")
        print(f"  Robustness Gap: {robustness_gap:.4f}")
        print(f"  Fidelity: {fidel_val:.4f}")
        attack_results_summary.append({
            "attack_type": attack_type,
            "attack_params": str(attack_params_dict),
            "adv_accuracy": float(acc_adv),
            "asr": float(asr),
            "robustness_gap": float(robustness_gap),
            "fidelity": float(fidel_val),
            "clean_accuracy_on_set": float(adv_runner.acc_clean)
        })

    # 5. Save results
    print("--- Adversarial Attack Summary ---")
    results_df = pd.DataFrame(attack_results_summary)
    print(results_df)
    results_df.to_csv("qvt_cifar_adversarial_results.csv", index=False)
    print("Adversarial results saved to qvt_cifar_adversarial_results.csv")
    
    # 6. Additional robustness evaluation across different epsilon values (similar to Adversarial_classfin.py)
    print("--- Evaluating Robustness Across Different Epsilon Values ---")
    eps_list = [0.05, 0.1, 0.15, 0.2]
    attack_list = ["FGSM", "PGD", "MIM", "APGD"]
    acc_adv_history = {
          "FGSM": [],
          "PGD": [],
          "MIM": [],
          "APGD": []
    }
    
    for eps in eps_list:
        print(f"Testing with eps = {eps}")
        for atk in attack_list:
            attacker = Adversarial(qvt_model_trained_instance, trained_qvt_params, x_test_adv_patched, y_test_adv_one_hot, batch_size=32)
            if atk == "FGSM":
                acc_adv, asr, robustness_gap, _ = attacker.run(tpe=atk, eps=eps)
            elif atk == "PGD":
                acc_adv, asr, robustness_gap, _ = attacker.run(tpe=atk, eps=eps, steps=100)
            elif atk == "MIM":
                acc_adv, asr, robustness_gap, _ = attacker.run(tpe=atk, eps=eps, steps=100, decay=1.0)
            elif atk == "APGD":
                acc_adv, asr, robustness_gap, _ = attacker.run(tpe=atk, eps=eps, steps=100, decay=0.75)
            
            acc_adv_history[atk].append(acc_adv)
            print(f"  {atk} (eps={eps}): Adversarial Accuracy = {acc_adv:.4f}")

    print("--- Robustness Accuracy History ---")
    print(acc_adv_history)

    file_name = f"{MODEL_NAME}_robustness_acc.json"
    save_path = os.path.join(project_path, file_name)
    with open(save_path, "w") as f:
        json.dump(acc_adv_history, f, indent=4)
    print(f"Robustness accuracy data saved to: {save_path}")
    
    print("Script finished.")