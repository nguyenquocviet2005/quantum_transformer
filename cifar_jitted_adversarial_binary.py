# qvt_cifar_adversarial_binary.py

#!/usr/bin/env python
# coding: utf-8

import os
import jax
import jax.numpy as jnp
from jax.nn import softmax, one_hot
from jax import lax
import optax
import pennylane as qml
import time
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import json
from functools import partial

from jax import config
config.update("jax_enable_x64", True)

# --- Global Configuration ---
# Set TRAIN_NEW_MODEL=True to train new models from scratch
# Set LOAD_EXISTING_MODEL=True to load previously trained models
# Only one should be True at a time
TRAIN_NEW_MODEL = True
LOAD_EXISTING_MODEL = False  
MODEL_NAME = "cifar_binary"

# Global variable to store the trained model instance for adversarial evaluation
qvt_model_trained_instance = None

project_path = "cifar_binary_adversarial" # Current directory for saving results
if not os.path.exists(project_path):
    os.makedirs(project_path)
# --- Binary Classification Configuration ---
CIFAR_BINARY_PAIRS = {
    "01": (0, 1),  # airplane vs automobile
    "19": (1, 9),  # automobile vs truck
    "35": (3, 5)   # cat vs dog
}

CIFAR_CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# --- Model Configuration Constants ---
D_PATCH_VALUE = 96
NUM_LAYERS = 2
PATCH_SIZE_CIFAR = 4
S_VALUE_CIFAR = (32 // PATCH_SIZE_CIFAR)**2
INPUT_PATCH_DIM_CIFAR = (PATCH_SIZE_CIFAR**2) * 3
N_QUBITS = 5
D_QSAL = 1
NUM_CLASSES = 2  # Binary classification

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)

# --- Helper Functions ---
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

# --- QViT Model Classes ---
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
        expected_length = 2 ** self.num_q
        if inputs.shape[-1] > expected_length:
            inputs_processed = inputs[..., :expected_length]
        elif inputs.shape[-1] < expected_length:
            pad_width = [(0, 0)] * (inputs.ndim - 1) + [(0, expected_length - inputs.shape[-1])]
            inputs_processed = jnp.pad(inputs, pad_width, mode='constant', constant_values=0)
        else:
            inputs_processed = inputs

        norm = jnp.linalg.norm(inputs_processed, axis=-1, keepdims=True)
        normalized_inputs = jnp.where(norm > 1e-9, inputs_processed / (norm + 1e-9),
                                      jnp.ones_like(inputs_processed) / jnp.sqrt(expected_length))
        qml.AmplitudeEmbedding(normalized_inputs, wires=range(self.num_q), normalize=False)

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
        batch_size, S, d_in = input_sequence.shape
        x_norm1 = layer_norm(input_sequence, layer_params['ln1_gamma'], layer_params['ln1_beta'])
        input_flat = jnp.reshape(x_norm1, (-1, self.d))

        Q_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['Q']))(input_flat)).T
        K_output_flat = jnp.array(jax.vmap(lambda x_patch: self.qnod(x_patch, layer_params['K']))(input_flat)).T
        V_output_flat = jnp.array(jax.vmap(lambda x_patch: self.vqnod(x_patch, layer_params['V']))(input_flat)).T

        Q_output = Q_output_flat.reshape(batch_size, S, 1)
        K_output = K_output_flat.reshape(batch_size, S, 1)
        V_output = V_output_flat.reshape(batch_size, S, self.d)

        alpha = jnp.exp(-(Q_output - K_output) ** 2)
        Sum_a = jnp.sum(alpha, axis=1, keepdims=True)
        alpha_normalized = alpha / (Sum_a + 1e-9)

        Q_exp = Q_output[:, :, None, :]
        K_exp = K_output[:, None, :, :]
        alpha_matrix = jnp.exp(-(Q_exp - K_exp)**2)
        Sum_a_matrix = jnp.sum(alpha_matrix, axis=2, keepdims=True)
        alpha_norm_matrix = alpha_matrix / (Sum_a_matrix + 1e-9)

        V_output_exp = V_output[:, None, :, :]
        weighted_V = alpha_norm_matrix * V_output_exp
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
    def __init__(self, S, n, D, num_layers, d_patch_config, num_classes):
        self.Qnn = QSANN_pennylane(S, n, D, num_layers, d_patch_config)
        self.d_patch = d_patch_config
        self.S = S
        self.num_layers = num_layers
        self.num_classes = num_classes

    def __call__(self, x_patches, params):
        batch_size, S_actual, input_patch_dim_actual = x_patches.shape

        x_flat_patches = x_patches.reshape(batch_size * S_actual, input_patch_dim_actual)
        projected_x_flat = jnp.dot(x_flat_patches, params['patch_embed_w']) + params['patch_embed_b']
        x_projected = projected_x_flat.reshape(batch_size, S_actual, self.d_patch)

        qnn_params_dict = params['qnn']
        x_processed_qnn = self.Qnn(x_projected, qnn_params_dict)

        x_final_norm = layer_norm(x_processed_qnn, params['final_ln_gamma'], params['final_ln_beta'])
        x_flat_for_head = x_final_norm.reshape(x_final_norm.shape[0], -1)

        logits = jnp.dot(x_flat_for_head, params['final']['weight']) + params['final']['bias']
        return logits

# --- Binary Dataset Functions ---
def filter_binary_dataset(dataset, class_pair):
    """Filter dataset to only include specified binary classes and relabel them as 0/1"""
    class_0, class_1 = class_pair
    
    filtered_data = []
    for x, y in dataset:
        if y == class_0:
            filtered_data.append((x, 0))
        elif y == class_1:
            filtered_data.append((x, 1))
    
    return filtered_data

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, filtered_data):
        self.data = filtered_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_cifar_binary_data(n_train, n_test_val, batch_size, class_pair, augment=False):
    """Load CIFAR-10 data filtered for binary classification"""
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

    # Filter datasets for binary classification
    train_filtered = filter_binary_dataset(trainset_full, class_pair)
    test_filtered = filter_binary_dataset(testset_full, class_pair)
    
    # Limit dataset sizes
    if n_train < len(train_filtered):
        train_indices = np.random.choice(len(train_filtered), n_train, replace=False)
        train_filtered = [train_filtered[i] for i in train_indices]
    
    if n_test_val < len(test_filtered):
        test_indices = np.random.choice(len(test_filtered), n_test_val, replace=False)
        test_filtered = [test_filtered[i] for i in test_indices]

    # Create datasets and loaders
    binary_trainset = BinaryDataset(train_filtered)
    binary_testset = BinaryDataset(test_filtered)
    
    trainloader = torch.utils.data.DataLoader(binary_trainset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    testloader = torch.utils.data.DataLoader(binary_testset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return trainloader, testloader

def load_cifar_binary_data_raw(n_samples, class_pair, train=False, batch_size=64):
    """Load raw CIFAR-10 data for adversarial attacks (0-1 range)"""
    transform = transforms.Compose([transforms.ToTensor()])
    
    dataset_full = torchvision.datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)
    
    # Filter for binary classes
    filtered_data = filter_binary_dataset(dataset_full, class_pair)
    
    if n_samples < len(filtered_data):
        indices = np.random.choice(len(filtered_data), n_samples, replace=False)
        filtered_data = [filtered_data[i] for i in indices]
    
    binary_dataset = BinaryDataset(filtered_data)
    dataloader = torch.utils.data.DataLoader(binary_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Extract all data
    all_x, all_y = [], []
    for x_batch_torch, y_batch_torch in dataloader:
        all_x.append(x_batch_torch)
        all_y.append(y_batch_torch)
    
    x_all_torch = torch.cat(all_x, dim=0)
    y_all_torch = torch.cat(all_y, dim=0)
    
    # Convert to numpy, expected format (N, H, W, C) for images
    x_all_np = x_all_torch.numpy().transpose(0, 2, 3, 1)  # (B,C,H,W) -> (B,H,W,C)
    y_all_np = y_all_torch.numpy()
    
    x_all_np = np.clip(x_all_np, 0., 1.)
    
    return jnp.array(x_all_np), jax.nn.one_hot(jnp.array(y_all_np), NUM_CLASSES)

# --- Parameter Initialization ---
def init_params(S, n, D, num_layers, d_patch_config, input_patch_dim, num_classes_out, key_seed=42):
    key = jax.random.PRNGKey(key_seed)
    d_ffn = d_patch_config * 4

    num_random_keys = 2 + 1 + num_layers * 5 + 1
    keys = jax.random.split(key, num_random_keys)
    key_idx = 0

    patch_embed_w = jax.random.normal(keys[key_idx], (input_patch_dim, d_patch_config), dtype=jnp.float64) * jnp.sqrt(1.0 / input_patch_dim); key_idx+=1
    patch_embed_b = jax.random.normal(keys[key_idx], (d_patch_config,), dtype=jnp.float64) * 0.01; key_idx+=1

    pos_encoding_params = jax.random.normal(keys[key_idx], (S, d_patch_config), dtype=jnp.float64) * 0.02; key_idx+=1

    qnn_layers_params = []
    for _ in range(num_layers):
        layer_params = {
            'Q': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx], (n * (D + 2),), dtype=jnp.float64) - 1),
            'K': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+1], (n * (D + 2),), dtype=jnp.float64) - 1),
            'V': (jnp.pi / 4) * (2 * jax.random.normal(keys[key_idx+2], (n * (D + 2),), dtype=jnp.float64) - 1),
            'ln1_gamma': jnp.ones((d_patch_config,), dtype=jnp.float64), 'ln1_beta': jnp.zeros((d_patch_config,), dtype=jnp.float64),
            'ffn_w1': jax.random.normal(keys[key_idx+3], (d_patch_config, d_ffn), dtype=jnp.float64) * jnp.sqrt(1.0 / d_patch_config),
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
            'weight': jax.random.normal(keys[key_idx], (d_patch_config * S, num_classes_out), dtype=jnp.float64) * 0.01,
            'bias': jnp.zeros((num_classes_out,), dtype=jnp.float64)
        }
    }
    return params

def count_parameters(params):
    return sum(leaf.size for leaf in jax.tree_util.tree_leaves(params))

def save_model_params(params, model_name, project_path="."):
    import pickle
    params_save_path = os.path.join(project_path, f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Trained parameters saved to: {params_save_path}")
    return params_save_path

def load_model_params(model_name, project_path="."):
    import pickle
    params_save_path = os.path.join(project_path, f"{model_name}_trained_params.pkl")
    with open(params_save_path, 'rb') as f:
        params = pickle.load(f)
    print(f"Trained parameters loaded from: {params_save_path}")
    return params

def save_model_config(config, model_name, project_path="."):
    config_save_path = os.path.join(project_path, f"{model_name}_config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Model configuration saved to: {config_save_path}")
    return config_save_path

def load_model_config(model_name, project_path="."):
    config_save_path = os.path.join(project_path, f"{model_name}_config.json")
    with open(config_save_path, 'r') as f:
        config = json.load(f)
    print(f"Model configuration loaded from: {config_save_path}")
    return config

def create_patches(images, patch_size=PATCH_SIZE_CIFAR):
    batch_size = images.shape[0]
    img_size = images.shape[1]
    num_patches_per_dim = img_size // patch_size
    patches = []
    for i in range(num_patches_per_dim):
        for j in range(num_patches_per_dim):
            patch = images[:, i*patch_size:(i+1)*patch_size,
                          j*patch_size:(j+1)*patch_size, :]
            patch = patch.reshape(batch_size, -1)
            patches.append(patch)
    return jnp.stack(patches, axis=1)

# --- Binary Accuracy Function ---
@jax.jit
def accuracy_binary(logits, labels):
    """Computes accuracy for binary classification."""
    predicted_class = jnp.argmax(logits, axis=-1)
    return jnp.mean(predicted_class == labels.squeeze())

# --- Loss Functions ---
@jax.jit
def softmax_cross_entropy_with_integer_labels(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels.squeeze())

# --- Adversarial Attack Class ---
class Adversarial:
    def __init__(self, model_fn, params, x, y, batch_size=64):
        self.model_fn = model_fn
        self.params = params
        self.x = x
        self.y = y
        self.batch_size = batch_size

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
        return model_fn(x, params)

    def run(self, tpe="FGSM", metric=False, fidelity_only=False, **kwargs):
        """
        Run adversarial attack and compute metrics.
        If fidelity_only=True, only compute and return fidelity (for fast per-epoch fidelity tracking).
        """
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

        def _run_core(eps):
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
                asr = float(total_successful_attack / num_correct_clean)
                robustness_gap = float(self.acc_clean - acc_adv)
                fidel_val = float(fidel_val_total / total_samples)
                return acc_adv, asr, robustness_gap, fidel_val
            else:
                fidel_val = float(fidel_val_total / total_samples)
                return fidel_val

        acc_adv_list, asr_list, robustness_gap_list, fidel_val_list = [], [], [], []
        for eps in eps_list:
            if fidelity_only:
                fidel_val = _run_core(eps)
                fidel_val_list.append(fidel_val)
            else:
                acc_adv, asr, robustness_gap, fidel_val = _run_core(eps)
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
            if fidelity_only:
                return fidel_val_list[0]
            else:
                return acc_adv_list[0], asr_list[0], robustness_gap_list[0], fidel_val_list[0]

# --- Evaluation Functions ---
@partial(jax.jit, static_argnames=('model_fn', 'batch_size'))
def evaluate(model_fn, params, x, y, batch_size):
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

@partial(jax.jit, static_argnames=('model_fn', 'batch_size'))
def evaluate_batched_for_adversarial(model_fn, params, x, batch_size):
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
        if x.ndim == 2:
            x_batched = x[None, :, :]
        else:
            x_batched = x
        grad_fn = jax.grad(lambda inp: jnp.linalg.norm(model_fn(inp, params), ord=2))
        g = grad_fn(x_batched)
        if x.ndim == 2:
            g = g[0]
        if norm_type == "l2":
            return jnp.linalg.norm(g, ord=2)
        else:
            return jnp.linalg.norm(g, ord=jnp.inf)
    return jnp.max(jax.vmap(single_grad_norm)(x_batch))

# --- Training Function ---
def train_single_qvt_model(
    n_train, n_test_val, n_epochs, batch_size, class_pair,
    s_val, num_q_n, d_val, num_layers_val, d_patch_val, input_patch_dim_val, num_classes_val,
    lr=0.001, augment_data=False, measure_fidelity=True, fidelity_frequency=1, measure_lipschitz=True, lipschitz_frequency=1
):
    """
    Train a single QVT model for binary classification.
    Args:
        measure_fidelity (bool): Whether to measure fidelity across epochs. Default: True
        fidelity_frequency (int): How often to measure fidelity (every N epochs). Default: 1 (every epoch)
        measure_lipschitz (bool): Whether to measure Lipschitz constants across epochs. Default: True
        lipschitz_frequency (int): How often to measure Lipschitz constants (every N epochs). Default: 1 (every epoch)
    """
    train_loader, val_loader = load_cifar_binary_data(n_train, n_test_val, batch_size, class_pair, augment=augment_data)

    global qvt_model_trained_instance
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
            logits = qvt_model_trained_instance(x_b, params_model)
            loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, y_b))
            return loss, logits

        (loss_val, logits_val), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, x_jax, y_jax)
        updates, new_opt_s = optimizer.update(grads, opt_s, p)
        new_p = optax.apply_updates(p, updates)
        batch_acc = accuracy_binary(logits_val, y_jax)
        return new_p, new_opt_s, loss_val, batch_acc

    current_params = params
    current_opt_state = opt_state

    attack_list = ["FGSM", "PGD", "MIM", "APGD"]
    fidelity = {
        "FGSM": [],
        "PGD": [],
        "MIM": [],
        "APGD": []
    }
    # Track Lipschitz constants across epochs
    lipschitz_constants = {
        "l2": [],
        "inf": []
    }

    # Only prepare fidelity data if we're going to measure it
    if measure_fidelity:
        # Load test data for fidelity calculation
        x_test_adv_images, y_test_adv_one_hot = load_cifar_binary_data_raw(
            2000, class_pair, train=False, batch_size=2000
        )
        x_test_adv_patched = create_patches(x_test_adv_images, patch_size=PATCH_SIZE_CIFAR)
        all_x_test_patches = x_test_adv_patched
        all_y_test_one_hot = y_test_adv_one_hot
        print(f"Fidelity measurement enabled - will measure every {fidelity_frequency} epoch(s) using test set")
    else:
        print("Fidelity measurement disabled for faster training")
    if measure_lipschitz:
        print(f"Lipschitz constant measurement enabled - will measure every {lipschitz_frequency} epoch(s)")
    else:
        print("Lipschitz constant measurement disabled for faster training")

    for epoch in range(n_epochs):
        start_time = time.time()
        epoch_train_loss, epoch_train_acc, num_train_batches = 0.0, 0.0, 0
        last_batch_x = None
        for batch_idx, (x_batch_torch, y_batch_torch) in enumerate(train_loader):
            x_np = x_batch_torch.numpy()
            y_np = y_batch_torch.numpy().reshape(-1, 1)
            x_np_denormalized = np.zeros_like(x_np)
            for i in range(x_np.shape[1]):
                x_np_denormalized[:, i, :, :] = x_np[:, i, :, :] * cifar_std_np[0,0,0,i] + cifar_mean_np[0,0,0,i]
            x_np = np.clip(x_np_denormalized, 0., 1.)
            x_np_hwc = x_np.transpose(0, 2, 3, 1)
            x_patches_np = create_patches(x_np_hwc, patch_size=PATCH_SIZE_CIFAR)
            x_jax_batch, y_jax_batch = jnp.array(x_patches_np), jnp.array(y_np)
            current_params, current_opt_state, b_loss, b_acc = update_batch(
                current_params, current_opt_state, x_jax_batch, y_jax_batch
            )
            epoch_train_loss += b_loss; epoch_train_acc += b_acc; num_train_batches += 1
            last_batch_x = x_jax_batch
        avg_epoch_train_loss = epoch_train_loss / num_train_batches
        avg_epoch_train_acc = epoch_train_acc / num_train_batches
        # Compute Lipschitz constant using the last batch
        if last_batch_x is not None and measure_lipschitz and (epoch + 1) % lipschitz_frequency == 0:
            lipschitz_l2 = float(lipschitz_bound_qvt(qvt_model_trained_instance, current_params, last_batch_x, "l2"))
            lipschitz_inf = float(lipschitz_bound_qvt(qvt_model_trained_instance, current_params, last_batch_x, "inf"))
            lipschitz_constants["l2"].append(lipschitz_l2)
            lipschitz_constants["inf"].append(lipschitz_inf)
            lipschitz_info = f" | Lipschitz L2: {lipschitz_l2:.6f} | Lipschitz Inf: {lipschitz_inf:.6f}"
        else:
            lipschitz_info = ""
        # Calculate fidelity for each attack type using test set (conditionally)
        if measure_fidelity and (epoch + 1) % fidelity_frequency == 0:
            attacker = Adversarial(qvt_model_trained_instance, current_params, all_x_test_patches, all_y_test_one_hot, batch_size=512)
            for atk in attack_list:
                if atk == "FGSM":
                    fidel_val = attacker.run(tpe=atk, eps=8/255, fidelity_only=True)
                elif atk == "PGD":
                    fidel_val = attacker.run(tpe=atk, eps=8/255, steps=30, fidelity_only=True)
                elif atk == "MIM":
                    fidel_val = attacker.run(tpe=atk, eps=8/255, steps=30, decay=1.0, fidelity_only=True)
                elif atk == "APGD":
                    fidel_val = attacker.run(tpe=atk, eps=8/255, steps=30, decay=0.75, fidelity_only=True)
                fidelity[atk].append(fidel_val)
        if val_loader and (epoch + 1) % 5 == 0:
            val_losses, val_accs, val_batches = 0,0,0
            for x_val_torch, y_val_torch in val_loader:
                x_val_np = x_val_torch.numpy()
                y_val_np_int = y_val_torch.numpy().reshape(-1,1)
                x_val_np_denormalized = np.zeros_like(x_val_np)
                for i in range(x_val_np.shape[1]):
                    x_val_np_denormalized[:, i, :, :] = x_val_np[:, i, :, :] * cifar_std_np[0,0,0,i] + cifar_mean_np[0,0,0,i]
                x_val_np = np.clip(x_val_np_denormalized, 0., 1.).transpose(0,2,3,1)
                x_val_patches = create_patches(x_val_np, patch_size=PATCH_SIZE_CIFAR)
                logits_val = qvt_model_trained_instance(jnp.array(x_val_patches), current_params)
                val_losses += jnp.mean(softmax_cross_entropy_with_integer_labels(logits_val, jnp.array(y_val_np_int)))
                val_accs += accuracy_binary(logits_val, jnp.array(y_val_np_int))
                val_batches +=1
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {avg_epoch_train_acc:.4f} | Val Loss: {val_losses/val_batches:.4f} | Val Acc: {val_accs/val_batches:.4f}{lipschitz_info} | Time: {time.time() - start_time:.2f}s")
        else:
            print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_epoch_train_loss:.4f} | Train Acc: {avg_epoch_train_acc:.4f}{lipschitz_info} | Time: {time.time() - start_time:.2f}s")
    # Note: fidelity dictionary will be empty if measure_fidelity=False, 
    # or will contain values only for epochs that match fidelity_frequency
    return current_params, qvt_model_trained_instance, fidelity, lipschitz_constants

# --- Experiment Runner ---
def run_binary_adversarial_experiments():
    """Run adversarial experiments for all CIFAR binary pairs"""
    results_all_pairs = []
    
    for pair_name, class_pair in CIFAR_BINARY_PAIRS.items():
        class_0_name = CIFAR_CLASS_NAMES[class_pair[0]]
        class_1_name = CIFAR_CLASS_NAMES[class_pair[1]]
        print(f"\n=== Running experiments for pair {pair_name}: {class_pair} ({class_0_name} vs {class_1_name}) ===")
        
        # Model name for this specific pair
        pair_model_name = f"{MODEL_NAME}_{pair_name}"
        
        if TRAIN_NEW_MODEL:
        # Train new model for this pair
            print("--- Training QVT Model ---")
            trained_qvt_params, qvt_model_instance, fidelity, lipschitz_constants = train_single_qvt_model(
                n_train=10000, n_test_val=2000, n_epochs=100, batch_size=64, class_pair=class_pair,
                s_val=S_VALUE_CIFAR, num_q_n=N_QUBITS, d_val=D_QSAL,
                num_layers_val=NUM_LAYERS, d_patch_val=D_PATCH_VALUE, 
                input_patch_dim_val=INPUT_PATCH_DIM_CIFAR, num_classes_val=NUM_CLASSES,
                lr=0.001, augment_data=False, measure_fidelity=True, fidelity_frequency=1, measure_lipschitz=True, lipschitz_frequency=1
            )
            print("--- QVT Model Trained ---")
            fidelity_file_name = f"{pair_model_name}_fidelity.json"
            fidelity_save_path = os.path.join(project_path, fidelity_file_name)
            with open(fidelity_save_path, "w") as f:
                json.dump(fidelity, f, indent=4)
            print(f"Fidelity data saved to: {fidelity_save_path}")
            # Save Lipschitz constants data
            lipschitz_file_name = f"{pair_model_name}_lipschitz.json"
            lipschitz_save_path = os.path.join(project_path, lipschitz_file_name)
            with open(lipschitz_save_path, "w") as f:
                json.dump(lipschitz_constants, f, indent=4)
            print(f"Lipschitz constants saved to: {lipschitz_save_path}")
            print(f"  - L2 Lipschitz constants: {len(lipschitz_constants['l2'])} values")
            print(f"  - Inf Lipschitz constants: {len(lipschitz_constants['inf'])} values")
            save_model_params(trained_qvt_params, pair_model_name)
            model_config = {
                'S': S_VALUE_CIFAR, 'n_qubits': N_QUBITS, 'D': D_QSAL,
                'num_layers': NUM_LAYERS, 'd_patch': D_PATCH_VALUE,
                'input_patch_dim': INPUT_PATCH_DIM_CIFAR, 'num_classes': NUM_CLASSES,
                'patch_size': PATCH_SIZE_CIFAR, 'class_pair': class_pair
            }
            save_model_config(model_config, pair_model_name)
            
        else:
            # Load existing model for this pair
            print("--- Loading Existing QVT Model ---")
            try:
                trained_qvt_params = load_model_params(pair_model_name)
                model_config = load_model_config(pair_model_name)
                
                # Verify model configuration matches current settings
                expected_config = {
                    'S': S_VALUE_CIFAR, 'n_qubits': N_QUBITS, 'D': D_QSAL,
                    'num_layers': NUM_LAYERS, 'd_patch': D_PATCH_VALUE,
                    'input_patch_dim': INPUT_PATCH_DIM_CIFAR, 'num_classes': NUM_CLASSES,
                    'patch_size': PATCH_SIZE_CIFAR, 'class_pair': class_pair
                }
                
                # Check if configurations match
                config_matches = all(model_config.get(k) == v for k, v in expected_config.items())
                if not config_matches:
                    print("WARNING: Loaded model configuration doesn't match current settings!")
                    print(f"Expected: {expected_config}")
                    print(f"Loaded: {model_config}")
                
                # Initialize model instance with loaded configuration
                global qvt_model_trained_instance
                qvt_model_trained_instance = QSANN_image_classifier(
                    S=model_config['S'], n=model_config['n_qubits'], D=model_config['D'],
                    num_layers=model_config['num_layers'], d_patch_config=model_config['d_patch'], 
                    num_classes=model_config['num_classes']
                )
                qvt_model_instance = qvt_model_trained_instance
                
                print(f"--- QVT Model Loaded Successfully for pair {pair_name} ---")
                
            except FileNotFoundError as e:
                print(f"ERROR: Could not load model files for {pair_model_name}")
                print(f"Error: {e}")
                print("Please set TRAIN_NEW_MODEL=True or ensure model files exist")
                continue
                
        
        # Prepare test data for attacks
        print("--- Preparing Test Data for Attacks ---")
        N_TEST_ADVERSARIAL = 2000
        x_test_adv_images, y_test_adv_one_hot = load_cifar_binary_data_raw(
            N_TEST_ADVERSARIAL, class_pair, train=False, batch_size=N_TEST_ADVERSARIAL
        )
        x_test_adv_patched = create_patches(x_test_adv_images, patch_size=PATCH_SIZE_CIFAR)
        
        # Initialize adversarial runner
        print("--- Initializing Adversarial Runner ---")
        adv_runner = Adversarial(
            model_fn=qvt_model_instance,
            params=trained_qvt_params,
            x=x_test_adv_patched,
            y=y_test_adv_one_hot,
            batch_size=64
        )
        print(f"Clean accuracy on {N_TEST_ADVERSARIAL} adversarial test samples: {adv_runner.acc_clean:.4f}")
        
        # Define and run attacks
        attacks_config = [
            {"type": "FGSM", "params": {"eps": 8/255}},
            {"type": "PGD", "params": {"eps": 8/255, "steps": 30}},
            {"type": "MIM", "params": {"eps": 8/255, "steps": 30, "decay": 1.0}},
            {"type": "APGD", "params": {"eps": 8/255, "steps": 30, "decay": 0.75}},
        ]
        
        pair_results = []
        for attack_info in attacks_config:
            attack_type = attack_info["type"]
            attack_params_dict = attack_info["params"]
            print(f"--- Running Attack: {attack_type} with params: {attack_params_dict} ---")
            
            acc_adv, asr, robustness_gap, fidel_val = adv_runner.run(
                tpe=attack_type, metric=False, **attack_params_dict
            )
            
            print(f"Results for {attack_type} ({attack_params_dict}):")
            print(f"  Adversarial Accuracy: {acc_adv:.4f}")
            print(f"  Attack Success Rate (ASR): {asr:.4f}")
            print(f"  Robustness Gap: {robustness_gap:.4f}")
            print(f"  Fidelity: {fidel_val:.4f}")
            
            result = {
                "pair_name": pair_name,
                "class_pair": str(class_pair),
                "class_names": f"{class_0_name} vs {class_1_name}",
                "attack_type": attack_type,
                "attack_params": str(attack_params_dict),
                "clean_accuracy": float(adv_runner.acc_clean),
                "adv_accuracy": float(acc_adv),
                "asr": float(asr),
                "robustness_gap": float(robustness_gap),
                "fidelity": float(fidel_val)
            }
            pair_results.append(result)
            results_all_pairs.append(result)
        
        # Save pair-specific results
        pair_df = pd.DataFrame(pair_results)
        pair_csv_name = f"qvt_cifar_binary_{pair_name}_adversarial_results.csv"
        pair_df.to_csv(pair_csv_name, index=False)
        print(f"Results for pair {pair_name} saved to {pair_csv_name}")
    
    # Save combined results
    print("\n=== Combined Results Summary ===")
    results_df = pd.DataFrame(results_all_pairs)
    print(results_df)
    results_df.to_csv("qvt_cifar_binary_all_adversarial_results.csv", index=False)
    print("Combined adversarial results saved to qvt_cifar_binary_all_adversarial_results.csv")
    
    return results_df

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting CIFAR Binary Adversarial Experiments")
    print(f"Binary pairs: {CIFAR_BINARY_PAIRS}")
    print("Class names mapping:")
    for pair_name, class_pair in CIFAR_BINARY_PAIRS.items():
        class_0_name = CIFAR_CLASS_NAMES[class_pair[0]]
        class_1_name = CIFAR_CLASS_NAMES[class_pair[1]]
        print(f"  {pair_name}: {class_pair} -> {class_0_name} vs {class_1_name}")
    
    # Run experiments for all pairs
    all_results = run_binary_adversarial_experiments()
    
    print("\nExperiments completed successfully!") 