import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.prc import KeyGen, Encode

# Parameters
n = 1024
message_length = 0
false_positive_rate = 1e-5
# t will now be a range
num_trials = 100  # Number of trials per error rate
error_rates = np.linspace(0, 0.3, 100)  # Sweep from 0 to 30% error
t_values = range(3, 16) # Iterate t from 3 to 15

all_results = {}

for t_val in t_values:
    print(f"\nRunning experiment for t = {t_val}")
    # 1. Generate Keys
    encoding_key, decoding_key = KeyGen(n, message_length=message_length, false_positive_rate=false_positive_rate, t=t_val)

    # 2. Generate PRC codeword (random message)
    X_pm1 = Encode(encoding_key)  # {-1, 1}
    x_prc_binary = ((1 - X_pm1.numpy(force=True)) / 2).astype(int)  # {0, 1}

    # 3. Get syndrome check parameters
    parity_check_matrix = decoding_key[1]
    r = parity_check_matrix.shape[0]
    one_time_pad = decoding_key[2]
    # Convert one_time_pad to numpy array if needed
    if hasattr(one_time_pad, 'numpy'):
        z_pub = one_time_pad.numpy(force=True)
    else:
        z_pub = np.array(one_time_pad)

    threshold = (0.5 - r ** (-0.25)) * r
    print(f"Syndrome threshold: {threshold:.2f} (r={r})")

    def detect(codeword_arg):
        # codeword_arg: numpy array of shape (n,) with values in {0,1}
        y_eff_arg = (codeword_arg ^ z_pub) % 2
        S_arg = (parity_check_matrix @ y_eff_arg) % 2
        hamming_weight_arg = np.sum(S_arg)
        return hamming_weight_arg < threshold, hamming_weight_arg

    # 4. Run experiment for current t_val
    current_t_results = []
    for p in tqdm(error_rates, desc=f"Error rate sweep (t={t_val})"):
        detections = 0
        for _ in range(num_trials):
            # Corrupt codeword: flip each bit with probability p
            flips = np.random.rand(n) < p
            corrupted = (x_prc_binary ^ flips.astype(int)) % 2
            detected, _ = detect(corrupted)
            if detected:
                detections += 1
        detection_rate = detections / num_trials
        current_t_results.append(detection_rate)
        # print(f"Error rate {p:.3f}: detection rate {detection_rate:.2f}") # Optional: print per error rate
    all_results[t_val] = current_t_results

# 5. Plot all results on the same graph
plt.figure(figsize=(10,7))
alpha_values = np.linspace(1.0, 0.2, len(all_results))
for t_val, results_list in all_results.items():
    plt.plot(error_rates, results_list, marker='o', markersize=3, label=f't={t_val}', color='blue', alpha=alpha_values[t_val-3])

plt.xlabel('Bit Error Rate ($p_{emb}$)')
plt.ylabel('Detection Rate')
plt.title(f'PRC Substitution Error Tolerance (n={n}, trials={num_trials}, t=3-15)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('prc_error_tolerance_multiple_t.png')
plt.close()

print("\nExperiment complete. Plot saved to prc_error_tolerance_multiple_t.png")


# --- Experiment 2: Vary n, fixed t ---
print("\n\nStarting Experiment 2: Varying n, fixed t")

# Parameters for n-sweep experiment
n_values = [256, 512, 1024, 2048]  # Sweep n
t_fixed = 7  # Fixed t value
# Re-use: message_length, false_positive_rate, num_trials, error_rates from above

all_results_n_sweep = {}

for n_val in n_values:
    print(f"\nRunning experiment for n = {n_val}, t = {t_fixed}")
    # 1. Generate Keys
    # Note: message_length, false_positive_rate are taken from the global scope defined earlier
    encoding_key_n, decoding_key_n = KeyGen(n_val, message_length=message_length, false_positive_rate=false_positive_rate, t=t_fixed)

    # 2. Generate PRC codeword (random message)
    X_pm1_n = Encode(encoding_key_n)  # {-1, 1}
    x_prc_binary_n = ((1 - X_pm1_n.numpy(force=True)) / 2).astype(int)  # {0, 1}, shape (n_val,)

    # 3. Get syndrome check parameters for current n_val
    parity_check_matrix_n = decoding_key_n[1]
    r_n = parity_check_matrix_n.shape[0]
    one_time_pad_n = decoding_key_n[2]
    if hasattr(one_time_pad_n, 'numpy'):
        z_pub_n = one_time_pad_n.numpy(force=True)
    else:
        z_pub_n = np.array(one_time_pad_n)

    threshold_n = (0.5 - r_n ** (-0.25)) * r_n
    print(f"Syndrome threshold for n={n_val}: {threshold_n:.2f} (r={r_n})")

    # Define detection function for current n_val parameters
    def detect_for_n(codeword_arg_n):
        # codeword_arg_n: numpy array of shape (n_val,) with values in {0,1}
        y_eff_arg_n = (codeword_arg_n ^ z_pub_n) % 2
        S_arg_n = (parity_check_matrix_n @ y_eff_arg_n) % 2
        hamming_weight_arg_n = np.sum(S_arg_n)
        return hamming_weight_arg_n < threshold_n, hamming_weight_arg_n

    # 4. Run experiment for current n_val
    current_n_results = []
    # error_rates and num_trials are taken from the global scope defined earlier
    for p_err in tqdm(error_rates, desc=f"Error rate sweep (n={n_val}, t={t_fixed})"):
        detections_n = 0
        for _ in range(num_trials):
            # Corrupt codeword: flip each bit with probability p_err
            flips_n = np.random.rand(n_val) < p_err  # Use n_val here
            corrupted_n = (x_prc_binary_n ^ flips_n.astype(int)) % 2
            detected_n, _ = detect_for_n(corrupted_n)
            if detected_n:
                detections_n += 1
        detection_rate_n = detections_n / num_trials
        current_n_results.append(detection_rate_n)
    all_results_n_sweep[n_val] = current_n_results

# 5. Plot all results for n_sweep on the same graph
plt.figure(figsize=(10,7))
colors = plt.cm.viridis(np.linspace(0, 1, len(all_results_n_sweep)))

for i, (n_val_plot, results_list_plot) in enumerate(all_results_n_sweep.items()):
    plt.plot(error_rates, results_list_plot, marker='.', markersize=4, linestyle='-', label=f'n={n_val_plot}', color=colors[i])

plt.xlabel('Bit Error Rate ($p_{emb}$)')
plt.ylabel('Detection Rate')
plt.title(f'PRC Error Tolerance (t={t_fixed}, trials={num_trials}, n varied)')
plt.legend()
plt.grid(True)
plt.tight_layout()
new_plot_filename = f'prc_error_tolerance_multiple_n_t_fixed_{t_fixed}.png'
plt.savefig(new_plot_filename)
plt.close()

print(f"\nExperiment (n sweep) complete. Plot saved to {new_plot_filename}")
