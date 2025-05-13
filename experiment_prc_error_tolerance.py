import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.prc import KeyGen, Encode

# Parameters
n = 1024
message_length = 0
false_positive_rate = 1e-5
t = 3
num_trials = 100  # Number of trials per error rate
error_rates = np.linspace(0, 0.3, 100)  # Sweep from 0 to 30% error

# 1. Generate Keys
encoding_key, decoding_key = KeyGen(n, message_length=message_length, false_positive_rate=false_positive_rate, t=t)

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

def detect(codeword):
    # codeword: numpy array of shape (n,) with values in {0,1}
    y_eff = (codeword ^ z_pub) % 2
    S = (parity_check_matrix @ y_eff) % 2
    hamming_weight = np.sum(S)
    return hamming_weight < threshold, hamming_weight

# 4. Run experiment
results = []
for p in tqdm(error_rates, desc="Error rate sweep"):
    detections = 0
    for _ in range(num_trials):
        # Corrupt codeword: flip each bit with probability p
        flips = np.random.rand(n) < p
        corrupted = (x_prc_binary ^ flips.astype(int)) % 2
        detected, _ = detect(corrupted)
        if detected:
            detections += 1
    detection_rate = detections / num_trials
    results.append(detection_rate)
    print(f"Error rate {p:.3f}: detection rate {detection_rate:.2f}")

# 5. Plot
plt.figure(figsize=(7,5))
plt.plot(error_rates, results, marker='o')
plt.xlabel('Bit Error Rate ($p_{emb}$)')
plt.ylabel('Detection Rate')
plt.title(f'PRC Substitution Error Tolerance (n={n}, t={t}, trials={num_trials})')
plt.grid(True)
plt.tight_layout()
plt.savefig('prc_error_tolerance.png')
plt.close()