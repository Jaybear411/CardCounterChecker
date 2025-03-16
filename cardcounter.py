import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# --- Step 1: Simulate a Blackjack Dataset ---
data = {
    "dealer_upcard": [10, 5, 10, 4, 10, 7, 8, 6, 10, 3],
    "player_hand_value": [17, 16, 15, 10, 18, 19, 13, 11, 20, 12],
    "action": [1, 0, 2, 1, 0, 1, 0, 2, 0, 1],  # 0 = Stand, 1 = Hit, 2 = Double Down
    "bet_size": [50, 50, 100, 50, 200, 75, 50, 120, 150, 60],
    "running_count": [1, 0, 2, -1, 3, 1, 0, 2, 3, -1],
}
df = pd.DataFrame(data)
print("Original Dataset:")
print(df)

# --- Step 2: Normalize the Dataset ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# --- Step 3: Apply Truncated SVD for Dimensionality Reduction ---
svd = TruncatedSVD(n_components=2)  # Reducing to 2 dimensions
X_svd = svd.fit_transform(X_scaled)
print("\nSVD Reduced Data (2D representation):")
print(X_svd)

# --- Step 4: Compare with Known Card Counters Using Cosine Similarity ---
# For demonstration, we create synthetic representations of known card counters in the reduced space.
known_card_counters = np.array([
    [0.8, 0.7],
    [0.9, 0.85],
    [0.7, 0.9]
])

# Compute cosine similarity between each hand's reduced representation and the known card counters.
similarities = cosine_similarity(X_svd, known_card_counters)
max_similarity = np.max(similarities, axis=1)  # Maximum similarity score per hand

# Add similarity scores to the dataset.
df['similarity_score'] = max_similarity

# --- Step 5: Flagging Suspected Card Counters ---
# Define a threshold for flagging potential card counters (e.g., similarity > 0.85)
threshold = 0.85
df['suspected_counter'] = df['similarity_score'] > threshold

print("\nSimilarity Scores and Suspected Card Counters:")
print(df)
