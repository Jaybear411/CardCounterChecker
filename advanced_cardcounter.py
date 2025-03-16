import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import seaborn as sns

class CardCounterDetector:
    def __init__(self, n_components=2, similarity_threshold=0.85, anomaly_contamination=0.1):
        self.n_components = n_components
        self.similarity_threshold = similarity_threshold
        self.anomaly_contamination = anomaly_contamination
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=n_components)
        self.anomaly_detector = IsolationForest(contamination=anomaly_contamination, random_state=42)
        self.known_card_counters = None
        
    def fit(self, X, known_counters=None):
        """
        Fit the detector on training data
        
        Parameters:
        -----------
        X : DataFrame
            The training data containing blackjack hands
        known_counters : np.array, optional
            Known card counter patterns in the reduced space
        """
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply SVD
        X_svd = self.svd.fit_transform(X_scaled)
        
        # Set known card counters or use defaults
        if known_counters is not None:
            self.known_card_counters = known_counters
        else:
            # Default synthetic card counter patterns
            self.known_card_counters = np.array([
                [0.8, 0.7],   # Conservative counter
                [0.9, 0.85],  # Aggressive counter
                [0.7, 0.9]    # Expert counter
            ])
        
        # Fit anomaly detector as an additional detection method
        self.anomaly_detector.fit(X_scaled)
        
        return self
    
    def detect(self, X):
        """
        Detect potential card counters in the data
        
        Parameters:
        -----------
        X : DataFrame
            The data containing blackjack hands to evaluate
            
        Returns:
        --------
        DataFrame with added columns for similarity scores and detection flags
        """
        # Create a copy to avoid modifying the original
        result = X.copy()
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Apply SVD
        X_svd = self.svd.transform(X_scaled)
        
        # Compute similarity to known card counters
        similarities = cosine_similarity(X_svd, self.known_card_counters)
        max_similarity = np.max(similarities, axis=1)
        
        # Detect anomalies using Isolation Forest
        anomaly_scores = self.anomaly_detector.decision_function(X_scaled)
        anomaly_flags = self.anomaly_detector.predict(X_scaled) == -1
        
        # Add results to the dataframe
        result['similarity_score'] = max_similarity
        result['anomaly_score'] = anomaly_scores
        result['suspected_counter_similarity'] = max_similarity > self.similarity_threshold
        result['suspected_counter_anomaly'] = anomaly_flags
        result['suspected_counter'] = (result['suspected_counter_similarity'] | 
                                      result['suspected_counter_anomaly'])
        
        return result, X_svd
    
    def visualize(self, X_svd, results):
        """
        Visualize the data in the reduced SVD space
        
        Parameters:
        -----------
        X_svd : np.array
            The data in the reduced SVD space
        results : DataFrame
            The results containing the detection flags
        """
        plt.figure(figsize=(12, 8))
        
        # Plot regular players
        regular = ~results['suspected_counter']
        plt.scatter(X_svd[regular, 0], X_svd[regular, 1], 
                   c='blue', label='Regular Player', alpha=0.6)
        
        # Plot suspected card counters
        counters = results['suspected_counter']
        plt.scatter(X_svd[counters, 0], X_svd[counters, 1], 
                   c='red', label='Suspected Counter', alpha=0.6)
        
        # Plot known card counter patterns
        plt.scatter(self.known_card_counters[:, 0], self.known_card_counters[:, 1], 
                   c='green', marker='*', s=200, label='Known Counter Pattern')
        
        # Add labels and legend
        plt.title('Player Behavior in SVD Space')
        plt.xlabel(f'Component 1 (Variance Explained: {self.svd.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Component 2 (Variance Explained: {self.svd.explained_variance_ratio_[1]:.2f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add a colorbar for similarity scores
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_svd[:, 0], X_svd[:, 1], 
                             c=results['similarity_score'], cmap='viridis', 
                             alpha=0.7)
        plt.colorbar(scatter, label='Similarity Score')
        
        # Plot known card counter patterns
        plt.scatter(self.known_card_counters[:, 0], self.known_card_counters[:, 1], 
                   c='red', marker='*', s=200, label='Known Counter Pattern')
        
        plt.title('Similarity Scores in SVD Space')
        plt.xlabel(f'Component 1 (Variance Explained: {self.svd.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'Component 2 (Variance Explained: {self.svd.explained_variance_ratio_[1]:.2f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Additional visualizations
        plt.figure(figsize=(10, 6))
        sns.heatmap(results[['similarity_score', 'anomaly_score', 
                           'suspected_counter_similarity', 'suspected_counter_anomaly', 
                           'suspected_counter']].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Between Detection Methods')
        plt.tight_layout()
        
        plt.show()

# --- Example Usage ---

# Generate a larger synthetic dataset
np.random.seed(42)
n_samples = 100

# Regular players (80%)
n_regular = int(n_samples * 0.8)
regular_data = {
    "dealer_upcard": np.random.randint(2, 12, n_regular),
    "player_hand_value": np.random.randint(4, 21, n_regular),
    "action": np.random.randint(0, 3, n_regular),  # 0 = Stand, 1 = Hit, 2 = Double Down
    "bet_size": np.random.randint(25, 150, n_regular),
    "running_count": np.random.randint(-3, 4, n_regular),
}

# Card counters (20%) - They adjust bets based on count and play more optimally
n_counters = n_samples - n_regular
counter_data = {
    "dealer_upcard": np.random.randint(2, 12, n_counters),
    "player_hand_value": np.random.randint(4, 21, n_counters),
    "action": np.random.randint(0, 3, n_counters),
}

# Card counters bet higher when count is higher
running_count = np.random.randint(-2, 8, n_counters)  # Biased toward positive counts
bet_size = np.array([max(50, min(500, 50 + count * 50)) for count in running_count])

counter_data["running_count"] = running_count
counter_data["bet_size"] = bet_size

# Combine datasets and shuffle
all_data = {}
for key in regular_data:
    all_data[key] = np.concatenate([regular_data[key], counter_data[key]])

# Create player_id to track individual players
all_data["player_id"] = np.concatenate([
    np.arange(1, n_regular + 1),
    np.arange(n_regular + 1, n_samples + 1)
])

# Create dataframe
df = pd.DataFrame(all_data)

# Initialize and fit the detector
detector = CardCounterDetector(n_components=2, similarity_threshold=0.8)
detector.fit(df[["dealer_upcard", "player_hand_value", "action", "bet_size", "running_count"]])

# Detect potential card counters
results, X_svd = detector.detect(df[["dealer_upcard", "player_hand_value", "action", "bet_size", "running_count"]])

# Add results to the original dataframe
df = pd.concat([df, results[["similarity_score", "anomaly_score", "suspected_counter"]]], axis=1)

# Print results summary
print("\nDetection Results:")
print(f"Total players: {len(df['player_id'].unique())}")
print(f"Suspected card counters: {df.groupby('player_id')['suspected_counter'].any().sum()}")

# Show detailed results for suspected counters
suspected_players = df[df['suspected_counter']]['player_id'].unique()
print("\nSuspected Card Counters (Player IDs):")
print(suspected_players)

# Display average statistics by player type
print("\nAverage Statistics by Player Type:")
player_types = df.groupby('player_id')['suspected_counter'].any()
player_stats = df.groupby('player_id').agg({
    'bet_size': 'mean',
    'running_count': 'mean',
    'similarity_score': 'mean',
    'anomaly_score': 'mean'
}).join(player_types)

print("\nRegular Players:")
print(player_stats[~player_stats['suspected_counter']].mean())
print("\nSuspected Card Counters:")
print(player_stats[player_stats['suspected_counter']].mean())

# Visualize the results
detector.visualize(X_svd, results)

# Save results to CSV
df.to_csv('cardcounter_results.csv', index=False)
print("\nResults saved to 'cardcounter_results.csv'")

# EXPERIMENTAL: Detect real-time play patterns
print("\n--- Real-time Detection Simulation ---")
print("Simulating a player's session...")

# Simulate a session of 10 hands for a potential card counter
session_data = {
    "dealer_upcard": [10, 7, 4, 9, 6, 3, 10, 8, 5, 7],
    "player_hand_value": [16, 18, 11, 15, 12, 10, 17, 19, 13, 14],
    "action": [0, 0, 2, 1, 1, 2, 0, 0, 1, 1],
    "bet_size": [50, 50, 100, 150, 200, 250, 300, 350, 250, 150],  # Increasing with count
    "running_count": [-1, 0, 1, 2, 3, 4, 5, 6, 5, 4]  # Count going up then down
}

session_df = pd.DataFrame(session_data)
session_results, session_svd = detector.detect(session_df)

print("\nReal-time Session Analysis:")
print(session_results[["similarity_score", "anomaly_score", "suspected_counter"]])
print(f"\nVerdict: {'Card counting detected!' if session_results['suspected_counter'].any() else 'Regular player'}")

# Plot bet size vs running count for the session
plt.figure(figsize=(10, 6))
plt.plot(session_df['running_count'], session_df['bet_size'], 'o-', color='purple')
plt.xlabel('Running Count')
plt.ylabel('Bet Size')
plt.title('Bet Size vs Running Count (Key Card Counting Indicator)')
plt.grid(True)
for i, txt in enumerate(range(1, len(session_df) + 1)):
    plt.annotate(f'Hand {txt}', (session_df['running_count'].iloc[i], session_df['bet_size'].iloc[i]),
                xytext=(5, 5), textcoords='offset points')
plt.show() 