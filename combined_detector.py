import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Import our custom modules
from advanced_cardcounter import CardCounterDetector
from pattern_recognition import BettingPatternAnalyzer

class CombinedCardCounterDetector:
    """
    A combined approach to detect card counting in Blackjack using both:
    1. SVD and cosine similarity (from CardCounterDetector)
    2. Betting pattern analysis (from BettingPatternAnalyzer)
    
    This provides more robust detection by using multiple methods.
    """
    
    def __init__(self, svd_components=2, similarity_threshold=0.8, 
                correlation_threshold=0.5, kelly_threshold=0.6, 
                anomaly_contamination=0.1):
        """
        Initialize the combined detector
        
        Parameters:
        -----------
        svd_components : int
            Number of SVD components to use
        similarity_threshold : float
            Threshold for SVD similarity detection
        correlation_threshold : float
            Threshold for bet-count correlation detection
        kelly_threshold : float
            Threshold for Kelly criterion adherence
        anomaly_contamination : float
            Expected fraction of anomalies for Isolation Forest
        """
        self.svd_detector = CardCounterDetector(
            n_components=svd_components,
            similarity_threshold=similarity_threshold,
            anomaly_contamination=anomaly_contamination
        )
        
        self.pattern_analyzer = BettingPatternAnalyzer(
            correlation_threshold=correlation_threshold,
            kelly_threshold=kelly_threshold
        )
        
    def fit(self, df):
        """
        Fit both detectors on the training data
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing blackjack hands with player_id
        """
        self.svd_detector.fit(
            df[["dealer_upcard", "player_hand_value", "action", "bet_size", "running_count"]]
        )
        return self
    
    def detect(self, df):
        """
        Apply both detection methods and combine the results
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing blackjack hands with player_id
            
        Returns:
        --------
        player_results : DataFrame
            Combined detection results by player
        hand_results : DataFrame
            Detection results for individual hands
        """
        # 1. SVD-based detection (hand level)
        svd_results, X_svd = self.svd_detector.detect(
            df[["dealer_upcard", "player_hand_value", "action", "bet_size", "running_count"]]
        )
        
        # Add SVD detection results to the original dataframe
        hand_results = df.copy()
        hand_results['svd_similarity'] = svd_results['similarity_score']
        hand_results['svd_suspected'] = svd_results['suspected_counter']
        
        # 2. Pattern-based detection (player level)
        pattern_results = self.pattern_analyzer.analyze_player_data(df)
        pattern_results = self.pattern_analyzer.cluster_players(pattern_results)
        
        # Compute player-level SVD results by aggregating hand-level results
        player_svd = hand_results.groupby('player_id').agg({
            'svd_similarity': 'mean',
            'svd_suspected': 'any'  # If any hand is flagged, flag the player
        })
        
        # Combine results
        player_results = pd.merge(
            pattern_results,
            player_svd,
            on='player_id'
        )
        
        # Final verdict: flagged by either method
        player_results['final_verdict'] = (
            player_results['suspected_counter'] | 
            player_results['svd_suspected']
        )
        
        return player_results, hand_results, X_svd
    
    def visualize_results(self, player_results, hand_results, X_svd):
        """
        Create visualizations to analyze detection results
        
        Parameters:
        -----------
        player_results : DataFrame
            Combined detection results by player
        hand_results : DataFrame
            Detection results for individual hands
        X_svd : numpy.ndarray
            Reduced SVD representation of hands
        """
        # 1. Basic detection summary
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        metrics = ['suspected_counter', 'svd_suspected', 'final_verdict']
        values = [player_results[metric].sum() for metric in metrics]
        labels = ['Pattern Analysis', 'SVD Analysis', 'Combined']
        
        # Create bar plot
        plt.bar(labels, values, color=['blue', 'green', 'red'])
        plt.axhline(len(player_results) * 0.2, color='black', linestyle='--', 
                   label='Expected Card Counters (20%)')
        
        plt.title('Detection Results by Method')
        plt.ylabel('Number of Players Flagged')
        plt.ylim(0, len(player_results))
        plt.legend()
        
        # 2. Agreement between methods
        plt.figure(figsize=(8, 8))
        
        # Create confusion matrix data
        confusion = pd.crosstab(
            player_results['suspected_counter'], 
            player_results['svd_suspected'],
            rownames=['Pattern'], 
            colnames=['SVD']
        )
        
        # Plot heatmap
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Agreement Between Detection Methods')
        
        # 3. Feature importance for detection
        plt.figure(figsize=(12, 6))
        
        # Create feature importance data for pattern analysis
        pattern_features = ['bet_count_correlation', 'kelly_adherence', 'betting_consistency']
        pattern_importance = []
        
        for feature in pattern_features:
            # Calculate logistic regression coefficient (approximation)
            flagged_mean = player_results[player_results['suspected_counter']][feature].mean()
            unflagged_mean = player_results[~player_results['suspected_counter']][feature].mean()
            importance = abs(flagged_mean - unflagged_mean)
            pattern_importance.append(importance)
        
        # Normalize importance values
        pattern_importance = pattern_importance / np.sum(pattern_importance)
        
        # Create feature importance data for SVD analysis
        # Using variance explained as importance measure
        svd_features = [f'SVD Component {i+1}' for i in range(self.svd_detector.n_components)]
        svd_importance = self.svd_detector.svd.explained_variance_ratio_
        
        # Plot feature importance
        plt.subplot(1, 2, 1)
        plt.bar(pattern_features, pattern_importance, color='blue')
        plt.title('Pattern Analysis Feature Importance')
        plt.xticks(rotation=45)
        plt.ylabel('Relative Importance')
        
        plt.subplot(1, 2, 2)
        plt.bar(svd_features, svd_importance, color='green')
        plt.title('SVD Component Importance')
        plt.ylabel('Variance Explained')
        
        plt.tight_layout()
        
        # 4. Detection results in SVD space
        self.svd_detector.visualize(X_svd, hand_results)
        
        # 5. Betting patterns
        self.pattern_analyzer.visualize_patterns(hand_results, player_results)
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Generate a synthetic dataset
    np.random.seed(42)
    n_players = 25
    n_hands_per_player = 20
    
    data = []
    
    # Create regular players (80%)
    n_regular = int(n_players * 0.8)
    for player_id in range(1, n_regular + 1):
        for _ in range(n_hands_per_player):
            dealer_upcard = np.random.randint(2, 12)
            player_hand_value = np.random.randint(4, 21)
            action = np.random.randint(0, 3)  # 0 = Stand, 1 = Hit, 2 = Double Down
            running_count = np.random.randint(-5, 6)
            
            # Regular players don't adjust bets based on count much
            bet_size = np.random.randint(25, 200)
            
            data.append({
                'player_id': player_id,
                'dealer_upcard': dealer_upcard,
                'player_hand_value': player_hand_value,
                'action': action,
                'bet_size': bet_size,
                'running_count': running_count
            })
    
    # Create card counters (20%)
    for player_id in range(n_regular + 1, n_players + 1):
        for _ in range(n_hands_per_player):
            dealer_upcard = np.random.randint(2, 12)
            player_hand_value = np.random.randint(4, 21)
            running_count = np.random.randint(-5, 6)
            
            # Card counters adjust strategy based on count
            if running_count > 0:
                # More likely to double down when count is high
                action_probs = [0.3, 0.3, 0.4]  # Stand, Hit, Double Down
            else:
                # More conservative when count is low
                action_probs = [0.5, 0.4, 0.1]  # Stand, Hit, Double Down
                
            action = np.random.choice([0, 1, 2], p=action_probs)
            
            # Card counters adjust bets based on count
            bet_size = max(50, 50 + running_count * 45 + np.random.randint(-20, 21))
            
            data.append({
                'player_id': player_id,
                'dealer_upcard': dealer_upcard,
                'player_hand_value': player_hand_value,
                'action': action,
                'bet_size': bet_size,
                'running_count': running_count
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize the combined detector
    detector = CombinedCardCounterDetector(
        svd_components=2,
        similarity_threshold=0.8,
        correlation_threshold=0.5,
        kelly_threshold=0.5
    )
    
    # Fit the detector
    detector.fit(df)
    
    # Detect potential card counters
    player_results, hand_results, X_svd = detector.detect(df)
    
    # Print results
    print("\nDetection Results:")
    print(f"Total players: {len(player_results)}")
    print(f"Players flagged by pattern analysis: {player_results['suspected_counter'].sum()}")
    print(f"Players flagged by SVD analysis: {player_results['svd_suspected'].sum()}")
    print(f"Players flagged by combined approach: {player_results['final_verdict'].sum()}")
    
    # Display detailed results for suspected counters
    print("\nSuspected Card Counters (Final Verdict):")
    suspected = player_results[player_results['final_verdict']]
    print(suspected[['player_id', 'bet_count_correlation', 'kelly_adherence', 
                    'svd_similarity', 'suspected_counter', 'svd_suspected']])
    
    # Check how well we detected the actual card counters
    actual_counters = set(range(n_regular + 1, n_players + 1))
    detected_counters = set(suspected['player_id'])
    
    true_positives = len(actual_counters.intersection(detected_counters))
    false_positives = len(detected_counters - actual_counters)
    false_negatives = len(actual_counters - detected_counters)
    
    print("\nDetection Performance:")
    print(f"True Positives: {true_positives} (correctly identified card counters)")
    print(f"False Positives: {false_positives} (regular players flagged as counters)")
    print(f"False Negatives: {false_negatives} (missed card counters)")
    
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
        print(f"Precision: {precision:.2f}")
    
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
        print(f"Recall: {recall:.2f}")
    
    # Visualize results
    detector.visualize_results(player_results, hand_results, X_svd)
    
    # Save results to CSV
    player_results.to_csv('player_detection_results.csv', index=False)
    print("\nResults saved to 'player_detection_results.csv'") 