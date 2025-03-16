import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns

class BettingPatternAnalyzer:
    """
    Analyzes betting patterns to detect card counting behavior
    
    Card counters typically display characteristic patterns:
    1. Increasing bets when the count is high (high cards remaining)
    2. Decreasing bets when the count is low (low cards remaining)
    3. More consistent strategy adherence compared to typical players
    """
    
    def __init__(self, correlation_threshold=0.6, kelly_threshold=0.2):
        """
        Initialize the BettingPatternAnalyzer
        
        Parameters:
        -----------
        correlation_threshold : float
            Threshold for detecting correlation between running count and bet size
        kelly_threshold : float
            Threshold for detecting optimal Kelly criterion behavior
        """
        self.correlation_threshold = correlation_threshold
        self.kelly_threshold = kelly_threshold
        self.scaler = StandardScaler()
        self.clustering = KMeans(n_clusters=2, random_state=42)
        
    def compute_bet_count_correlation(self, player_data):
        """
        Compute correlation between bet size and running count for a player
        
        Parameters:
        -----------
        player_data : DataFrame
            DataFrame containing bet_size and running_count for a player
            
        Returns:
        --------
        corr_coef : float
            Pearson correlation coefficient
        p_value : float
            p-value for the correlation
        """
        if len(player_data) < 5:  # Need at least 5 hands for meaningful correlation
            return 0, 1.0
            
        corr_coef, p_value = pearsonr(player_data['running_count'], player_data['bet_size'])
        return corr_coef, p_value
    
    def compute_kelly_adherence(self, player_data):
        """
        Compute how closely a player follows Kelly criterion betting
        
        Parameters:
        -----------
        player_data : DataFrame
            DataFrame containing bet_size and running_count for a player
            
        Returns:
        --------
        kelly_adherence : float
            Measure of adherence to Kelly criterion (0-1, higher = better adherence)
        """
        if len(player_data) < 5:
            return 0
            
        # Simplified Kelly model: bet should be proportional to advantage
        # Advantage is approximated by running_count / constant
        optimal_bets = player_data['running_count'].apply(lambda x: max(50, 50 + x * 50) if x > 0 else 50)
        actual_bets = player_data['bet_size']
        
        # Calculate MSE between optimal and actual bets (normalized)
        max_bet = max(actual_bets.max(), optimal_bets.max())
        mse = ((optimal_bets - actual_bets) ** 2).mean() / (max_bet ** 2)
        
        # Convert to adherence score (1 = perfect, 0 = worst)
        kelly_adherence = 1 - min(1, mse)
        return kelly_adherence
    
    def compute_betting_consistency(self, player_data):
        """
        Compute betting consistency relative to game state
        
        Parameters:
        -----------
        player_data : DataFrame
            DataFrame for a player
            
        Returns:
        --------
        consistency_score : float
            Measure of betting consistency (higher = more consistent strategy)
        """
        if len(player_data) < 5:
            return 0
            
        # Group by running count and compute variance of bets
        # Low variance at same count = more consistent strategy
        grouped = player_data.groupby('running_count')['bet_size'].agg(['mean', 'std'])
        
        # If std is NaN (only one sample), replace with 0
        grouped['std'] = grouped['std'].fillna(0)
        
        # Compute weighted average of std / mean (coefficient of variation)
        # Weight by number of samples in each group
        cv_values = []
        weights = []
        
        for count, group in grouped.iterrows():
            n_samples = len(player_data[player_data['running_count'] == count])
            if group['mean'] > 0:
                cv = group['std'] / group['mean']
            else:
                cv = 0
            cv_values.append(cv)
            weights.append(n_samples)
            
        if sum(weights) == 0:
            return 0
            
        weighted_cv = np.average(cv_values, weights=weights)
        
        # Convert to consistency score (1 = perfect consistency, 0 = highly variable)
        consistency_score = 1 - min(1, weighted_cv)
        return consistency_score
    
    def analyze_player_data(self, df):
        """
        Analyze betting patterns for all players in the dataset
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame containing player data with player_id, bet_size, and running_count
            
        Returns:
        --------
        player_analysis : DataFrame
            DataFrame containing pattern analysis results for each player
        """
        # Group by player
        player_ids = df['player_id'].unique()
        
        # Initialize results
        results = {
            'player_id': [],
            'hands_played': [],
            'avg_bet': [],
            'bet_count_correlation': [],
            'correlation_p_value': [],
            'kelly_adherence': [],
            'betting_consistency': [],
            'suspected_counter': []
        }
        
        # Analyze each player
        for player_id in player_ids:
            player_data = df[df['player_id'] == player_id]
            
            # Basic stats
            hands_played = len(player_data)
            avg_bet = player_data['bet_size'].mean()
            
            # Pattern analysis
            corr, p_value = self.compute_bet_count_correlation(player_data)
            kelly_adherence = self.compute_kelly_adherence(player_data)
            betting_consistency = self.compute_betting_consistency(player_data)
            
            # Flag as suspected counter based on criteria
            is_suspected = (
                (abs(corr) > self.correlation_threshold and p_value < 0.05) or
                (kelly_adherence > self.kelly_threshold and betting_consistency > 0.7)
            )
            
            # Store results
            results['player_id'].append(player_id)
            results['hands_played'].append(hands_played)
            results['avg_bet'].append(avg_bet)
            results['bet_count_correlation'].append(corr)
            results['correlation_p_value'].append(p_value)
            results['kelly_adherence'].append(kelly_adherence)
            results['betting_consistency'].append(betting_consistency)
            results['suspected_counter'].append(is_suspected)
        
        return pd.DataFrame(results)
    
    def cluster_players(self, player_analysis):
        """
        Cluster players based on their betting patterns
        
        Parameters:
        -----------
        player_analysis : DataFrame
            DataFrame containing pattern analysis results for each player
            
        Returns:
        --------
        player_analysis : DataFrame
            Input DataFrame with added cluster label column
        """
        # Select features for clustering
        features = player_analysis[['bet_count_correlation', 'kelly_adherence', 'betting_consistency']]
        
        # Standardize the features
        X_scaled = self.scaler.fit_transform(features)
        
        # Apply clustering
        clusters = self.clustering.fit_predict(X_scaled)
        
        # Add cluster labels to results
        player_analysis['cluster'] = clusters
        
        return player_analysis
    
    def visualize_patterns(self, df, player_analysis):
        """
        Visualize betting patterns for analysis
        
        Parameters:
        -----------
        df : DataFrame
            Original data with player_id, bet_size, and running_count
        player_analysis : DataFrame
            Results of analyze_player_data
        """
        # Setup plot style
        plt.style.use('seaborn-whitegrid')
        
        # 1. Correlation between running count and bet size
        plt.figure(figsize=(10, 6))
        sns.histplot(player_analysis['bet_count_correlation'], bins=20, kde=True)
        plt.axvline(self.correlation_threshold, color='red', linestyle='--', 
                   label=f'Threshold ({self.correlation_threshold})')
        plt.axvline(-self.correlation_threshold, color='red', linestyle='--')
        plt.title('Distribution of Bet-Count Correlation Coefficients')
        plt.xlabel('Pearson Correlation Coefficient')
        plt.ylabel('Count')
        plt.legend()
        
        # 2. Kelly Adherence vs Betting Consistency scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(player_analysis['kelly_adherence'], 
                             player_analysis['betting_consistency'],
                             c=player_analysis['suspected_counter'], 
                             cmap='coolwarm', alpha=0.7, s=100)
        
        # Annotate points with player IDs
        for i, txt in enumerate(player_analysis['player_id']):
            plt.annotate(txt, 
                        (player_analysis['kelly_adherence'].iloc[i], 
                         player_analysis['betting_consistency'].iloc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.axhline(0.7, color='gray', linestyle='--')
        plt.axvline(self.kelly_threshold, color='gray', linestyle='--')
        plt.colorbar(scatter, label='Suspected Counter')
        plt.title('Kelly Criterion Adherence vs Betting Consistency')
        plt.xlabel('Kelly Adherence Score')
        plt.ylabel('Betting Consistency Score')
        
        # 3. Betting patterns for specific players
        suspected = player_analysis[player_analysis['suspected_counter']]['player_id'].values
        regular = player_analysis[~player_analysis['suspected_counter']]['player_id'].values
        
        if len(suspected) > 0 and len(regular) > 0:
            # Select one suspected counter and one regular player
            suspected_id = suspected[0]
            regular_id = regular[0]
            
            plt.figure(figsize=(12, 10))
            
            # Suspected counter
            plt.subplot(2, 1, 1)
            suspected_data = df[df['player_id'] == suspected_id]
            plt.scatter(suspected_data['running_count'], suspected_data['bet_size'], 
                       color='red', alpha=0.7, s=100)
            
            # Add trend line
            z = np.polyfit(suspected_data['running_count'], suspected_data['bet_size'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(suspected_data['running_count'].min(), 
                                 suspected_data['running_count'].max(), 100)
            plt.plot(x_range, p(x_range), "r--", alpha=0.8)
            
            plt.title(f'Betting Pattern: Suspected Counter (Player {suspected_id})')
            plt.xlabel('Running Count')
            plt.ylabel('Bet Size')
            plt.grid(True)
            
            # Regular player
            plt.subplot(2, 1, 2)
            regular_data = df[df['player_id'] == regular_id]
            plt.scatter(regular_data['running_count'], regular_data['bet_size'], 
                       color='blue', alpha=0.7, s=100)
            
            # Add trend line
            z = np.polyfit(regular_data['running_count'], regular_data['bet_size'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(regular_data['running_count'].min(), 
                                 regular_data['running_count'].max(), 100)
            plt.plot(x_range, p(x_range), "b--", alpha=0.8)
            
            plt.title(f'Betting Pattern: Regular Player (Player {regular_id})')
            plt.xlabel('Running Count')
            plt.ylabel('Bet Size')
            plt.grid(True)
            
            plt.tight_layout()
        
        # 4. Cluster visualization
        plt.figure(figsize=(10, 8))
        cluster_scatter = plt.scatter(player_analysis['bet_count_correlation'], 
                                     player_analysis['kelly_adherence'],
                                     c=player_analysis['cluster'], 
                                     cmap='viridis', alpha=0.7, s=100)
        
        plt.colorbar(cluster_scatter, label='Cluster')
        plt.title('Clustering Results: Correlation vs Kelly Adherence')
        plt.xlabel('Bet-Count Correlation')
        plt.ylabel('Kelly Adherence Score')
        
        # Show all plots
        plt.tight_layout()
        plt.show()


# Example usage (when run as a standalone script)
if __name__ == "__main__":
    # Generate a synthetic dataset
    np.random.seed(42)
    n_players = 20
    n_hands_per_player = 15
    
    data = []
    
    # Create regular players (80%)
    for player_id in range(1, int(n_players * 0.8) + 1):
        for _ in range(n_hands_per_player):
            running_count = np.random.randint(-5, 6)
            # Regular players don't adjust bets based on count much
            bet_size = np.random.randint(25, 200)
            data.append({
                'player_id': player_id,
                'running_count': running_count,
                'bet_size': bet_size
            })
    
    # Create card counters (20%)
    for player_id in range(int(n_players * 0.8) + 1, n_players + 1):
        for _ in range(n_hands_per_player):
            running_count = np.random.randint(-5, 6)
            # Card counters adjust bets based on count
            bet_size = max(50, 50 + running_count * 45 + np.random.randint(-20, 21))
            data.append({
                'player_id': player_id,
                'running_count': running_count,
                'bet_size': bet_size
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Initialize the analyzer
    analyzer = BettingPatternAnalyzer(correlation_threshold=0.5, kelly_threshold=0.6)
    
    # Analyze player data
    player_analysis = analyzer.analyze_player_data(df)
    
    # Cluster players
    player_analysis = analyzer.cluster_players(player_analysis)
    
    # Print results
    print("Player Analysis Results:")
    print(player_analysis)
    
    print("\nSuspected Card Counters:")
    print(player_analysis[player_analysis['suspected_counter']])
    
    # Visualize results
    analyzer.visualize_patterns(df, player_analysis) 