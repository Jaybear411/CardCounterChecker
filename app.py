from flask import Flask, request, jsonify
try:
    from flask_cors import CORS
    print("Successfully imported Flask-CORS")
except ImportError:
    print("ERROR: Flask-CORS is not installed. Please run: pip install flask-cors")
    print("Continuing without CORS support (API requests from frontend may fail)")
    
    # Define a dummy CORS class to avoid errors
    class CORS:
        def __init__(self, app):
            print("Using dummy CORS implementation")
            pass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from scipy import linalg
from scipy.stats import multivariate_normal
import random
import json
import warnings
warnings.filterwarnings('ignore')

print("Starting Flask Card Counter Detector app...")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
print("Flask app initialized with CORS")

# Card counting detection system
class CardCounterDetector:
    def __init__(self, n_components=3, similarity_threshold=0.70):
        self.n_components = n_components
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=n_components)
        self.pca = PCA(n_components=n_components)
        
        # Known card counter patterns (expanded to 3 dimensions)
        # These represent different card counting strategies in the reduced dimensionality space
        self.known_card_counters = np.array([
            [0.85, 0.75, 0.65],   # Conservative counter (correlates bet with count but more subtle)
            [0.95, 0.90, 0.80],   # Aggressive counter (strong bet-count correlation)
            [0.75, 0.85, 0.90],   # Expert counter (sophisticated pattern, less detectable)
            [0.90, 0.70, 0.85],   # Hi-Lo counter (classic counting system)
            [0.80, 0.90, 0.75],   # Omega II counter (more complex counting system)
            [0.70, 0.80, 0.95]    # Wong Halves counter (precision betting strategy)
        ])
        
        # Store history for visualization and analysis
        self.history = []
        self.player_data = pd.DataFrame()
        
        # Correlation matrix and Cholesky decomposition
        self.correlation_matrix = None
        self.cholesky_factor = None
        
        # Kalman filter parameters
        self.kalman_state = np.zeros(3)  # Initial state estimate
        self.kalman_covariance = np.eye(3)  # Initial covariance estimate
        self.process_noise = 0.01 * np.eye(3)  # Process noise
        self.measurement_noise = 0.1 * np.eye(3)  # Measurement noise
        
        # Markov transition matrix for betting patterns
        self.markov_matrix = np.array([
            [0.7, 0.2, 0.1],  # Low bet -> Low, Medium, High
            [0.3, 0.5, 0.2],  # Medium bet -> Low, Medium, High
            [0.2, 0.3, 0.5]   # High bet -> Low, Medium, High
        ])
        
        print("Enhanced CardCounterDetector initialized with advanced linear algebra capabilities")
        
    def fit(self, X):
        """Fit the detector on initial data"""
        print("Fitting detector on initial data")
        if len(X) < 2:
            print("Warning: Not enough data for proper fitting. Using default values.")
            return self
            
        # Store the data for future analysis
        self.player_data = pd.concat([self.player_data, X], ignore_index=True)
        
        # Standard scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # SVD and PCA transformations
        self.svd.fit(X_scaled)
        self.pca.fit(X_scaled)
        
        # Calculate correlation matrix with handling for NaN values
        try:
            # Replace any NaN values with 0
            X_clean = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            self.correlation_matrix = np.corrcoef(X_clean, rowvar=False)
            
            # Check for NaN or inf values in correlation matrix
            if np.isnan(self.correlation_matrix).any() or np.isinf(self.correlation_matrix).any():
                print("Warning: NaN or inf values in correlation matrix. Using regularization.")
                self.correlation_matrix = np.eye(X_clean.shape[1])
            
            # Ensure the correlation matrix is positive definite
            min_eig = np.min(np.linalg.eigvals(self.correlation_matrix))
            if min_eig < 0:
                print(f"Adjusting correlation matrix with min eigenvalue: {min_eig}")
                self.correlation_matrix -= 10*min_eig * np.eye(*self.correlation_matrix.shape)
            
            # Compute Cholesky decomposition
            try:
                self.cholesky_factor = linalg.cholesky(self.correlation_matrix, lower=True)
                print("Cholesky decomposition successful")
            except np.linalg.LinAlgError:
                print("Warning: Correlation matrix not positive definite. Using regularization.")
                # Add a small regularization term
                reg_matrix = self.correlation_matrix + 1e-6 * np.eye(self.correlation_matrix.shape[0])
                self.cholesky_factor = linalg.cholesky(reg_matrix, lower=True)
                
        except Exception as e:
            print(f"Error in correlation calculation: {e}")
            print("Using identity matrix as fallback")
            self.correlation_matrix = np.eye(X_scaled.shape[1])
            self.cholesky_factor = np.eye(X_scaled.shape[1])
        
        print("Detector fitted successfully with correlation analysis and Cholesky decomposition")
        return self
    
    def kalman_update(self, measurement):
        """Update the Kalman filter with a new measurement"""
        try:
            # Check for NaN or inf values in measurement
            if np.isnan(measurement).any() or np.isinf(measurement).any():
                print("Warning: NaN or inf values in measurement. Using zeros.")
                measurement = np.zeros_like(measurement)
                
            # Prediction step
            predicted_state = self.kalman_state
            predicted_covariance = self.kalman_covariance + self.process_noise
            
            # Ensure covariance matrices are positive definite
            min_eig = np.min(np.linalg.eigvals(predicted_covariance))
            if min_eig < 0:
                predicted_covariance -= 10*min_eig * np.eye(*predicted_covariance.shape)
                
            # Calculate innovation covariance
            innovation_cov = predicted_covariance + self.measurement_noise
            
            # Ensure innovation covariance is invertible
            if np.linalg.det(innovation_cov) < 1e-10:
                print("Warning: Innovation covariance is singular. Adding regularization.")
                innovation_cov += 1e-6 * np.eye(innovation_cov.shape[0])
            
            # Update step
            kalman_gain = predicted_covariance @ np.linalg.inv(innovation_cov)
            self.kalman_state = predicted_state + kalman_gain @ (measurement - predicted_state)
            self.kalman_covariance = (np.eye(3) - kalman_gain) @ predicted_covariance
            
            return self.kalman_state
            
        except Exception as e:
            print(f"Error in Kalman update: {e}")
            # Return current state without updating
            return self.kalman_state
    
    def predict_next_bet(self, current_bet_category):
        """Predict the next bet using Markov chain"""
        # Map bet size to category (0=low, 1=medium, 2=high)
        if current_bet_category not in [0, 1, 2]:
            current_bet_category = 1  # Default to medium
            
        # Get probabilities for next bet
        next_bet_probs = self.markov_matrix[current_bet_category]
        
        # Return the most likely next bet category
        return np.argmax(next_bet_probs)
    
    def detect(self, hand_data):
        """Detect if a hand exhibits card counting behavior using advanced techniques"""
        print(f"Detecting card counting for hand data: {hand_data}")
        # Convert to DataFrame if it's a dict
        if isinstance(hand_data, dict):
            hand_data = pd.DataFrame([hand_data])
            
        # Add to player data history
        self.player_data = pd.concat([self.player_data, hand_data], ignore_index=True)
            
        # Scale the data
        try:
            X_scaled = self.scaler.transform(hand_data)
        except Exception as e:
            print(f"Error in scaling data: {e}")
            # Create a fallback scaled data with zeros
            X_scaled = np.zeros((hand_data.shape[0], len(self.scaler.mean_)))
        
        # Apply dimensionality reduction techniques
        try:
            X_svd = self.svd.transform(X_scaled)
        except Exception as e:
            print(f"Error in SVD transform: {e}")
            X_svd = np.zeros((X_scaled.shape[0], self.n_components))
        
        # If we have enough data, apply PCA as well
        try:
            if len(self.player_data) >= 3:
                X_pca = self.pca.transform(X_scaled)
            else:
                X_pca = np.zeros((X_scaled.shape[0], self.n_components))
        except Exception as e:
            print(f"Error in PCA transform: {e}")
            X_pca = np.zeros((X_scaled.shape[0], self.n_components))
        
        # Combine SVD and PCA results
        X_combined = (X_svd + X_pca) / 2
        
        # Update Kalman filter
        try:
            kalman_state = self.kalman_update(X_combined[0])
        except Exception as e:
            print(f"Error in Kalman update: {e}")
            kalman_state = np.zeros(self.n_components)
        
        # Compute similarity to known card counters
        try:
            similarities = cosine_similarity(X_combined, self.known_card_counters)
            max_similarity = np.max(similarities, axis=1)
        except Exception as e:
            print(f"Error in similarity calculation: {e}")
            max_similarity = np.array([0.0])
        
        # Calculate Mahalanobis distance if we have enough data
        mahalanobis_distance = 0
        if len(self.player_data) >= 5 and self.correlation_matrix is not None:
            try:
                # Ensure correlation matrix is invertible
                if np.linalg.det(self.correlation_matrix) > 1e-10:
                    inv_corr = np.linalg.inv(self.correlation_matrix)
                    x_centered = X_scaled[0] - np.mean(self.scaler.transform(self.player_data), axis=0)
                    mahalanobis_distance = np.sqrt(x_centered @ inv_corr @ x_centered.T)
                else:
                    print("Warning: Correlation matrix is singular, cannot compute Mahalanobis distance")
            except np.linalg.LinAlgError:
                print("Warning: Could not compute Mahalanobis distance - LinAlgError")
            except Exception as e:
                print(f"Warning: Could not compute Mahalanobis distance - {e}")
        
        # Calculate probability of card counting using multiple factors
        counter_probability = 0.0
        
        # Base similarity weight (30%)
        counter_probability += 0.3 * max_similarity[0]
        
        # Mahalanobis distance weight (20%)
        if mahalanobis_distance > 3.0:
            counter_probability += 0.2
        elif mahalanobis_distance > 2.0:
            counter_probability += 0.1
        
        # Bet-count correlation weight (25%)
        try:
            if len(self.player_data) >= 3:
                # Calculate correlation between bet size and true count (more accurate than running count)
                bet_count_corr = np.corrcoef(self.player_data['bet_size'], self.player_data['true_count'])[0, 1]
                if not np.isnan(bet_count_corr) and not np.isinf(bet_count_corr):
                    counter_probability += 0.25 * abs(bet_count_corr)
                    
                # Also check correlation with deck penetration (card counters bet more with higher penetration)
                bet_penetration_corr = np.corrcoef(self.player_data['bet_size'], self.player_data['deck_penetration'])[0, 1]
                if not np.isnan(bet_penetration_corr) and not np.isinf(bet_penetration_corr):
                    counter_probability += 0.15 * abs(bet_penetration_corr)
        except Exception as e:
            print(f"Error calculating correlations: {e}")
        
        # Betting pattern analysis (10%)
        try:
            if len(self.player_data) >= 5:
                # Calculate coefficient of variation for bet sizes
                bet_mean = np.mean(self.player_data['bet_size'])
                bet_std = np.std(self.player_data['bet_size'])
                bet_cv = bet_std / max(1, bet_mean)  # Coefficient of variation
                
                # Card counters typically have higher bet variation
                if bet_cv > 0.5:
                    counter_probability += 0.1
                elif bet_cv > 0.3:
                    counter_probability += 0.05
        except Exception as e:
            print(f"Error in betting pattern analysis: {e}")
        
        # Determine bet category for prediction (0=low, 1=medium, 2=high)
        try:
            bet_size = hand_data['bet_size'].values[0]
            if bet_size < 50:
                bet_category = 0
            elif bet_size < 100:
                bet_category = 1
            else:
                bet_category = 2
                
            # Predict next bet using Markov chain
            predicted_next_bet = self.predict_next_bet(bet_category)
        except Exception as e:
            print(f"Error determining bet category or predicting next bet: {e}")
            predicted_next_bet = 1  # Default to medium
        
        # Flag as suspected counter based on combined probability
        suspected = counter_probability > self.similarity_threshold
        
        # Store in history
        self.history.append({
            'hand_number': len(self.history) + 1,
            'similarity_score': float(max_similarity[0]),
            'mahalanobis_distance': float(mahalanobis_distance),
            'counter_probability': float(counter_probability),
            'kalman_state': kalman_state.tolist(),
            'suspected': bool(suspected)
        })
        
        result = {
            'similarity_score': float(max_similarity[0]),
            'mahalanobis_distance': float(mahalanobis_distance),
            'counter_probability': float(counter_probability),
            'kalman_state': kalman_state.tolist(),
            'predicted_next_bet_category': int(predicted_next_bet),
            'suspected': bool(suspected),
            'history': self.history
        }
        
        print(f"Detection result: counter_probability={result['counter_probability']}, suspected={result['suspected']}")
        return result
    
    def get_correlation_analysis(self):
        """Get correlation analysis of player behavior"""
        if len(self.player_data) < 3:
            return {"error": "Not enough data for correlation analysis"}
            
        try:
            # Calculate correlation matrix
            corr_matrix = self.player_data.corr().round(3)
            
            # Handle NaN values in correlation matrix
            corr_matrix = corr_matrix.fillna(0)
            
            # Convert to dictionary for JSON serialization
            corr_dict = {}
            for col in corr_matrix.columns:
                corr_dict[col] = {}
                for idx in corr_matrix.index:
                    corr_dict[col][idx] = corr_matrix.loc[idx, col]
                    
            # Calculate eigenvalues and eigenvectors
            eigenvalues = []
            eigenvectors = []
            
            if self.correlation_matrix is not None:
                try:
                    # Check for NaN or inf values
                    if not np.isnan(self.correlation_matrix).any() and not np.isinf(self.correlation_matrix).any():
                        eigenvalues, eigenvectors = np.linalg.eig(self.correlation_matrix)
                        
                        # Sort by eigenvalue magnitude
                        idx = eigenvalues.argsort()[::-1]
                        eigenvalues = eigenvalues[idx].real.tolist()
                        eigenvectors = eigenvectors[:, idx].real.tolist()
                except Exception as e:
                    print(f"Error calculating eigenvalues: {e}")
            
            # Safely convert Cholesky factor to list
            cholesky_list = None
            if self.cholesky_factor is not None:
                try:
                    cholesky_list = self.cholesky_factor.tolist()
                except Exception as e:
                    print(f"Error converting Cholesky factor to list: {e}")
                    
            return {
                "correlation_matrix": corr_dict,
                "eigenvalues": eigenvalues,
                "eigenvectors": eigenvectors,
                "cholesky_factor": cholesky_list
            }
            
        except Exception as e:
            print(f"Error in correlation analysis: {e}")
            return {
                "error": f"Error in correlation analysis: {str(e)}",
                "correlation_matrix": {},
                "eigenvalues": [],
                "eigenvectors": [],
                "cholesky_factor": None
            }
    
    def reset(self):
        """Reset the detector history"""
        print("Resetting detector history")
        self.history = []
        self.player_data = pd.DataFrame()
        self.correlation_matrix = None
        self.cholesky_factor = None
        self.kalman_state = np.zeros(3)
        self.kalman_covariance = np.eye(3)
        return {'status': 'reset successful'}

# Blackjack game logic
class BlackjackGame:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.running_count = 0
        self.hands_played = 0
        self.true_count = 0
        self.deck_penetration = 0.0
        self.player_win_streak = 0
        self.dealer_win_streak = 0
        self.player_wins = 0
        self.dealer_wins = 0
        self.pushes = 0
        self.player_blackjacks = 0
        self.dealer_blackjacks = 0
        self.player_busts = 0
        self.dealer_busts = 0
        self.bet_history = []
        self.count_history = []
        self.win_history = []
        
        # Card values for Hi-Lo counting system
        self.card_count_values = {
            '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,  # Low cards (count +1)
            '7': 0, '8': 0, '9': 0,                  # Neutral cards (count 0)
            '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards (count -1)
        }
        
        self.reset_game()
        print("Enhanced BlackjackGame initialized with additional tracking metrics")
        
    def reset_game(self):
        """Reset the game state"""
        print("Resetting game state")
        self.create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.hands_played += 1
        # Calculate true count (running count / decks remaining)
        decks_remaining = len(self.deck) / 52
        self.true_count = self.running_count / max(1, decks_remaining)
        # Calculate deck penetration (percentage of cards dealt)
        self.deck_penetration = 1.0 - (len(self.deck) / 104.0)  # 2 decks = 104 cards
        print(f"Game reset complete. Hands played: {self.hands_played}, True count: {self.true_count:.2f}, Deck penetration: {self.deck_penetration:.2f}")
        return {'status': 'game reset'}
    
    def create_deck(self):
        """Create a new deck of cards (2 decks)"""
        print("Creating new deck (2 decks)")
        suits = ['♥', '♦', '♣', '♠']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        # Create 2 decks
        self.deck = []
        for _ in range(2):
            for suit in suits:
                for value in values:
                    self.deck.append({'value': value, 'suit': suit})
        
        # Shuffle the deck
        random.shuffle(self.deck)
        print(f"Deck created and shuffled. Total cards: {len(self.deck)}")
        
    def deal_initial_cards(self):
        """Deal the initial cards for player and dealer"""
        print("Dealing initial cards")
        if len(self.deck) < 4:
            print("Not enough cards, creating new deck")
            self.create_deck()
            self.running_count = 0
            
        self.player_hand = [self.deck.pop(), self.deck.pop()]
        self.dealer_hand = [self.deck.pop(), self.deck.pop()]
        
        # Update running count
        for card in self.player_hand:
            self.running_count += self.card_count_values[card['value']]
        
        # Only one dealer card is visible
        self.running_count += self.card_count_values[self.dealer_hand[0]['value']]
        
        # Calculate true count (running count / decks remaining)
        decks_remaining = len(self.deck) / 52
        self.true_count = self.running_count / max(1, decks_remaining)
        
        # Calculate deck penetration (percentage of cards dealt)
        self.deck_penetration = 1.0 - (len(self.deck) / 104.0)  # 2 decks = 104 cards
        
        # Add to count history
        self.count_history.append({
            'hand_number': self.hands_played,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration
        })
        
        print(f"Initial deal complete. Player hand: {self.player_hand}, Dealer upcard: {self.dealer_hand[0]}")
        print(f"Running count: {self.running_count}, True count: {self.true_count:.2f}, Deck penetration: {self.deck_penetration:.2f}")
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': [self.dealer_hand[0], {'value': '?', 'suit': '?'}],
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration
        }
    
    def hit(self):
        """Player takes another card"""
        print("Player hits")
        if len(self.deck) < 1:
            print("Not enough cards, creating new deck")
            self.create_deck()
            self.running_count = 0
            
        card = self.deck.pop()
        self.player_hand.append(card)
        print(f"Player drew: {card}")
        
        # Update running count
        self.running_count += self.card_count_values[card['value']]
        
        # Calculate true count (running count / decks remaining)
        decks_remaining = len(self.deck) / 52
        self.true_count = self.running_count / max(1, decks_remaining)
        
        # Calculate deck penetration (percentage of cards dealt)
        self.deck_penetration = 1.0 - (len(self.deck) / 104.0)  # 2 decks = 104 cards
        
        print(f"Running count updated to: {self.running_count}, True count: {self.true_count:.2f}, Deck penetration: {self.deck_penetration:.2f}")
        
        game_status = self.check_game_status()
        
        if game_status['game_over']:
            print(f"Game over after hit: {game_status['message']}")
            self.update_game_statistics(game_status)
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': [self.dealer_hand[0], {'value': '?', 'suit': '?'}] if not game_status['game_over'] else self.dealer_hand,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration,
            'game_status': game_status
        }
    
    def stand(self):
        """Player stands, dealer plays"""
        print("Player stands, dealer plays")
        # Dealer hits until 17 or higher
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        print(f"Dealer initial hand value: {dealer_value}")
        
        while dealer_value < 17:
            if len(self.deck) < 1:
                print("Not enough cards, creating new deck")
                self.create_deck()
                self.running_count = 0
                
            card = self.deck.pop()
            self.dealer_hand.append(card)
            print(f"Dealer drew: {card}")
            
            # Update running count
            self.running_count += self.card_count_values[card['value']]
            
            dealer_value = self.calculate_hand_value(self.dealer_hand)
            print(f"Dealer hand value now: {dealer_value}")
        
        # Calculate true count (running count / decks remaining)
        decks_remaining = len(self.deck) / 52
        self.true_count = self.running_count / max(1, decks_remaining)
        
        # Calculate deck penetration (percentage of cards dealt)
        self.deck_penetration = 1.0 - (len(self.deck) / 104.0)  # 2 decks = 104 cards
        
        print(f"Running count updated to: {self.running_count}, True count: {self.true_count:.2f}, Deck penetration: {self.deck_penetration:.2f}")
        
        game_status = self.check_game_status()
        print(f"Game over after stand: {game_status['message']}")
        
        self.update_game_statistics(game_status)
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': self.dealer_hand,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration,
            'game_status': game_status
        }
    
    def double_down(self):
        """Player doubles down (double bet, one more card, then stand)"""
        print("Player doubles down")
        if len(self.deck) < 1:
            print("Not enough cards, creating new deck")
            self.create_deck()
            self.running_count = 0
            
        card = self.deck.pop()
        self.player_hand.append(card)
        print(f"Player drew: {card}")
        
        # Update running count
        self.running_count += self.card_count_values[card['value']]
        
        # Calculate true count (running count / decks remaining)
        decks_remaining = len(self.deck) / 52
        self.true_count = self.running_count / max(1, decks_remaining)
        
        # Calculate deck penetration (percentage of cards dealt)
        self.deck_penetration = 1.0 - (len(self.deck) / 104.0)  # 2 decks = 104 cards
        
        print(f"Running count updated to: {self.running_count}, True count: {self.true_count:.2f}, Deck penetration: {self.deck_penetration:.2f}")
        
        # Now dealer plays (same as stand)
        return self.stand()
    
    def calculate_hand_value(self, hand):
        """Calculate the value of a hand, accounting for Aces"""
        value = 0
        aces = 0
        
        for card in hand:
            if card['value'] in ['J', 'Q', 'K']:
                value += 10
            elif card['value'] == 'A':
                aces += 1
                value += 11
            else:
                value += int(card['value'])
        
        # Adjust for Aces if needed
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
            
        return value
    
    def check_game_status(self):
        """Check if the game is over and determine the winner"""
        player_value = self.calculate_hand_value(self.player_hand)
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        
        game_over = False
        winner = None
        message = ""
        
        # Check for blackjack (21 with 2 cards)
        player_blackjack = player_value == 21 and len(self.player_hand) == 2
        dealer_blackjack = dealer_value == 21 and len(self.dealer_hand) == 2
        
        if player_blackjack or dealer_blackjack:
            game_over = True
            if player_blackjack and dealer_blackjack:
                winner = "push"
                message = "Both have Blackjack! Push."
            elif player_blackjack:
                winner = "player"
                message = "Player has Blackjack! Player wins!"
            else:
                winner = "dealer"
                message = "Dealer has Blackjack! Dealer wins!"
        
        # Check for bust (over 21)
        elif player_value > 21:
            game_over = True
            winner = "dealer"
            message = "Player busts! Dealer wins."
        elif dealer_value > 21:
            game_over = True
            winner = "player"
            message = "Dealer busts! Player wins."
        
        # If dealer is done (stand), compare hands
        elif len(self.player_hand) > 2 or dealer_value >= 17:
            game_over = True
            if player_value > dealer_value:
                winner = "player"
                message = "Player wins!"
            elif dealer_value > player_value:
                winner = "dealer"
                message = "Dealer wins!"
            else:
                winner = "push"
                message = "Push (tie)!"
        
        if game_over:
            print(f"Game over: {message}")
        
        return {
            'game_over': game_over,
            'winner': winner,
            'message': message,
            'player_value': player_value,
            'dealer_value': dealer_value,
            'player_blackjack': player_blackjack if 'player_blackjack' in locals() else False,
            'dealer_blackjack': dealer_blackjack if 'dealer_blackjack' in locals() else False
        }
    
    def update_game_statistics(self, game_status):
        """Update game statistics based on game outcome"""
        if not game_status['game_over']:
            return
            
        winner = game_status['winner']
        
        # Update win/loss statistics
        if winner == "player":
            self.player_wins += 1
            self.player_win_streak += 1
            self.dealer_win_streak = 0
            
            # Check for player blackjack
            if game_status.get('player_blackjack', False):
                self.player_blackjacks += 1
                
            # Check for dealer bust
            if game_status['dealer_value'] > 21:
                self.dealer_busts += 1
                
        elif winner == "dealer":
            self.dealer_wins += 1
            self.dealer_win_streak += 1
            self.player_win_streak = 0
            
            # Check for dealer blackjack
            if game_status.get('dealer_blackjack', False):
                self.dealer_blackjacks += 1
                
            # Check for player bust
            if game_status['player_value'] > 21:
                self.player_busts += 1
                
        else:  # push
            self.pushes += 1
            self.player_win_streak = 0
            self.dealer_win_streak = 0
            
        # Add to win history
        self.win_history.append({
            'hand_number': self.hands_played,
            'winner': winner,
            'player_value': game_status['player_value'],
            'dealer_value': game_status['dealer_value'],
            'running_count': self.running_count,
            'true_count': self.true_count
        })
        
        print(f"Game statistics updated: Player wins: {self.player_wins}, Dealer wins: {self.dealer_wins}, Pushes: {self.pushes}")
    
    def get_hand_data(self, bet_size):
        """Get the current hand data for card counting detection"""
        player_value = self.calculate_hand_value(self.player_hand)
        dealer_upcard_value = 0
        
        # Convert dealer upcard to numeric value
        if self.dealer_hand[0]['value'] in ['J', 'Q', 'K']:
            dealer_upcard_value = 10
        elif self.dealer_hand[0]['value'] == 'A':
            dealer_upcard_value = 11
        else:
            dealer_upcard_value = int(self.dealer_hand[0]['value'])
        
        # Determine action (0=Stand, 1=Hit, 2=Double Down)
        if len(self.player_hand) > 2:
            action = 1  # Hit
        else:
            action = 0  # Stand (default)
        
        # Add bet to history
        self.bet_history.append({
            'hand_number': self.hands_played,
            'bet_size': bet_size,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration
        })
        
        print(f"Generating hand data for detection with bet size: {bet_size}")
        
        return {
            'dealer_upcard': dealer_upcard_value,
            'player_hand_value': player_value,
            'action': action,
            'bet_size': bet_size,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration,
            'player_win_streak': self.player_win_streak,
            'dealer_win_streak': self.dealer_win_streak,
            'hands_played': self.hands_played
        }
    
    def get_game_statistics(self):
        """Get current game statistics"""
        return {
            'hands_played': self.hands_played,
            'player_wins': self.player_wins,
            'dealer_wins': self.dealer_wins,
            'pushes': self.pushes,
            'player_blackjacks': self.player_blackjacks,
            'dealer_blackjacks': self.dealer_blackjacks,
            'player_busts': self.player_busts,
            'dealer_busts': self.dealer_busts,
            'player_win_streak': self.player_win_streak,
            'dealer_win_streak': self.dealer_win_streak,
            'running_count': self.running_count,
            'true_count': self.true_count,
            'deck_penetration': self.deck_penetration,
            'bet_history': self.bet_history,
            'win_history': self.win_history,
            'count_history': self.count_history
        }

# Initialize our classes
print("Creating game objects...")
blackjack_game = BlackjackGame()
detector = CardCounterDetector()

# Initialize the detector with more comprehensive sample data
print("Initializing detector with comprehensive sample data...")
initial_data = pd.DataFrame([
    # Regular player patterns (no card counting)
    {
        'dealer_upcard': 10, 'player_hand_value': 17, 'action': 0, 'bet_size': 50, 
        'running_count': 1, 'true_count': 0.5, 'deck_penetration': 0.1,
        'player_win_streak': 0, 'dealer_win_streak': 0, 'hands_played': 1
    },
    {
        'dealer_upcard': 5, 'player_hand_value': 16, 'action': 0, 'bet_size': 50, 
        'running_count': 0, 'true_count': 0.0, 'deck_penetration': 0.15,
        'player_win_streak': 0, 'dealer_win_streak': 1, 'hands_played': 2
    },
    {
        'dealer_upcard': 8, 'player_hand_value': 19, 'action': 0, 'bet_size': 75, 
        'running_count': 3, 'true_count': 1.5, 'deck_penetration': 0.2,
        'player_win_streak': 1, 'dealer_win_streak': 0, 'hands_played': 3
    },
    {
        'dealer_upcard': 6, 'player_hand_value': 12, 'action': 1, 'bet_size': 50, 
        'running_count': -2, 'true_count': -1.0, 'deck_penetration': 0.25,
        'player_win_streak': 0, 'dealer_win_streak': 1, 'hands_played': 4
    },
    {
        'dealer_upcard': 9, 'player_hand_value': 20, 'action': 0, 'bet_size': 50, 
        'running_count': -1, 'true_count': -0.5, 'deck_penetration': 0.3,
        'player_win_streak': 0, 'dealer_win_streak': 2, 'hands_played': 5
    },
    
    # Conservative card counter patterns
    {
        'dealer_upcard': 7, 'player_hand_value': 15, 'action': 0, 'bet_size': 25, 
        'running_count': -4, 'true_count': -2.0, 'deck_penetration': 0.35,
        'player_win_streak': 0, 'dealer_win_streak': 0, 'hands_played': 6
    },
    {
        'dealer_upcard': 4, 'player_hand_value': 14, 'action': 0, 'bet_size': 100, 
        'running_count': 6, 'true_count': 3.0, 'deck_penetration': 0.4,
        'player_win_streak': 1, 'dealer_win_streak': 0, 'hands_played': 7
    },
    {
        'dealer_upcard': 3, 'player_hand_value': 18, 'action': 0, 'bet_size': 125, 
        'running_count': 8, 'true_count': 4.0, 'deck_penetration': 0.45,
        'player_win_streak': 2, 'dealer_win_streak': 0, 'hands_played': 8
    },
    
    # Aggressive card counter patterns
    {
        'dealer_upcard': 2, 'player_hand_value': 11, 'action': 2, 'bet_size': 25, 
        'running_count': -6, 'true_count': -3.0, 'deck_penetration': 0.5,
        'player_win_streak': 0, 'dealer_win_streak': 1, 'hands_played': 9
    },
    {
        'dealer_upcard': 10, 'player_hand_value': 16, 'action': 0, 'bet_size': 200, 
        'running_count': 10, 'true_count': 5.0, 'deck_penetration': 0.55,
        'player_win_streak': 0, 'dealer_win_streak': 0, 'hands_played': 10
    },
    {
        'dealer_upcard': 9, 'player_hand_value': 13, 'action': 1, 'bet_size': 250, 
        'running_count': 12, 'true_count': 6.0, 'deck_penetration': 0.6,
        'player_win_streak': 1, 'dealer_win_streak': 0, 'hands_played': 11
    },
    
    # Expert card counter patterns with deep deck penetration
    {
        'dealer_upcard': 8, 'player_hand_value': 12, 'action': 1, 'bet_size': 25, 
        'running_count': -8, 'true_count': -4.0, 'deck_penetration': 0.65,
        'player_win_streak': 0, 'dealer_win_streak': 2, 'hands_played': 12
    },
    {
        'dealer_upcard': 7, 'player_hand_value': 17, 'action': 0, 'bet_size': 300, 
        'running_count': 14, 'true_count': 7.0, 'deck_penetration': 0.7,
        'player_win_streak': 0, 'dealer_win_streak': 0, 'hands_played': 13
    },
    {
        'dealer_upcard': 6, 'player_hand_value': 10, 'action': 2, 'bet_size': 350, 
        'running_count': 16, 'true_count': 8.0, 'deck_penetration': 0.75,
        'player_win_streak': 1, 'dealer_win_streak': 0, 'hands_played': 14
    },
    
    # Mixed patterns to improve robustness
    {
        'dealer_upcard': 5, 'player_hand_value': 19, 'action': 0, 'bet_size': 75, 
        'running_count': 2, 'true_count': 1.0, 'deck_penetration': 0.8,
        'player_win_streak': 2, 'dealer_win_streak': 0, 'hands_played': 15
    },
    {
        'dealer_upcard': 4, 'player_hand_value': 20, 'action': 0, 'bet_size': 100, 
        'running_count': 4, 'true_count': 2.0, 'deck_penetration': 0.85,
        'player_win_streak': 3, 'dealer_win_streak': 0, 'hands_played': 16
    }
])
detector.fit(initial_data)

@app.route('/api/new-game', methods=['POST'])
def new_game():
    """Start a new blackjack game"""
    print("API call: /api/new-game")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50)
    print(f"Starting new game with bet size: {bet_size}")
    
    # Reset the game state
    blackjack_game.reset_game()
    
    # Deal initial cards
    game_data = blackjack_game.deal_initial_cards()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    detection = detector.detect(hand_data)
    
    # Get game statistics
    statistics = blackjack_game.get_game_statistics()
    
    print(f"Returning new game data and detection results")
    return jsonify({
        'game_data': game_data,
        'detection': detection,
        'statistics': statistics
    })

@app.route('/api/hit', methods=['POST'])
def hit():
    """Player hits"""
    print("API call: /api/hit")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50)
    
    # Player takes another card
    game_data = blackjack_game.hit()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    detection = detector.detect(hand_data)
    
    # Get game statistics
    statistics = blackjack_game.get_game_statistics()
    
    print(f"Returning hit data and detection results")
    return jsonify({
        'game_data': game_data,
        'detection': detection,
        'statistics': statistics
    })

@app.route('/api/stand', methods=['POST'])
def stand():
    """Player stands"""
    print("API call: /api/stand")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50)
    
    # Player stands
    game_data = blackjack_game.stand()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    detection = detector.detect(hand_data)
    
    # Get game statistics
    statistics = blackjack_game.get_game_statistics()
    
    print(f"Returning stand data and detection results")
    return jsonify({
        'game_data': game_data,
        'detection': detection,
        'statistics': statistics
    })

@app.route('/api/double-down', methods=['POST'])
def double_down():
    """Player doubles down"""
    print("API call: /api/double-down")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50) * 2  # Double the bet
    
    # Player doubles down
    game_data = blackjack_game.double_down()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    detection = detector.detect(hand_data)
    
    # Get game statistics
    statistics = blackjack_game.get_game_statistics()
    
    print(f"Returning double-down data and detection results")
    return jsonify({
        'game_data': game_data,
        'detection': detection,
        'statistics': statistics
    })

@app.route('/api/reset-detector', methods=['POST'])
def reset_detector():
    """Reset the detector history"""
    print("API call: /api/reset-detector")
    detector.reset()
    return jsonify({'status': 'reset successful'})

@app.route('/api/correlation-analysis', methods=['GET'])
def correlation_analysis():
    """Get correlation analysis of player behavior"""
    print("API call: /api/correlation-analysis")
    analysis = detector.get_correlation_analysis()
    return jsonify(analysis)

@app.route('/api/game-statistics', methods=['GET'])
def game_statistics():
    """Get current game statistics"""
    print("API call: /api/game-statistics")
    statistics = blackjack_game.get_game_statistics()
    return jsonify(statistics)

@app.route('/api/status', methods=['GET'])
def status():
    """Get the status of the API"""
    print("API call: /api/status")
    return jsonify({
        'status': 'ok',
        'message': 'Card Counter Detector API is running'
    })

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True) 