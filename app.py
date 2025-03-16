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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import random
import json

print("Starting Flask Card Counter Detector app...")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
print("Flask app initialized with CORS")

# Card counting detection system
class CardCounterDetector:
    def __init__(self, n_components=2, similarity_threshold=0.70):
        self.n_components = n_components
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()
        self.svd = TruncatedSVD(n_components=n_components)
        self.known_card_counters = np.array([
            [0.8, 0.7],   # Conservative counter
            [0.9, 0.85],  # Aggressive counter
            [0.7, 0.9]    # Expert counter
        ])
        # Store history for visualization
        self.history = []
        print("CardCounterDetector initialized")
        
    def fit(self, X):
        """Fit the detector on initial data"""
        print("Fitting detector on initial data")
        X_scaled = self.scaler.fit_transform(X)
        self.svd.fit(X_scaled)
        print("Detector fitted successfully")
        return self
        
    def detect(self, hand_data):
        """Detect if a hand exhibits card counting behavior"""
        print(f"Detecting card counting for hand data: {hand_data}")
        # Convert to DataFrame if it's a dict
        if isinstance(hand_data, dict):
            hand_data = pd.DataFrame([hand_data])
            
        # Scale the data
        X_scaled = self.scaler.transform(hand_data)
        
        # Apply SVD
        X_svd = self.svd.transform(X_scaled)
        
        # Compute similarity to known card counters
        similarities = cosine_similarity(X_svd, self.known_card_counters)
        max_similarity = np.max(similarities, axis=1)
        
        # Flag as suspected counter based on threshold
        suspected = max_similarity > self.similarity_threshold
        
        # Store in history
        self.history.append({
            'hand_number': len(self.history) + 1,
            'similarity_score': float(max_similarity[0]),
            'suspected': bool(suspected[0])
        })
        
        result = {
            'similarity_score': float(max_similarity[0]),
            'suspected': bool(suspected[0]),
            'history': self.history
        }
        print(f"Detection result: similarity_score={result['similarity_score']}, suspected={result['suspected']}")
        return result
    
    def reset(self):
        """Reset the detector history"""
        print("Resetting detector history")
        self.history = []
        return {'status': 'reset successful'}

# Blackjack game logic
class BlackjackGame:
    def __init__(self):
        self.deck = []
        self.player_hand = []
        self.dealer_hand = []
        self.running_count = 0
        self.hands_played = 0
        
        # Card values for Hi-Lo counting system
        self.card_count_values = {
            '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,  # Low cards (count +1)
            '7': 0, '8': 0, '9': 0,                  # Neutral cards (count 0)
            '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards (count -1)
        }
        
        self.reset_game()
        print("BlackjackGame initialized")
        
    def reset_game(self):
        """Reset the game state"""
        print("Resetting game state")
        self.create_deck()
        self.player_hand = []
        self.dealer_hand = []
        self.hands_played += 1
        print(f"Game reset complete. Hands played: {self.hands_played}")
        return {'status': 'game reset'}
    
    def create_deck(self):
        """Create a new deck of cards (6 decks)"""
        print("Creating new deck (6 decks)")
        suits = ['♥', '♦', '♣', '♠']
        values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        
        # Create 6 decks
        self.deck = []
        for _ in range(6):
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
        
        print(f"Initial deal complete. Player hand: {self.player_hand}, Dealer upcard: {self.dealer_hand[0]}")
        print(f"Running count: {self.running_count}")
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': [self.dealer_hand[0], {'value': '?', 'suit': '?'}],
            'running_count': self.running_count
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
        print(f"Running count updated to: {self.running_count}")
        
        game_status = self.check_game_status()
        
        if game_status['game_over']:
            print(f"Game over after hit: {game_status['message']}")
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': [self.dealer_hand[0], {'value': '?', 'suit': '?'}] if not game_status['game_over'] else self.dealer_hand,
            'running_count': self.running_count,
            'game_status': game_status
        }
    
    def stand(self):
        """Player stands, dealer plays"""
        print("Player stands, dealer plays")
        # Dealer hits until 17 or higher
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        print(f"Dealer initial hand value: {dealer_value}")
        
        while self.calculate_hand_value(self.dealer_hand) < 17:
            if len(self.deck) < 1:
                print("Not enough cards, creating new deck")
                self.create_deck()
                self.running_count = 0
                
            card = self.deck.pop()
            self.dealer_hand.append(card)
            print(f"Dealer drew: {card}")
            self.running_count += self.card_count_values[card['value']]
            print(f"Running count updated to: {self.running_count}")
            print(f"Dealer hand value now: {self.calculate_hand_value(self.dealer_hand)}")
        
        game_status = self.check_game_status()
        print(f"Game over after stand: {game_status['message']}")
        
        return {
            'player_hand': self.player_hand,
            'dealer_hand': self.dealer_hand,
            'running_count': self.running_count,
            'game_status': game_status
        }
    
    def double_down(self):
        """Player doubles their bet and gets one more card"""
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
        print(f"Running count updated to: {self.running_count}")
        
        # Player stands after doubling down
        print("Player automatically stands after double down")
        return self.stand()
    
    def calculate_hand_value(self, hand):
        """Calculate the value of a hand"""
        value = 0
        aces = 0
        
        for card in hand:
            if card['value'] in ['J', 'Q', 'K']:
                value += 10
            elif card['value'] == 'A':
                value += 11
                aces += 1
            else:
                value += int(card['value'])
        
        # Adjust for aces if needed
        while value > 21 and aces > 0:
            value -= 10
            aces -= 1
            
        return value
    
    def check_game_status(self):
        """Check the status of the game"""
        player_value = self.calculate_hand_value(self.player_hand)
        dealer_value = self.calculate_hand_value(self.dealer_hand)
        
        print(f"Checking game status. Player: {player_value}, Dealer: {dealer_value}")
        
        game_over = False
        winner = None
        message = ""
        
        # Check if player busts
        if player_value > 21:
            game_over = True
            winner = "dealer"
            message = "Player busts! Dealer wins."
        # Check if dealer busts
        elif dealer_value > 21:
            game_over = True
            winner = "player"
            message = "Dealer busts! Player wins."
        # Check if player has blackjack
        elif len(self.player_hand) == 2 and player_value == 21:
            game_over = True
            winner = "player"
            message = "Blackjack! Player wins."
        # Check if dealer has blackjack
        elif len(self.dealer_hand) == 2 and dealer_value == 21:
            game_over = True
            winner = "dealer"
            message = "Dealer has Blackjack! Dealer wins."
        # Compare hands if both players stand
        elif len(self.player_hand) >= 2 and player_value <= 21 and dealer_value >= 17:
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
            'dealer_value': dealer_value
        }
    
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
        
        print(f"Generating hand data for detection with bet size: {bet_size}")
        
        return {
            'dealer_upcard': dealer_upcard_value,
            'player_hand_value': player_value,
            'action': action,
            'bet_size': bet_size,
            'running_count': self.running_count
        }

# Initialize our classes
print("Creating game objects...")
blackjack_game = BlackjackGame()
detector = CardCounterDetector()

# Initialize the detector with some sample data
print("Initializing detector with sample data...")
initial_data = pd.DataFrame([
    {
        'dealer_upcard': 10, 'player_hand_value': 17, 'action': 0, 'bet_size': 50, 'running_count': 1
    },
    {
        'dealer_upcard': 5, 'player_hand_value': 16, 'action': 0, 'bet_size': 50, 'running_count': 0
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
    
    print(f"Returning new game data and detection results")
    return jsonify({
        'game_data': game_data,
        'detection': detection
    })

@app.route('/api/hit', methods=['POST'])
def hit():
    """Player hits"""
    print("API call: /api/hit")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50)
    print(f"Player hitting with bet size: {bet_size}")
    
    # Player hits
    game_data = blackjack_game.hit()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result if game is over
    detection = None
    if game_data['game_status']['game_over']:
        print("Game is over after hit, detecting card counting")
        detection = detector.detect(hand_data)
    else:
        print("Game continues after hit, no detection performed")
    
    print(f"Returning hit results")
    return jsonify({
        'game_data': game_data,
        'detection': detection
    })

@app.route('/api/stand', methods=['POST'])
def stand():
    """Player stands"""
    print("API call: /api/stand")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50)
    print(f"Player standing with bet size: {bet_size}")
    
    # Player stands
    game_data = blackjack_game.stand()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    print("Game is over after stand, detecting card counting")
    detection = detector.detect(hand_data)
    
    print(f"Returning stand results")
    return jsonify({
        'game_data': game_data,
        'detection': detection
    })

@app.route('/api/double-down', methods=['POST'])
def double_down():
    """Player doubles down"""
    print("API call: /api/double-down")
    data = request.json
    print(f"Request data: {data}")
    bet_size = data.get('bet_size', 50) * 2  # Double the bet
    print(f"Player doubling down with bet size: {bet_size}")
    
    # Player doubles down
    game_data = blackjack_game.double_down()
    
    # Get hand data for detection
    hand_data = blackjack_game.get_hand_data(bet_size)
    
    # Add detection result
    print("Game is over after double down, detecting card counting")
    detection = detector.detect(hand_data)
    
    print(f"Returning double down results")
    return jsonify({
        'game_data': game_data,
        'detection': detection
    })

@app.route('/api/reset-detector', methods=['POST'])
def reset_detector():
    """Reset the card counting detector"""
    print("API call: /api/reset-detector")
    detector.reset()
    return jsonify({'status': 'detector reset successful'})

@app.route('/api/status', methods=['GET'])
def status():
    """Check API status"""
    print("API call: /api/status")
    return jsonify({
        'status': 'running',
        'message': 'BlackjackCardCounter API is running'
    })

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True) 