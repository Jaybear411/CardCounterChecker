import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

class BlackjackDataGenerator:
    """
    Generates realistic Blackjack hand data for testing card counting detection algorithms.
    
    This class simulates both regular players and card counters with configurable parameters
    to create datasets that mimic real-world Blackjack play.
    """
    
    def __init__(self, random_seed=42):
        """
        Initialize the data generator
        
        Parameters:
        -----------
        random_seed : int
            Seed for random number generation (for reproducibility)
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Define card values for Hi-Lo counting system
        self.card_values = {
            2: 1, 3: 1, 4: 1, 5: 1, 6: 1,  # Low cards (count +1)
            7: 0, 8: 0, 9: 0,              # Neutral cards (count 0)
            10: -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1  # High cards (count -1)
        }
        
        # Map face cards to value 10 for hand value calculation
        self.card_play_values = {
            2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
            'J': 10, 'Q': 10, 'K': 10, 'A': 11  # Ace is initially 11, can be 1 if needed
        }
    
    def _draw_card(self, deck):
        """
        Draw a card from the deck
        
        Parameters:
        -----------
        deck : list
            Current deck of cards
            
        Returns:
        --------
        card : str or int
            The drawn card
        """
        if len(deck) == 0:
            raise ValueError("Deck is empty")
        
        card_index = np.random.randint(0, len(deck))
        card = deck.pop(card_index)
        return card
    
    def _create_deck(self, num_decks=6):
        """
        Create a shoe with multiple decks
        
        Parameters:
        -----------
        num_decks : int
            Number of decks to use
            
        Returns:
        --------
        deck : list
            A list representing the shoe with multiple decks
        """
        one_deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'J', 'Q', 'K', 'A'] * 4
        return one_deck * num_decks
    
    def _calculate_hand_value(self, hand):
        """
        Calculate the value of a blackjack hand
        
        Parameters:
        -----------
        hand : list
            List of cards in the hand
            
        Returns:
        --------
        value : int
            Value of the hand
        """
        value = sum(self.card_play_values[card] for card in hand)
        
        # Adjust for aces if needed
        num_aces = hand.count('A')
        while value > 21 and num_aces > 0:
            value -= 10  # Change an Ace from 11 to 1
            num_aces -= 1
            
        return value
    
    def _decide_action(self, player_hand_value, dealer_upcard, running_count, 
                      is_card_counter=False):
        """
        Decide the action to take based on hand value and dealer upcard
        
        Parameters:
        -----------
        player_hand_value : int
            Value of the player's hand
        dealer_upcard : int or str
            Dealer's upcard
        running_count : int
            Current running count
        is_card_counter : bool
            Whether this is a card counter or regular player
            
        Returns:
        --------
        action : int
            0 = Stand, 1 = Hit, 2 = Double Down
        """
        # Convert dealer face card to value
        if dealer_upcard in ['J', 'Q', 'K']:
            dealer_upcard_value = 10
        elif dealer_upcard == 'A':
            dealer_upcard_value = 11
        else:
            dealer_upcard_value = dealer_upcard
        
        # Basic strategy (simplified)
        if player_hand_value >= 17:
            return 0  # Stand
        elif player_hand_value <= 8:
            return 1  # Hit
        elif player_hand_value == 11:
            return 2  # Double down on 11
        elif player_hand_value == 10 and dealer_upcard_value < 10:
            return 2  # Double down on 10 vs dealer <10
        elif player_hand_value == 9 and 3 <= dealer_upcard_value <= 6:
            return 2  # Double down on 9 vs dealer 3-6
        elif 12 <= player_hand_value <= 16 and 2 <= dealer_upcard_value <= 6:
            return 0  # Stand on 12-16 vs dealer 2-6
        else:
            return 1  # Hit in all other cases
            
        # Card counter modifications
        if is_card_counter:
            # If count is very positive (lots of high cards left), more aggressive
            if running_count >= 3:
                if player_hand_value == 16 and dealer_upcard_value == 10:
                    return 0  # Stand instead of hit with high count
                elif player_hand_value == 10 and dealer_upcard_value == 10:
                    return 2  # Double down instead of hit with high count
                elif player_hand_value == 9 and dealer_upcard_value == 2:
                    return 2  # Double down instead of hit with high count
            
            # If count is very negative (lots of low cards left), more conservative
            if running_count <= -3:
                if player_hand_value == 12 and dealer_upcard_value == 3:
                    return 1  # Hit instead of stand with low count
                elif player_hand_value == 16 and dealer_upcard_value == 6:
                    return 1  # Hit instead of stand with low count
    
    def _decide_bet(self, running_count, true_count, is_card_counter=False, 
                   min_bet=25, max_bet=500):
        """
        Decide bet size based on count (for card counters) or randomly (for regular players)
        
        Parameters:
        -----------
        running_count : int
            Current running count
        true_count : float
            True count (running count divided by decks remaining)
        is_card_counter : bool
            Whether this is a card counter
        min_bet : int
            Minimum bet size
        max_bet : int
            Maximum bet size
            
        Returns:
        --------
        bet_size : int
            The bet size
        """
        if is_card_counter:
            # Card counters increase bets when the count is favorable
            if true_count <= 0:
                # Minimum bet when count is 0 or negative
                bet_size = min_bet
            else:
                # Increase bet proportional to the true count
                bet_size = min(max_bet, int(min_bet * (1 + true_count)))
                
            # Add some noise to the bet size (don't be too obvious)
            noise = np.random.randint(-10, 11)
            bet_size = max(min_bet, min(max_bet, bet_size + noise))
        else:
            # Regular players bet somewhat randomly, but may have patterns unrelated to count
            # Some might increase bets after wins or decrease after losses
            base_bet = np.random.randint(min_bet, min_bet * 3)
            bet_size = min(max_bet, base_bet)
            
        return bet_size
    
    def generate_data(self, n_players=20, n_hands_per_player=50, counter_ratio=0.2,
                     num_decks=6, min_bet=25, max_bet=500):
        """
        Generate a realistic Blackjack dataset
        
        Parameters:
        -----------
        n_players : int
            Number of players to simulate
        n_hands_per_player : int
            Number of hands per player
        counter_ratio : float
            Ratio of card counters to regular players
        num_decks : int
            Number of decks in the shoe
        min_bet : int
            Minimum bet size
        max_bet : int
            Maximum bet size
            
        Returns:
        --------
        data : pandas.DataFrame
            DataFrame containing the simulated Blackjack hands
        """
        data = []
        
        # Determine number of card counters
        n_counters = int(n_players * counter_ratio)
        n_regular = n_players - n_counters
        
        # Create player IDs
        player_ids = list(range(1, n_players + 1))
        counter_ids = player_ids[:n_counters]
        regular_ids = player_ids[n_counters:]
        
        # Simulate games for each player
        for player_id in player_ids:
            is_card_counter = player_id in counter_ids
            
            # Initialize a new shoe for this player
            deck = self._create_deck(num_decks=num_decks)
            np.random.shuffle(deck)
            
            # Initialize running count
            running_count = 0
            
            # Track position in the deck to calculate true count
            initial_deck_size = len(deck)
            
            for hand_num in range(n_hands_per_player):
                # Calculate true count
                decks_remaining = max(1, len(deck) / 52)
                true_count = running_count / decks_remaining
                
                # Decide bet size
                bet_size = self._decide_bet(
                    running_count, 
                    true_count, 
                    is_card_counter,
                    min_bet, 
                    max_bet
                )
                
                # Draw initial cards
                player_hand = [self._draw_card(deck), self._draw_card(deck)]
                dealer_hand = [self._draw_card(deck), self._draw_card(deck)]
                
                # Update running count based on visible cards
                for card in player_hand:
                    running_count += self.card_values[card]
                
                # Only one dealer card is visible
                dealer_upcard = dealer_hand[0]
                running_count += self.card_values[dealer_upcard]
                
                # Calculate hand values
                player_hand_value = self._calculate_hand_value(player_hand)
                
                # Decide action
                action = self._decide_action(
                    player_hand_value, 
                    dealer_upcard, 
                    running_count, 
                    is_card_counter
                )
                
                # Record the hand
                data.append({
                    'player_id': player_id,
                    'is_counter': is_card_counter,
                    'hand_num': hand_num + 1,
                    'dealer_upcard': dealer_upcard,
                    'player_hand_value': player_hand_value,
                    'action': action,
                    'bet_size': bet_size,
                    'running_count': running_count,
                    'true_count': true_count,
                    'decks_remaining': decks_remaining
                })
                
                # If the shoe is getting low, reshuffle
                if len(deck) < initial_deck_size * 0.25:
                    deck = self._create_deck(num_decks=num_decks)
                    np.random.shuffle(deck)
                    running_count = 0
                    initial_deck_size = len(deck)
        
        return pd.DataFrame(data)
    
    def visualize_data(self, df):
        """
        Visualize the generated data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            The generated data
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Bet size vs true count, colored by player type
        plt.subplot(2, 2, 1)
        sns.scatterplot(
            data=df, 
            x='true_count', 
            y='bet_size', 
            hue='is_counter', 
            alpha=0.5,
            palette=['blue', 'red']
        )
        plt.title('Bet Size vs True Count')
        plt.xlabel('True Count')
        plt.ylabel('Bet Size')
        
        # Plot 2: Distribution of bet sizes by player type
        plt.subplot(2, 2, 2)
        sns.boxplot(
            data=df, 
            x='is_counter', 
            y='bet_size'
        )
        plt.title('Bet Size Distribution by Player Type')
        plt.xlabel('Is Card Counter')
        plt.ylabel('Bet Size')
        
        # Plot 3: Action distribution by player type
        plt.subplot(2, 2, 3)
        action_counts = df.groupby(['is_counter', 'action']).size().unstack()
        action_counts.plot(kind='bar', ax=plt.gca())
        plt.title('Action Distribution by Player Type')
        plt.xlabel('Is Card Counter')
        plt.ylabel('Count')
        plt.legend(['Stand', 'Hit', 'Double Down'])
        
        # Plot 4: Bet correlation with count
        plt.subplot(2, 2, 4)
        
        # Calculate correlation for each player
        correlations = []
        for player_id in df['player_id'].unique():
            player_data = df[df['player_id'] == player_id]
            corr = player_data['bet_size'].corr(player_data['true_count'])
            is_counter = player_data['is_counter'].iloc[0]
            correlations.append({
                'player_id': player_id,
                'correlation': corr,
                'is_counter': is_counter
            })
        
        corr_df = pd.DataFrame(correlations)
        sns.boxplot(
            data=corr_df, 
            x='is_counter', 
            y='correlation'
        )
        plt.title('Bet-Count Correlation by Player Type')
        plt.xlabel('Is Card Counter')
        plt.ylabel('Correlation Coefficient')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate realistic Blackjack data')
    parser.add_argument('--n_players', type=int, default=20, help='Number of players')
    parser.add_argument('--n_hands', type=int, default=50, help='Number of hands per player')
    parser.add_argument('--counter_ratio', type=float, default=0.2, help='Ratio of card counters')
    parser.add_argument('--output', type=str, default='blackjack_data.csv', help='Output CSV file')
    parser.add_argument('--visualize', action='store_true', help='Visualize the generated data')
    
    args = parser.parse_args()
    
    # Create generator
    generator = BlackjackDataGenerator()
    
    # Generate data
    print(f"Generating data for {args.n_players} players ({int(args.n_players * args.counter_ratio)} card counters)")
    print(f"Each player will play {args.n_hands} hands")
    
    df = generator.generate_data(
        n_players=args.n_players,
        n_hands_per_player=args.n_hands,
        counter_ratio=args.counter_ratio
    )
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Data saved to {args.output}")
    
    # Print summary statistics
    print("\nData Summary:")
    print(f"Total hands: {len(df)}")
    print(f"Players: {df['player_id'].nunique()}")
    print(f"Card counters: {df[df['is_counter']]['player_id'].nunique()}")
    print(f"Regular players: {df[~df['is_counter']]['player_id'].nunique()}")
    
    # Visualize if requested
    if args.visualize:
        generator.visualize_data(df) 