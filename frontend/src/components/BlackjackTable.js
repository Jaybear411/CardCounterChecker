import React from 'react';
import './BlackjackTable.css';

const BlackjackTable = ({ playerHand, dealerHand, gameOver, message }) => {
  // Function to render a playing card
  const renderCard = (card) => {
    const { value, suit } = card;
    
    // Determine card color based on suit
    const cardColor = suit === '♥' || suit === '♦' ? 'red' : 'black';
    
    // For face-down card
    if (value === '?') {
      return (
        <div className="card face-down">
          <div className="card-back"></div>
        </div>
      );
    }
    
    return (
      <div className={`card ${cardColor}`}>
        <div className="card-value-top">{value}</div>
        <div className="card-suit">{suit}</div>
        <div className="card-value-bottom">{value}</div>
      </div>
    );
  };

  // Calculate hand value
  const calculateHandValue = (hand) => {
    if (!hand) return 0;
    
    let value = 0;
    let aces = 0;
    
    hand.forEach(card => {
      if (card.value === '?') return;
      
      if (['J', 'Q', 'K'].includes(card.value)) {
        value += 10;
      } else if (card.value === 'A') {
        value += 11;
        aces += 1;
      } else {
        value += parseInt(card.value);
      }
    });
    
    // Adjust for aces if needed
    while (value > 21 && aces > 0) {
      value -= 10;
      aces -= 1;
    }
    
    return value;
  };

  const playerValue = calculateHandValue(playerHand);
  const dealerValue = calculateHandValue(dealerHand);
  
  return (
    <div className="blackjack-table">
      <div className="table-felt">
        <div className="dealer-area">
          <h2>Dealer {gameOver && <span>({dealerValue})</span>}</h2>
          <div className="hand dealer-hand">
            {dealerHand && dealerHand.map((card, index) => (
              <div key={index} className="card-container">
                {renderCard(card)}
              </div>
            ))}
          </div>
        </div>
        
        <div className="player-area">
          <h2>Player ({playerValue})</h2>
          <div className="hand player-hand">
            {playerHand && playerHand.map((card, index) => (
              <div key={index} className="card-container">
                {renderCard(card)}
              </div>
            ))}
          </div>
        </div>
        
        {message && (
          <div className="game-message">
            <p>{message}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BlackjackTable; 