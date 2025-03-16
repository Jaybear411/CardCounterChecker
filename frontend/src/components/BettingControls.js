import React from 'react';
import './BettingControls.css';

const BettingControls = ({ betSize, setBetSize, bankroll, gameOver }) => {
  // Predefined bet sizes
  const betSizes = [10, 25, 50, 100, 500];

  // Increment bet by 10
  const increaseBet = () => {
    if (betSize + 10 <= bankroll) {
      setBetSize(betSize + 10);
    }
  };

  // Decrement bet by 10
  const decreaseBet = () => {
    if (betSize - 10 >= 10) {
      setBetSize(betSize - 10);
    }
  };

  // Set bet to specific amount
  const setBet = (amount) => {
    if (amount <= bankroll) {
      setBetSize(amount);
    }
  };

  return (
    <div className="betting-controls">
      <h3>Bet Size: ${betSize}</h3>
      
      <div className="bet-amount-controls">
        <button 
          onClick={decreaseBet} 
          disabled={betSize <= 10 || !gameOver}
          className="bet-button"
        >
          -
        </button>
        <span>${betSize}</span>
        <button 
          onClick={increaseBet} 
          disabled={betSize >= bankroll || !gameOver}
          className="bet-button"
        >
          +
        </button>
      </div>
      
      <div className="bet-chips">
        {betSizes.map((amount) => (
          <button
            key={amount}
            onClick={() => setBet(amount)}
            disabled={amount > bankroll || !gameOver}
            className={`chip ${betSize === amount ? 'selected' : ''}`}
          >
            ${amount}
          </button>
        ))}
      </div>
    </div>
  );
};

export default BettingControls; 