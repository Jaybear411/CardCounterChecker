import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import BlackjackTable from './components/BlackjackTable';
import CounterDetection from './components/CounterDetection';
import BettingControls from './components/BettingControls';

// Set the base URL for API calls (helpful for debugging)
// Use this instead of relying on the proxy in package.json
const API_BASE_URL = 'http://127.0.0.1:5000';

// Helper function to make API calls with proper error handling
const apiCall = async (endpoint, method = 'get', data = null) => {
  console.log(`Making ${method.toUpperCase()} request to ${endpoint}`, data);
  
  try {
    const url = `${API_BASE_URL}${endpoint}`;
    const config = { 
      url,
      method, 
      data,
      headers: {
        'Content-Type': 'application/json'
      }
    };
    
    console.log('API request config:', config);
    const response = await axios(config);
    console.log(`Response from ${endpoint}:`, response.data);
    return { success: true, data: response.data };
  } catch (error) {
    console.error(`Error calling ${endpoint}:`, error);
    
    // Detailed error logging
    if (error.response) {
      // The server responded with a status code outside the 2xx range
      console.error('Response data:', error.response.data);
      console.error('Response status:', error.response.status);
      console.error('Response headers:', error.response.headers);
    } else if (error.request) {
      // The request was made but no response was received
      console.error('No response received:', error.request);
    } else {
      // Something happened in setting up the request that triggered an Error
      console.error('Request setup error:', error.message);
    }
    
    return { success: false, error };
  }
};

// Ping the server to check if it's available
const checkServerStatus = async () => {
  try {
    const result = await apiCall('/api/status');
    return result.success;
  } catch (error) {
    console.error('Server status check failed:', error);
    return false;
  }
};

function App() {
  const [gameData, setGameData] = useState(null);
  const [detection, setDetection] = useState(null);
  const [betSize, setBetSize] = useState(50);
  const [bankroll, setBankroll] = useState(1000);
  const [gameOver, setGameOver] = useState(false);
  const [message, setMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [serverError, setServerError] = useState(false);

  // Check server connection on component mount
  useEffect(() => {
    const checkConnection = async () => {
      const isServerAvailable = await checkServerStatus();
      if (!isServerAvailable) {
        setServerError(true);
        setMessage('Cannot connect to game server. Please ensure the backend is running.');
      } else {
        setServerError(false);
        startNewGame();
      }
    };
    
    checkConnection();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start a new game
  const startNewGame = async () => {
    try {
      setLoading(true);
      const result = await apiCall('/api/new-game', 'post', { bet_size: betSize });
      
      if (result.success) {
        setGameData(result.data.game_data);
        setDetection(result.data.detection);
        setGameOver(false);
        setMessage('');
        setServerError(false);
      } else {
        setMessage('Error starting new game. Please try again.');
      }
    } catch (error) {
      console.error('Error starting new game:', error);
      setMessage('Error starting new game. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Player hits
  const hit = async () => {
    if (gameOver || loading) return;
    
    try {
      setLoading(true);
      const result = await apiCall('/api/hit', 'post', { bet_size: betSize });
      
      if (result.success) {
        setGameData(result.data.game_data);
        
        if (result.data.game_data.game_status.game_over) {
          handleGameOver(result.data.game_data.game_status, result.data.detection);
        }
      } else {
        setMessage('Error hitting. Please try again.');
      }
    } catch (error) {
      console.error('Error hitting:', error);
      setMessage('Error hitting. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Player stands
  const stand = async () => {
    if (gameOver || loading) return;
    
    try {
      setLoading(true);
      const result = await apiCall('/api/stand', 'post', { bet_size: betSize });
      
      if (result.success) {
        setGameData(result.data.game_data);
        handleGameOver(result.data.game_data.game_status, result.data.detection);
      } else {
        setMessage('Error standing. Please try again.');
      }
    } catch (error) {
      console.error('Error standing:', error);
      setMessage('Error standing. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Player doubles down
  const doubleDown = async () => {
    if (gameOver || loading || bankroll < betSize * 2) return;
    
    try {
      setLoading(true);
      const result = await apiCall('/api/double-down', 'post', { bet_size: betSize });
      
      if (result.success) {
        setGameData(result.data.game_data);
        handleGameOver(result.data.game_data.game_status, result.data.detection);
      } else {
        setMessage('Error doubling down. Please try again.');
      }
    } catch (error) {
      console.error('Error doubling down:', error);
      setMessage('Error doubling down. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Reset detector
  const resetDetector = async () => {
    try {
      const result = await apiCall('/api/reset-detector', 'post');
      
      if (result.success) {
        setDetection(prev => ({...prev, history: []}));
        setMessage('Card counting detector reset.');
      } else {
        setMessage('Error resetting detector. Please try again.');
      }
    } catch (error) {
      console.error('Error resetting detector:', error);
      setMessage('Error resetting detector. Please try again.');
    }
  };

  // Handle game over
  const handleGameOver = (gameStatus, detectionData) => {
    setGameOver(true);
    setMessage(gameStatus.message);
    setDetection(detectionData);
    
    // Update bankroll based on game outcome
    if (gameStatus.winner === 'player') {
      if (gameStatus.message.includes('Blackjack')) {
        // Blackjack pays 3:2
        setBankroll(prevBankroll => prevBankroll + betSize * 1.5);
      } else {
        setBankroll(prevBankroll => prevBankroll + betSize);
      }
    } else if (gameStatus.winner === 'dealer') {
      setBankroll(prevBankroll => prevBankroll - betSize);
    }
    // Push (tie) - no change to bankroll
  };

  return (
    <div className="app">
      <header>
        <h1>Blackjack Card Counter Detector</h1>
        <div className="bankroll">
          <span>Bankroll: ${bankroll}</span>
          <button onClick={resetDetector} className="reset-button">
            Reset Detector
          </button>
        </div>
      </header>

      <main>
        {serverError ? (
          <div className="server-error">
            <h2>Server Connection Error</h2>
            <p>{message}</p>
            <p>
              Make sure the Flask backend is running on {API_BASE_URL} and that you've installed all requirements:
              <pre>pip install flask flask-cors</pre>
            </p>
            <button onClick={() => window.location.reload()}>Retry Connection</button>
          </div>
        ) : (
          <>
            <div className="game-container">
              {gameData ? (
                <BlackjackTable 
                  playerHand={gameData.player_hand}
                  dealerHand={gameData.dealer_hand}
                  gameOver={gameOver}
                  message={message}
                />
              ) : (
                <div className="loading">
                  {loading ? 'Loading game...' : 'Waiting for game data...'}
                </div>
              )}

              <div className="game-controls">
                <BettingControls 
                  betSize={betSize}
                  setBetSize={setBetSize}
                  bankroll={bankroll}
                  gameOver={gameOver}
                />
                
                <div className="action-buttons">
                  <button onClick={hit} disabled={gameOver || loading}>Hit</button>
                  <button onClick={stand} disabled={gameOver || loading}>Stand</button>
                  <button 
                    onClick={doubleDown} 
                    disabled={gameOver || loading || bankroll < betSize * 2 || (gameData && gameData.player_hand.length > 2)}
                  >
                    Double Down
                  </button>
                  {gameOver && (
                    <button onClick={startNewGame} className="new-game-button">
                      New Game
                    </button>
                  )}
                </div>
              </div>
            </div>

            <div className="detection-container">
              <CounterDetection 
                detection={detection} 
                runningCount={gameData ? gameData.running_count : 0}
              />
            </div>
          </>
        )}
      </main>

      <footer>
        <p>Created for CardCounterChecker</p>
      </footer>
    </div>
  );
}

export default App; 