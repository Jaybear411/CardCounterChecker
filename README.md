# Card Counter Checker

A system that uses advanced linear algebra techniques including Cholesky decomposition, correlation matrices, Kalman filtering, and Markov chains to detect card counting behavior in Blackjack players.

## Overview

Card counting in Blackjack gives players an advantage by tracking the ratio of high to low cards. Casinos want to detect such strategies by analyzing betting behavior and decision-making patterns.

This system:
1. Collects a dataset of Blackjack hands
2. Transforms the data into a numerical matrix using features like dealer upcard, player hand value, action taken, bet size, and running count
3. Applies multiple dimensionality reduction techniques (SVD and PCA) to extract latent behavioral patterns
4. Uses correlation matrices and Cholesky decomposition to analyze relationships between variables
5. Implements Kalman filtering for real-time tracking of player behavior
6. Uses Markov chains to predict future betting patterns
7. Calculates Mahalanobis distance to detect outliers in player behavior
8. Flags players with behavior similar to card counters

## Features

- **Data Simulation:** Creates synthetic datasets where each row represents a Blackjack hand
- **Normalization:** Scales features using StandardScaler for uniformity
- **Advanced Dimensionality Reduction:** Combines SVD and PCA to capture key variations in behavior
- **Correlation Analysis:** Uses correlation matrices to identify relationships between betting patterns and card counting indicators
- **Cholesky Decomposition:** Decomposes the correlation matrix for more efficient calculations and simulation
- **Kalman Filtering:** Provides real-time tracking and prediction of player behavior
- **Markov Chain Modeling:** Predicts future betting patterns based on current behavior
- **Mahalanobis Distance:** Detects outliers in player behavior accounting for feature correlations
- **Cosine Similarity:** Compares reduced representations to known card counter patterns
- **Multi-method Detection:** Combines multiple approaches for more robust detection
- **Visualization:** Creates intuitive plots to interpret detection results
- **Web Application:** Includes a web-based frontend to play Blackjack and see real-time card counting detection

## Modules

This repository contains several modules that work together or independently:

1. **Basic Version (`cardcounter.py`)**: A simple implementation that demonstrates the core concepts of using SVD and cosine similarity to detect card counting behavior.

2. **Advanced Version (`advanced_cardcounter.py`)**: A more comprehensive implementation that includes:
   - Object-oriented design with a `CardCounterDetector` class
   - Enhanced visualization of player behaviors in SVD space
   - Anomaly detection using Isolation Forest as an additional detection method
   - Player tracking across multiple hands
   - Real-time session analysis
   - Statistical comparisons between regular players and suspected card counters

3. **Pattern Recognition (`pattern_recognition.py`)**: Specialized in analyzing betting patterns:
   - Calculates correlation between bet size and running count
   - Measures adherence to Kelly criterion betting strategy
   - Assesses consistency of betting behavior
   - Uses clustering to group similar betting patterns
   - Provides detailed visualizations of betting behavior

4. **Combined Detector (`combined_detector.py`)**: Integrates the SVD approach with betting pattern analysis:
   - Uses multiple detection methods for improved accuracy
   - Reduces false positives and false negatives
   - Provides comprehensive performance metrics
   - Creates comparative visualizations between methods
   - Measures agreement between different detection approaches

5. **Web Application (`app.py` and `frontend/`)**: A Flask + React web app that:
   - Allows users to play Blackjack in the browser
   - Tracks player decisions and bet sizes
   - Analyzes behavior in real-time using advanced linear algebra techniques
   - Displays a graph showing how suspicious the play pattern appears
   - Provides a visual indicator of detection risk level
   - Includes correlation analysis and eigenvalue decomposition for deeper insights

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
- statsmodels
- pykalman
- Flask (for web app)
- React (for frontend)

## Installation

### For the Analytics Modules

```bash
pip install -r requirements.txt
```

### For the Web Application

The easiest way to install and run the web application is to use the provided script:

```bash
# Make the script executable (if not already)
chmod +x run_app.sh

# Run the application (starts both backend and frontend)
./run_app.sh
```

Or manually:

```bash
# Install backend dependencies
pip install flask flask-cors numpy pandas scikit-learn scipy statsmodels pykalman

# Start the Flask backend
python app.py

# In a separate terminal
# Install frontend dependencies
cd frontend
npm install

# Start the React frontend
npm start
```

## Usage

### Analytics Modules

Run the basic version:

```bash
python cardcounter.py
```

Run the advanced version with visualizations:

```bash
python advanced_cardcounter.py
```

Run the pattern recognition module:

```bash
python pattern_recognition.py
```

Run the combined detector:

```bash
python combined_detector.py
```

### Web Application

After starting the web application (see Installation section above), the game should automatically open in your default browser at http://localhost:3000.

If the browser doesn't open automatically, you can manually navigate to:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000

#### Troubleshooting
If you see "Loading game..." for an extended period:
1. Check your browser's console (F12) for network errors
2. Ensure the Flask backend is running at http://localhost:5000
3. Verify you've installed flask-cors: `pip install flask-cors`
4. Try the API status endpoint: http://localhost:5000/api/status

## How It Works

1. **Data Representation:** Each Blackjack hand is represented by the dealer's upcard, player's hand value, action taken, bet size, and running count.
2. **Dimensionality Reduction:** SVD and PCA transform this high-dimensional data into a lower-dimensional space that captures the most important patterns.
3. **Correlation Analysis:** The system calculates correlation matrices to identify relationships between variables.
4. **Cholesky Decomposition:** The correlation matrix is decomposed using Cholesky factorization for more efficient calculations.
5. **Kalman Filtering:** Real-time tracking of player behavior with noise reduction.
6. **Markov Chain Modeling:** Prediction of future betting patterns based on current behavior.
7. **Similarity Analysis:** The system compares player behavior to known card counter patterns using cosine similarity.
8. **Mahalanobis Distance:** Detects outliers in player behavior accounting for feature correlations.
9. **Combined Decision:** Multiple detection methods are used together to arrive at a final verdict.

## Advanced Linear Algebra Techniques

### Correlation Matrices and Cholesky Decomposition
The system calculates correlation matrices between different player behaviors (bet size, running count, actions) and uses Cholesky decomposition to factorize these matrices. This allows for:
- Identifying relationships between betting patterns and card counting indicators
- More efficient calculations in high-dimensional spaces
- Generating correlated random variables for simulation
- Solving linear systems efficiently

### Kalman Filtering
Kalman filtering is used for real-time tracking of player behavior with noise reduction:
- Predicts future behavior based on past observations
- Reduces noise in measurements
- Updates predictions based on new observations
- Provides a more accurate estimate of player behavior over time

### Markov Chain Modeling
The system uses Markov chains to model and predict betting patterns:
- Predicts future bet sizes based on current betting behavior
- Models the transition probabilities between different bet sizes
- Identifies suspicious patterns in betting behavior
- Compares actual betting behavior to predicted behavior

### Mahalanobis Distance
Mahalanobis distance is used to detect outliers in player behavior:
- Accounts for correlations between features
- More accurate than Euclidean distance for correlated data
- Identifies unusual combinations of features
- Provides a statistical measure of how unusual a player's behavior is

## Web Application Features

- **Interactive Blackjack Game:** Play against a dealer with standard rules
- **Betting Control:** Set your bet size before each hand
- **Real-time Detection:** See how likely your play pattern matches known card counters
- **Visualization:** Graph showing your detection score over time
- **Risk Level Indicator:** Visual feedback on how suspicious your play appears
- **Correlation Analysis:** View the correlation matrix between different player behaviors
- **Eigenvalue Decomposition:** See the principal components of player behavior
- **Cholesky Decomposition:** View the Cholesky factors of the correlation matrix
- **Kalman Filter State:** Track the estimated state of player behavior over time
- **Markov Chain Predictions:** See predictions of future betting patterns 