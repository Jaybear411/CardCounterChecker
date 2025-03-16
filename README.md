# Card Counter Checker

A system that uses Singular Value Decomposition (SVD) and linear algebra to detect card counting behavior in Blackjack players.

## Overview

Card counting in Blackjack gives players an advantage by tracking the ratio of high to low cards. Casinos want to detect such strategies by analyzing betting behavior and decision-making patterns.

This system:
1. Collects a dataset of Blackjack hands
2. Transforms the data into a numerical matrix using features like dealer upcard, player hand value, action taken, bet size, and running count
3. Applies Singular Value Decomposition (SVD) to reduce dimensionality and extract latent behavioral patterns
4. Uses cosine similarity to compare each hand's behavior to those of known card counters
5. Flags players with behavior similar to card counters

## Features

- **Data Simulation:** Creates synthetic datasets where each row represents a Blackjack hand
- **Normalization:** Scales features using StandardScaler for uniformity
- **SVD Application:** Reduces the dataset to principal components capturing key variations in behavior
- **Cosine Similarity:** Compares reduced representations to known card counter patterns
- **Anomaly Detection:** Uses Isolation Forest to identify outlier betting behavior
- **Pattern Recognition:** Analyzes betting patterns relative to the running count
- **Kelly Criterion Analysis:** Checks how closely betting behavior follows optimal strategies
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
   - Analyzes behavior in real-time for card counting patterns
   - Displays a graph showing how suspicious the play pattern appears
   - Provides a visual indicator of detection risk level

## Requirements

- Python 3.x
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- scipy
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
pip install flask flask-cors numpy pandas scikit-learn

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
2. **Dimensionality Reduction:** SVD transforms this high-dimensional data into a lower-dimensional space that captures the most important patterns.
3. **Similarity Analysis:** The system compares player behavior to known card counter patterns using cosine similarity.
4. **Pattern Analysis:** Betting patterns are analyzed for correlation with the running count.
5. **Combined Decision:** Multiple detection methods are used together to arrive at a final verdict.

## Web Application Features

- **Interactive Blackjack Game:** Play against a dealer with standard rules
- **Betting Control:** Set your bet size before each hand
- **Real-time Detection:** See how likely your play pattern matches known card counters
- **Visualization:** Graph showing your detection score over time
- **Risk Level Indicator:** Visual feedback on how suspicious your play appears

## Customization

- Modify the synthetic data generation to match your real-world dataset
- Tune the various thresholds for each detection method
- Adjust the number of SVD components based on your dataset's complexity
- Change the visualization settings to highlight specific aspects of the data 