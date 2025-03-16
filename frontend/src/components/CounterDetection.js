import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import './CounterDetection.css';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const CounterDetection = ({ detection, runningCount }) => {
  // Prepare chart data if detection history exists
  const chartData = detection && detection.history ? {
    labels: detection.history.map(item => `Hand ${item.hand_number}`),
    datasets: [
      {
        label: 'Similarity Score',
        data: detection.history.map(item => item.similarity_score),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.4,
      },
      {
        label: 'Suspicion Threshold',
        data: detection.history.map(() => detection.similarity_threshold || 0.70),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
      },
    ],
  } : null;

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Card Counting Detection Score Over Time',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += context.parsed.y.toFixed(3);
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        min: 0,
        max: 1,
        title: {
          display: true,
          text: 'Similarity Score'
        }
      }
    }
  };

  // Calculate status indicator class based on detection
  const getStatusClass = () => {
    if (!detection) return 'neutral';
    
    const { similarity_score } = detection;
    if (similarity_score > 0.85) return 'high';
    if (similarity_score > 0.7) return 'medium';
    return 'low';
  };

  // Get status text based on detection
  const getStatusText = () => {
    if (!detection) return 'No data';
    
    const { similarity_score } = detection;
    if (similarity_score > 0.85) return 'High Risk';
    if (similarity_score > 0.7) return 'Medium Risk';
    return 'Low Risk';
  };

  return (
    <div className="counter-detection">
      <h2>Card Counting Detection</h2>
      
      <div className="detection-info">
        <div className="detection-stats">
          <div className="stat-item">
            <span className="stat-label">Running Count:</span>
            <span className="stat-value">{runningCount}</span>
          </div>
          
          {detection && (
            <>
              <div className="stat-item">
                <span className="stat-label">Current Score:</span>
                <span className="stat-value">
                  {detection.similarity_score.toFixed(3)}
                </span>
              </div>
              
              <div className="stat-item">
                <span className="stat-label">Status:</span>
                <div className={`status-indicator ${getStatusClass()}`}>
                  {getStatusText()}
                </div>
              </div>
              
              <div className="stat-item">
                <span className="stat-label">Hands Analyzed:</span>
                <span className="stat-value">
                  {detection.history ? detection.history.length : 0}
                </span>
              </div>
            </>
          )}
        </div>
        
        <div className="detection-explanation">
          <h3>How it works:</h3>
          <p>
            This system detects card counting by analyzing your betting patterns 
            and decisions using Singular Value Decomposition (SVD) and linear algebra.
            Higher similarity scores indicate a betting pattern similar to known card counters.
          </p>
        </div>
      </div>
      
      <div className="detection-chart">
        {chartData ? (
          <Line data={chartData} options={chartOptions} />
        ) : (
          <div className="no-data">
            Play more hands to generate detection data
          </div>
        )}
      </div>
    </div>
  );
};

export default CounterDetection; 