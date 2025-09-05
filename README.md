# F1 Race Analysis

A personal project for analyzing Formula 1 race data and predicting Q3 qualifying outcomes using machine learning. This is a work in progress focused on fine-tuning the prediction model.

## 🏁 Features

- **Circuit-Specific Analysis**: Uses 2024 data to calculate team and driver performance at specific circuits
- **Automatic Trend Improvement**: Applies 2023→2024 performance gains to project 2025 pole times
- **Advanced Feature Engineering**: 18+ features including driver form, team performance, track characteristics
- **Multiple ML Models**: Random Forest, Logistic Regression, and Gradient Boosting with cross-validation
- **Realistic Predictions**: Percentage-based calculations anchored to actual 2024 pole times
- **Interactive CLI**: User-friendly interface for analysis and predictions
- **Data Leakage Prevention**: Strict temporal train/test split (2022-2023 training, 2024 testing)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-race-analysis.git
cd f1-race-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the system:
```bash
python main.py
```

### First-Time Setup

The system will automatically:
- Download and cache F1 data from FastF1
- Train models on 2022-2023 data (current regulations era)
- Save trained models for fast future predictions
- Validate data leakage prevention

## 📊 Usage

### Main Menu Options

1. **Analyze Current Season** - View race results, driver comparisons, tire strategies
2. **Compare Drivers** - Head-to-head driver performance analysis
3. **Show Tire Strategies** - Visualize tire compound usage and performance
4. **View Session Information** - Detailed session data and timing
5. **Q3 Qualification Prediction** - Predict Q3 outcomes for current sessions
6. **2025 Q3 Prediction (Enhanced Model)** - Predict 2025 qualifying with circuit-specific analysis
7. **Train Models Once** - Retrain models with latest data

### 2025 Prediction Example

```bash
python main.py
# Select option 6: "2025 Q3 Prediction (Enhanced Model)"
# Enter driver code: HAM
# Enter team: Mercedes
# Enter event: Italy 2025
```

**Sample Output:**
```
🎯 REALISTIC 2025 Q3 Qualification Prediction for Italy 2025
============================================================

🏁 Team baseline (Mercedes at Italian Grand Prix):
   Team Gap: 0.65%
   Q3 Rate: 80.0%
   Events: 1

👤 Driver adjustment (HAM at Italian Grand Prix):
   Delta vs RUS: -0.12% (faster)
   Q3 Rate: 85.0%
   Events: 1

⏱️  Lap Time Calculation:
  Formula: estimated_time = pole_2025 × (1 + (team_gap_pct + driver_delta_pct)/100)
  Formula: estimated_time = 79.698 × (1 + 0.53/100)
  Formula: estimated_time = 79.698 × 1.0053
  Estimated Driver Lap Time: 1:19.950

🎯 Prediction: ✅ QUALIFIES for Q3
📊 Model confidence: 78.5%
```

## 🧠 Technical Details

### Model Architecture

- **Training Data**: 2022-2023 seasons (current regulations era)
- **Test Data**: 2024 season
- **Models**: Random Forest, Logistic Regression, Gradient Boosting
- **Cross-Validation**: 5-fold stratified cross-validation
- **Class Balancing**: SMOTE oversampling + class_weight='balanced'

### Feature Engineering

**Driver Features:**
- Recent form (last 5 races)
- Q3 qualification rate
- Performance vs teammate

**Team Features:**
- Average gap to pole position
- Q3 qualification rate
- Average grid position

**Track Features:**
- Track type (permanent/street)
- Difficulty level
- Overtaking opportunities
- Historical Q3 results

**Technical Features:**
- Tire compound performance
- Tire life and freshness
- Relative performance metrics

### Circuit-Specific Calculations

**Team Baseline:**
```
team_gap_pct = (team_best_q3_lap_2024_at_circuit - pole_q3_2024) / pole_q3_2024 * 100
```

**Driver Adjustment:**
```
driver_delta_pct = (driver_best_q3_lap_2024_at_circuit - teammate_best_q3_lap_2024_at_circuit) / pole_q3_2024 * 100
```

**2025 Projection:**
```
projected_pole_2025 = pole_2024 - trend_improvement
estimated_driver_time = projected_pole_2025 × (1 + (team_gap_pct + driver_delta_pct)/100)
```

## 📁 Project Structure

```
f1-race-analysis/
├── main.py                 # CLI entry point
├── data_utils.py          # Data fetching and processing
├── model_utils.py         # ML models and predictions
├── visuals.py             # Plotting and visualization
├── requirements.txt       # Dependencies
├── README.md             # This file
├── models/               # Trained models
│   ├── random_forest_model.joblib
│   ├── logistic_regression_model.joblib
│   ├── gradient_boosting_model.joblib
│   ├── preprocessing_bundle.joblib
│   └── model_metrics.csv
├── cache/                # FastF1 data cache
└── plots/                # Generated visualizations
```

## 🔧 Configuration

### Model Training

To retrain models with latest data:
```bash
python main.py
# Select option 7: "Train Models Once (2022-2023 → 2024)"
```

### Data Sources

- **FastF1**: Official F1 timing data
- **Seasons**: 2022-2025 (current regulations era)
- **Cache**: Automatic caching for faster subsequent runs

## 📈 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall prediction correctness
- **F1-Score**: Balanced precision and recall
- **AUC**: Area under ROC curve
- **Precision**: True positive rate
- **Recall**: Sensitivity

Typical performance on 2024 test data:
- **Random Forest**: F1-Score ~0.75, Accuracy ~0.80
- **Logistic Regression**: F1-Score ~0.72, Accuracy ~0.78
- **Gradient Boosting**: F1-Score ~0.77, Accuracy ~0.82

## 🛠️ Dependencies

- `fastf1` - F1 data access
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `seaborn` - Statistical visualization
- `joblib` - Model persistence
- `imbalanced-learn` - Class balancing (optional)

## 🚨 Data Leakage Prevention

The system implements strict temporal separation:
- **Training**: 2022-2023 data only
- **Testing**: 2024 data only
- **Features**: Calculated using only historical data
- **Validation**: Comprehensive leakage detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) for F1 data access
- Formula 1 for providing official timing data

---

**Note**: This is a personal project for educational purposes. Predictions are based on historical data and should not be used for betting or gambling.