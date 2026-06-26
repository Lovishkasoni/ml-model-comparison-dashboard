#  ML Model Comparison Dashboard

A powerful web-based machine learning dashboard that allows you to **train, compare, and optimize multiple ML algorithms** on your dataset. Perfect for data scientists, students, and ML enthusiasts!

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

##  Features

###  Core Capabilities

- **Easy Data Upload**: Upload any CSV file (classification or regression)
- **ML Algorithms**: Train multiple models simultaneously
  - Logistic Regression
  - Linear Regression
  - Random Forest
  - XGBoost (Gradient Boosting0
  - Decision Tree

- **Comprehensive Metrics Comparison**:
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - **Regression**: MSE, RMSE, MAE, R² Score

- **Feature Importance Visualization**: Understand which features drive predictions
- **Hyperparameter Tuning**: Adjust model parameters and retrain instantly
- **Best Model Recommendation**: Automatic identification of top-performing model

**Access at**: `https://ml-model-comparison-dashboard.onrender.com`

## 📖 Usage Guide

### Step 1️⃣: Upload Your Dataset

1. Open `https://ml-model-comparison-dashboard.onrender.com` in your browser
2. Click the **upload area** to select your CSV file or you can download the sample dataset from backend/customer_churn_dataset.csv
3. Choose the **Target Column** (what you want to predict)
4. Click **"Upload & Preprocess"**

**Supported Formats:**
- CSV files with numeric and categorical features
- Classification datasets (binary or multi-class)
- Regression datasets (continuous target)

**Data Cleaning Tips:**
- Remove ID columns (CustomerID, RowNumber, etc.)
- Handle missing values before upload
- Ensure target column exists

### Step 2️⃣: Train All Models

1. After preprocessing, click **"Train All Models"** button
2. Wait for training to complete (typically 30-60 seconds)
3. Dashboard displays results automatically

**What Happens:**
- All the models train on your data
- Metrics calculated for each model
- Best model highlighted
- Feature importance computed

### Step 3️⃣: Compare Results

1. View the **Performance Metrics Table** showing all models' results
2. Identify the **Best Performing Model** (highlighted in green)
3. Compare metrics across different models

### Step 4️⃣: Analyze Feature Importance

1. Select a model from the **"Feature Importance"** dropdown
2. View which features are most important for predictions
3. Higher importance = stronger prediction influence

### Step 5️⃣: Tune Hyperparameters

1. Select a model from the **"Hyperparameter Tuning"** section
2. Adjust parameter values (e.g., learning_rate, n_neighbors)
3. Click **"Retrain with New Parameters"**
4. Compare new metrics with original results

---

## Project Structure

```
ml-model-comparison-dashboard/
│
├── 📂 backend/
│   ├── app.py                    # Flask application & routes
│   ├── ml_models.py              # ML algorithms implementation
│   ├── data_processor.py         # Data preprocessing & cleaning
│   ├── requirements.txt          # Python dependencies
│   └── sample_data/
│       └── churn_dataset.csv     # Sample dataset for testing
│
├── 📂 frontend/
│   ├── index.html                # Main dashboard UI
│   ├── style.css                 # Styling & layouts
│   └── script.js                 # Client-side logic
│
├── Dockerfile                     # Docker container config
├── docker-compose.yml            # Multi-container setup
└── README.md                     # This file
```
---

##  API Endpoints

### Backend Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve dashboard frontend |
| `/health` | GET | Check server health |
| `/upload` | POST | Upload CSV & preprocess data |
| `/train` | POST | Train all models |
| `/feature-importance/<model>` | GET | Get feature importance |
| `/tune` | POST | Tune hyperparameters & retrain |

---

### Issue: Training Takes Too Long


---

### Issue: CSV Upload Not Working

**Checklist:**
- File is `.csv` format
- Has header row with column names
- Target column exists in data
- No spaces in column names (use underscores)
- File size < 50MB

---

## 📦 Dependencies

### Backend
- Python
- Flask
- Scikit-Learn
- XGBoost
- Pandas
- NumPy

### Frontend
- HTML5
- CSS3
- JavaScript
- Plotly.js

### Deployment
- Docker
- Render

---

## Usage Examples

### Example 1: Customer Churn Prediction

```
Dataset: Telecom customer data
Target: Churn (Yes/No)
Features: Tenure, Monthly Charges, Contract Length, etc.

Results: Decision Tree achieves 98% accuracy
Best Feature: Support Calls (0.36 importance)
```

### Example 2: House Price Prediction

```
Dataset: Real estate properties
Target: Price (continuous)
Features: Square feet, Bedrooms, Location, etc.

Results: XGBoost achieves R² = 0.89
Best Feature: Square Feet (0.65 importance)
```

---

## Example Workflow
1. Upload CSV dataset
2. Select target column
3. Preprocess data automatically
4. Train all models
5. Compare performance metrics
6. Analyze feature importance
7. Tune hyperparameters
8. Select best-performing model

---

## Learning Outcomes

### This project demonstrates:

- Machine Learning Pipeline Design
- Data Preprocessing Automation
- Model Evaluation & Comparison
- Feature Engineering Concepts
- Full-Stack ML Application Development
- REST API Development
- Docker Containerization
- Interactive Data Visualization

---

### Community Contributions
Pull requests are welcome! Please follow:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## Support & Contact

### Getting Help

- Check **Troubleshooting** section above
- Open an issue on GitHub
- Review code comments in source files

### Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ML Algorithms Guide](https://scikit-learn.org/stable/modules/supervised_learning.html)
- [Docker Guide](https://docs.docker.com/)

---

## License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## Author

**Lovishka Soni**
- GitHub: [@Lovishkasoni](https://github.com/Lovishkasoni)

---

## Acknowledgments

- Scikit-learn community for excellent ML libraries
- Flask team for the web framework
- Docker for containerization

---


Give this project a ⭐ if you found it helpful!

---

**Last Updated**: June 26, 2026
**Version**: 1.0.0
