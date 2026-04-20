# 🤖 ML Model Comparison Dashboard

A powerful web-based machine learning dashboard that allows you to **train, compare, and optimize multiple ML algorithms** on your dataset. Perfect for data scientists, students, and ML enthusiasts!

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Flask](https://img.shields.io/badge/flask-2.3+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🌟 Features

### ✨ Core Capabilities

- **📤 Easy Data Upload**: Upload any CSV file (classification or regression)
- **🤖 7 ML Algorithms**: Train multiple models simultaneously
  - Logistic Regression
  - Linear Regression
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost (Gradient Boosting)
  - K-Nearest Neighbors (KNN)
  - Decision Tree

- **📊 Comprehensive Metrics Comparison**:
  - **Classification**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - **Regression**: MSE, RMSE, MAE, R² Score

- **⭐ Feature Importance Visualization**: Understand which features drive predictions
- **⚙️ Hyperparameter Tuning**: Adjust model parameters and retrain instantly
- **🎯 Best Model Recommendation**: Automatic identification of top-performing model
- **🐳 Docker Ready**: One-command deployment with Docker Compose

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (Recommended)
  - Download: https://www.docker.com/products/docker-desktop
- **OR Python 3.11+** with pip (Alternative)

### Installation & Setup

#### **Option 1: Docker (Recommended - Easiest)** 🐳

```bash
# Clone the repository
git clone https://github.com/Lovishkasonii/ml-model-comparison-dashboard.git
cd ml-model-comparison-dashboard

# Build and run with Docker
docker-compose build
docker-compose up
```

**Access at**: `http://localhost:5000`

#### **Option 2: Local Setup (Python)** 🐍

```bash
# Clone the repository
git clone https://github.com/Lovishkasonii/ml-model-comparison-dashboard.git
cd ml-model-comparison-dashboard

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Run the Flask app
python backend/app.py
```

**Access at**: `http://localhost:5000`

---

## 📖 Usage Guide

### Step 1️⃣: Upload Your Dataset

1. Open `http://localhost:5000` in your browser
2. Click the **📤 upload area** to select your CSV file
3. Choose the **Target Column** (what you want to predict)
4. Click **"Upload & Preprocess"**

**✅ Supported Formats:**
- CSV files with numeric and categorical features
- Classification datasets (binary or multi-class)
- Regression datasets (continuous target)

**⚠️ Data Cleaning Tips:**
- Remove ID columns (CustomerID, RowNumber, etc.)
- Handle missing values before upload
- Ensure target column exists

### Step 2️⃣: Train All Models

1. After preprocessing, click **"Train All Models"** button
2. Wait for training to complete (typically 30-60 seconds)
3. Dashboard displays results automatically

**What Happens:**
- ✅ All 7 models train on your data
- ✅ Metrics calculated for each model
- ✅ Best model highlighted
- ✅ Feature importance computed

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

## 📁 Project Structure

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
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory (optional):

```env
FLASK_ENV=development
FLASK_DEBUG=True
SERVER_PORT=5000
```

### Hyperparameter Defaults

Edit `backend/ml_models.py` to adjust default hyperparameters:

```python
'Random Forest': RandomForestClassifier(
    n_estimators=50,      # Increase for better accuracy (slower)
    max_depth=10,         # Limit tree depth to prevent overfitting
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)
```

---

## 📊 API Endpoints

### Backend Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve dashboard frontend |
| `/health` | GET | Check server health |
| `/upload` | POST | Upload CSV & preprocess data |
| `/train` | POST | Train all models |
| `/feature-importance/<model>` | GET | Get feature importance |
| `/tune` | POST | Tune hyperparameters & retrain |

### Request/Response Examples

**Upload Data**
```bash
curl -X POST http://localhost:5000/upload \
  -F "file=@dataset.csv" \
  -F "target_column=Churn"
```

**Train Models**
```bash
curl -X POST http://localhost:5000/train
```

**Get Feature Importance**
```bash
curl http://localhost:5000/feature-importance/Random%20Forest
```

---

## 🐛 Troubleshooting

### Issue: Port 5000 Already in Use

**Solution**: Change port in `docker-compose.yml`

```yaml
services:
  ml-dashboard:
    ports:
      - "5001:5000"  # Change to 5001 or any available port
```

Then access: `http://localhost:5001`

---

### Issue: Docker Build Fails

**Solution**: Rebuild without cache

```bash
docker-compose down
docker-compose build --no-cache
docker-compose up
```

---

### Issue: Training Takes Too Long

**Solution**: Reduce model complexity in `ml_models.py`

```python
# Reduce from 100 to 50
'Random Forest': RandomForestClassifier(n_estimators=50)
```

---

### Issue: "No module named 'pkg_resources'"

**Solution**: On Windows PowerShell (as Administrator)

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r backend/requirements.txt
```

---

### Issue: CSV Upload Not Working

**Checklist:**
- ✅ File is `.csv` format
- ✅ Has header row with column names
- ✅ Target column exists in data
- ✅ No spaces in column names (use underscores)
- ✅ File size < 50MB

---

## 📦 Dependencies

### Backend
- **Flask** - Web framework
- **Flask-CORS** - Cross-origin support
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms
- **XGBoost** - Gradient boosting
- **Matplotlib** - Visualization
- **Seaborn** - Statistical plots
- **Plotly** - Interactive charts
- **Joblib** - Model serialization

See `backend/requirements.txt` for exact versions.

---

## 💡 Usage Examples

### Example 1: Customer Churn Prediction

```
Dataset: Telecom customer data
Target: Churn (Yes/No)
Features: Tenure, Monthly Charges, Contract Length, etc.

Results: Random Forest achieves 85% accuracy
Best Feature: Contract Length (0.42 importance)
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

## 🎓 Learning Outcomes

By using this dashboard, you'll learn:

- [ ] **Data Preprocessing**: Cleaning and preparing data for ML
- [ ] **Model Training**: Working with 7 different ML algorithms
- [ ] **Model Evaluation**: Understanding performance metrics
- [ ] **Feature Engineering**: Identifying important features
- [ ] **Hyperparameter Tuning**: Optimizing model performance
- [ ] **Web Development**: Building ML web applications
- [ ] **Docker**: Containerizing applications

---

## 🚀 Advanced Usage

### Using with Your Own Datasets

1. **Prepare your CSV** with features and target column
2. **Auto Remove feature to remove least important feature** (CustomerID, RowNumber, etc.)
3. **Handle missing values** (drop or impute)
4. **Ensure numeric/categorical consistency**
5. **Upload to dashboard**

### Exporting Results

After training, you can:
- 📸 Screenshot the metrics table
- 📊 Export feature importance plots
- 💾 Check for the best model parameters
- 📝 Document model performance

---

## 🔮 Future Enhancements

### Planned Features
- [ ] Cross-validation support
- [ ] Model persistence (save/load trained models)
- [ ] Batch prediction API
- [ ] Advanced data visualization (PCA, t-SNE)
- [ ] Anomaly detection
- [ ] Time series forecasting
- [ ] Ensemble methods
- [ ] Model explainability (SHAP values)
- [ ] Real-time model monitoring
- [ ] Model deployment options

### Community Contributions
Pull requests are welcome! Please follow:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📞 Support & Contact

### Getting Help

- 📖 Check **Troubleshooting** section above
- 🐛 Open an issue on GitHub
- 💬 Review code comments in source files

### Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [ML Algorithms Guide](https://scikit-learn.org/stable/modules/supervised_learning.html)
- [Docker Guide](https://docs.docker.com/)

---

## 📄 License

This project is licensed under the **MIT License** - see LICENSE file for details.

---

## 👤 Author

**Lovishka Soni**
- GitHub: [@Lovishkasoni](https://github.com/Lovishkasoni)

---

## Acknowledgments

- Scikit-learn community for excellent ML libraries
- Flask team for the web framework
- Docker for containerization

---

## 📈 Project Statistics

- **Models Supported**: 7
- **Metrics Types**: 5+
- **Data Formats**: CSV
- **Response Time**: < 60 seconds
- **Deployment**: Docker-ready

---

## 🎯 Quick Reference

### Common Commands

```bash
# Start dashboard
docker-compose up

# Rebuild after changes
docker-compose build --no-cache

# Stop dashboard
docker-compose down

# View logs
docker-compose logs -f

# Remove everything
docker-compose down -v
```

---


Give this project a ⭐ if you found it helpful!

---

**Last Updated**: April 20, 2026
**Version**: 1.0.0
