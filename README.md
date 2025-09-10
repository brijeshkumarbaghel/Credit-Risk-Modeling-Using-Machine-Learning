# Credit Risk Modeling Using Machine Learning

A comprehensive credit risk assessment system built with Python and machine learning algorithms to predict loan default probability using the German Credit Dataset.

## 🎯 Project Overview

This project implements a complete credit risk modeling pipeline that analyzes borrower characteristics and predicts credit risk using various machine learning algorithms. The system helps financial institutions make informed lending decisions by assessing the probability of loan defaults.

## 📊 Dataset

- **Dataset**: German Credit Data
- **Records**: 1000 loan applications
- **Features**: 10 key attributes including demographics, financial status, and loan details
- **Target**: Binary classification (Good/Bad credit risk)

### Key Features:
- **Age**: Borrower's age
- **Sex**: Gender of the borrower
- **Job**: Employment category (0-3)
- **Housing**: Housing status (own, rent, free)
- **Saving accounts**: Saving account status
- **Checking account**: Checking account status
- **Credit amount**: Loan amount requested
- **Duration**: Loan duration in months
- **Purpose**: Loan purpose
- **Risk**: Target variable (good/bad)

## 🔧 Technologies Used

- **Python 3.x**
- **Data Analysis**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, XGBoost
- **Model Persistence**: Joblib
- **Development Environment**: Jupyter Notebook

## 📁 Project Structure

```
Credit-Risk-Modeling-Using-Machine-Learn/
│
├── analysis_model.ipynb          # Main analysis and modeling notebook
├── german_credit_data.csv        # Dataset
├── XGBoost_credit_model.pkl      # Trained XGBoost model
├── Checking account_encoder.pkl  # Checking account encoder
├── Housing_encoder.pkl           # Housing encoder
├── Job_encoder.pkl              # Job encoder
├── Saving accounts_encoder.pkl   # Saving accounts encoder
├── Sex_encoder.pkl              # Sex encoder
├── target_encoder.pkl           # Target variable encoder
└── README.md                    # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib jupyter
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/brijeshkumarbaghel/Credit-Risk-Modeling-Using-Machine-Learn.git
cd Credit-Risk-Modeling-Using-Machine-Learn
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook analysis_model.ipynb
```

## 📈 Analysis Pipeline

### 1. Exploratory Data Analysis (EDA)
- Data loading and preprocessing
- Missing value analysis and handling
- Statistical summary and data distribution analysis
- Correlation analysis between features

### 2. Data Visualization
- **Numerical Features**: Histograms and box plots for Age, Credit Amount, Duration
- **Categorical Features**: Count plots for demographic and account information
- **Risk Analysis**: Distribution analysis by risk categories
- **Advanced Visualizations**: Scatter plots, violin plots, and pivot tables

### 3. Data Preprocessing
- **Missing Value Treatment**: Dropped null values
- **Feature Encoding**: Label encoding for categorical variables
- **Target Encoding**: Binary encoding for risk classification
- **Train-Test Split**: 80-20 stratified split

### 4. Model Development
Four machine learning algorithms were implemented and tuned:

#### Models Tested:
1. **Decision Tree Classifier**
2. **Random Forest Classifier** 
3. **Extra Trees Classifier**
4. **XGBoost Classifier** (Best performing)

#### Hyperparameter Tuning:
- Grid Search CV with 5-fold cross-validation
- Class weight balancing for imbalanced dataset
- Comprehensive parameter optimization

## 🏆 Model Performance

The XGBoost classifier achieved the best performance and was selected as the final model:

- **Algorithm**: XGBoost Classifier
- **Cross-validation**: 5-fold
- **Class Balancing**: Applied scale_pos_weight
- **Model File**: `XGBoost_credit_model.pkl`

## 💡 Key Insights

1. **Credit Amount vs Age**: Older borrowers tend to request higher loan amounts
2. **Duration Impact**: Longer loan durations correlate with higher risk
3. **Employment Status**: Job category significantly influences credit risk
4. **Account Status**: Checking and saving account status are strong predictors

## 🔮 Model Usage

```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('XGBoost_credit_model.pkl')

# Load encoders
encoders = {}
for feature in ['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account']:
    encoders[feature] = joblib.load(f'{feature}_encoder.pkl')

# Prepare new data and make predictions
# predictions = model.predict(encoded_data)
```

## 📊 Business Value

- **Risk Assessment**: Automated credit risk evaluation
- **Decision Support**: Data-driven lending decisions  
- **Cost Reduction**: Reduced manual underwriting effort
- **Compliance**: Standardized risk assessment process

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Brijesh Kumar Baghel**
- GitHub: [@brijeshkumarbaghel](https://github.com/brijeshkumarbaghel)

## 🙏 Acknowledgments

- German Credit Dataset from UCI Machine Learning Repository
- Scikit-learn and XGBoost communities for excellent ML libraries
- Open source community for tools and resources

---

*For questions or suggestions, please open an issue or contact the author.*