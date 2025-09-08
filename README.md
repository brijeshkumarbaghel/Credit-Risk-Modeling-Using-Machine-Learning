# Credit Risk Modeling System

This project is a machine learning-based credit risk modeling system using XGBoost and various encoders. It predicts credit risk based on German credit data and provides a user-friendly interface for analysis and prediction.

## Features
- Predicts credit risk using a trained XGBoost model
- Encodes categorical features with pre-trained encoders
- Uses German credit data for training and evaluation
- Interactive interface (e.g., Streamlit or notebook)

## Files
- `app.py`: Main application script
- `analysis_model.ipynb`: Data analysis and model development notebook
- `german_credit_data.csv`: Dataset
- `XGBoost_credit_model.pkl`: Trained XGBoost model
- Encoder files: `Checking account_encoder.pkl`, `Housing_encoder.pkl`, `Saving accounts_encoder.pkl`, `Sex_encoder.pkl`, `target_encoder.pkl`

## Getting Started
1. Clone the repository:
   ```powershell
   git clone https://github.com/your-username/credit-risk-modeling-system.git
   ```
2. Install required Python packages:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the application:
   ```powershell
   python app.py
   ```

## Usage
- Modify `app.py` or use the notebook for custom analysis.
- Input new data to predict credit risk.

## License
This project is licensed under the MIT License.

## Author
- Brijesh KumarR
