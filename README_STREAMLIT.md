# 🧠 Alzheimer's Disease Prediction System - Streamlit App

A complete web application built with Streamlit for predicting Alzheimer's disease using machine learning.

## Features

### 📊 Multi-Page Interface
- **Home**: Welcome page with disease information and dataset statistics
- **Prediction**: Interactive patient assessment with real-time predictions
- **About**: Project overview and methodology
- **Model Info**: Model architecture, performance metrics, and feature statistics

### 🎯 Key Capabilities
- Real-time patient risk assessment
- Interactive input sliders for all clinical features
- Probability-based predictions with confidence levels
- Comprehensive model performance metrics
- Dataset statistics and feature information
- Models comparison chart
- Professional UI with custom styling

### 🤖 Machine Learning Model
- **Algorithm**: Gradient Boosting Classifier
- **Type**: Binary Classification (Alzheimer's vs. Healthy)
- **Training Accuracy**: 96.68%
- **Testing Accuracy**: 95.81%
- **Preprocessing**: StandardScaler normalization

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

1. **Clone or download the project**
   ```bash
   cd /home/astro/final project creativa
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model file exists**
   - The file `pipeline_final.joblib` must be in the project directory
   - The file `alzheimers_disease_data.csv` must be in the project directory

## Running the Application

### Start the Streamlit App
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

### Alternative Port
If port 8501 is busy, specify a different port:
```bash
streamlit run app.py --server.port 8502
```

## Usage Guide

### 1. Home Page
- View information about Alzheimer's disease
- See dataset overview and statistics
- Understand why early detection is important

### 2. Prediction Page
- Use sliders to input patient clinical values
- Features range from minimum to maximum values in training data
- Click "Make Prediction" button to get results
- View prediction confidence levels
- See all input features in a summary table

### 3. About Page
- Learn about the project objectives
- Understand data sources and features used
- Review the methodology
- Important disclaimer about clinical use

### 4. Model Info Page
- View model architecture details
- Check performance metrics
- Review dataset statistics
- See comparison of different models tested

## Project Structure

```
/home/astro/final project creativa/
├── app.py                      # Main Streamlit application
├── pipeline_final.joblib       # Trained ML model
├── alzheimers_disease_data.csv # Training dataset
├── requirements.txt            # Python dependencies
├── models.py                   # ML model utilities
├── ex_011.ipynb               # Data preprocessing notebook
└── README.md                   # This file
```

## Model Performance

### Classification Metrics (Test Set)
```
                precision    recall  f1-score   support
        Healthy       0.96      0.98      0.97       277
    Alzheimer's       0.96      0.92      0.94       153
        accuracy                           0.96       430
      macro avg       0.96      0.95      0.95       430
   weighted avg       0.96      0.96      0.96       430
```

## Features Used in Prediction

The model uses 30+ clinical and cognitive assessment features including:
- MMSE (Mini-Mental State Examination) Score
- Patient Age
- Gender
- Cognitive Test Scores
- Blood Biomarkers
- CSF Biomarkers
- MRI Measurements
- Health Indicators

## Clinical Features Explained

- **Age**: Patient's age in years
- **Gender**: 0 = Male, 1 = Female
- **MMSE**: Cognitive screening score (0-30, higher is better)
- **Biomarkers**: Protein levels indicating neurological changes
- **MRI**: Brain volume and structural measurements

## Important Disclaimer

⚠️ **Medical Disclaimer**: This application is for educational and research purposes only. 
The predictions made by this model should NOT be used for clinical diagnosis or treatment decisions.

Always consult with qualified healthcare professionals, neurologists, or physicians for:
- Diagnosis confirmation
- Treatment planning
- Medical advice
- Clinical decision-making

## Troubleshooting

### Issue: "Model file not found"
- Solution: Ensure `pipeline_final.joblib` exists in the same directory as `app.py`

### Issue: "Data file not found"
- Solution: Ensure `alzheimers_disease_data.csv` exists in the project directory

### Issue: Package installation errors
- Solution: Try installing packages individually or use a virtual environment

### Issue: Port already in use
- Solution: Use `streamlit run app.py --server.port 8502` with a different port

## Development Notes

### Customization Options

1. **Change Model**: Replace `pipeline_final.joblib` with another trained model
2. **Modify Features**: Edit the feature list in the feature selection section
3. **Update Styling**: Modify CSS in the custom CSS section
4. **Add Pages**: Add more navigation options in the sidebar

### Performance Optimization

- Model is cached using `@st.cache_resource` for faster loading
- Dataset is cached using `@st.cache_data` to avoid reloading
- Predictions are computed only when button is clicked

## Contact & Support

For issues or improvements, please refer to the project documentation or contact the development team.

## License

This project is provided for educational purposes.

---

**Last Updated**: February 2026
**Version**: 1.0
**Framework**: Streamlit 1.28+
