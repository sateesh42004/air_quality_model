# Air Quality Prediction (PM2.5)

This project predicts **PM2.5 concentrations** using meteorological and air quality data with a **Linear Regression model**.  
The goal is to understand how factors like **PM10, air temperature, and time features** influence PM2.5 pollution levels.

---

## 📂 Project Structure

```
├── air_quality.csv        # Dataset
├── air_quality_model.py   # Main script (training & evaluation)
├── README.md              # Project documentation
```

---

##  Requirements

Install dependencies before running the project:

```bash
pip install pandas scikit-learn matplotlib
```

---

## How to Run

1. Clone this repository or download the project files.  
2. Place your dataset `air_quality.csv` in the project directory.  
3. Run the script:

```bash
python air_quality_model.py
```

---

## 📊 Dataset Features

- **year** – Year of measurement  
- **month** – Month of measurement  
- **day** – Day of measurement  
- **hour** – Hour of measurement  
- **PM10** – Particulate Matter ≤ 10 µm  
- **AT** – Air Temperature  

Target variable:  
- **PM2.5** – Particulate Matter ≤ 2.5 µm  

---

##  Model Details

- **Algorithm:** Linear Regression  
- **Data Split:** 80% training / 20% testing  
- **Evaluation Metrics:**  
  - Mean Squared Error (MSE)  
  - R² Score  

---

## Results

- Scatter plot of **Actual vs Predicted PM2.5** values is generated.  
- Predictions generally follow the trend of actual values.  
- The model performs well for **low PM2.5 values**, but **underestimates at higher values**.  

---

## Future Improvements

- Try advanced models: **Random Forest, Gradient Boosting, XGBoost**  
- Add **feature scaling** for better regression performance  
- Explore **time-series forecasting** methods  
- Perform **feature engineering** (humidity, wind speed, seasonal effects, etc.)  

---

## Example Output

- **Mean Squared Error (MSE):** *(varies with dataset)*  
- **R² Score:** *(shows model accuracy)*  

Example Plot:  

![Example Scatter Plot](example_output.png)  
*(Blue dots = Predictions, Red dashed line = Perfect prediction line)*  

---

 
