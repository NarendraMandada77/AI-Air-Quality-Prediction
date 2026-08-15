# 🌍 AI Air Quality Prediction & Personalized Health Advisory

An end-to-end **AI and Deep Learning-based air quality forecasting platform** designed to predict AQI trends, analyze atmospheric conditions, and provide personalized health recommendations.

The system combines historical air-quality datasets, deep-learning time-series models, live weather and air-quality information, CPCB-based AQI calculations, and an interactive Streamlit dashboard into a single application.

---

## 🚀 Project Overview

Air pollution can vary significantly across locations and over time. This project uses historical pollutant measurements and temporal patterns to forecast future **Air Quality Index (AQI)** values.

The platform also analyzes the predicted air-quality conditions and provides health recommendations based on different user profiles.

### Main capabilities

* 📡 Live weather and air-quality data integration
* 📈 Multi-step AQI forecasting
* 🧠 LSTM, GRU, and Transformer deep-learning models
* 🇮🇳 CPCB-based Indian AQI calculation
* 🩺 Personalized health recommendations
* 🌦️ Weather and environmental condition analysis
* 📊 Interactive Streamlit dashboard
* 📉 Model performance comparison
* 🗺️ Location-based air-quality analysis
* ⚙️ Automated data cleaning and feature engineering

---

# ✨ Key Features

## 1. 🌐 Live Environmental Data

The application can retrieve current environmental information for locations around the world using the **Open-Meteo API**.

The live integration provides weather and air-quality information that can be used alongside historical data for analysis and visualization.

---

## 2. 🤖 Deep Learning AQI Forecasting

Three different neural-network architectures are implemented for time-series forecasting:

* **LSTM**
* **GRU**
* **Transformer**

The models learn temporal relationships from historical air-quality observations using **14-day sequences**.

Future AQI values can be forecast for multiple horizons:

* 1 day
* 3 days
* 7 days

This allows the system to analyze both short-term and near-future pollution trends.

---

## 3. 🇮🇳 CPCB AQI Calculation

The system implements the **Indian National Air Quality Index methodology** to calculate AQI values from pollutant concentrations.

The system considers major pollutants such as:

* PM2.5
* PM10
* NO
* NO2
* NOx
* NH3
* CO
* SO2
* O3
* Benzene
* Toluene
* Xylene

The resulting AQI is assigned to an appropriate category:

| AQI Range | Category     |
| --------: | ------------ |
|      0–50 | Good         |
|    51–100 | Satisfactory |
|   101–200 | Moderate     |
|   201–300 | Poor         |
|   301–400 | Very Poor    |
|  401–500+ | Severe       |

The application also identifies the pollutant contributing most significantly to the AQI.

---

# 🩺 Personalized Health Advisory

The system provides environmental health recommendations based on different user groups.

### Supported profiles

**1. General Public**

Provides general precautions and recommendations for daily activities.

**2. Asthma / Respiratory Conditions**

Provides additional precautions during elevated pollution levels, including recommendations concerning outdoor exposure and respiratory protection.

**3. Cardiovascular / Heart Conditions**

Provides guidance for limiting strenuous outdoor activities during unhealthy air-quality conditions.

**4. Elderly Users**

Provides additional precautions for older adults during periods of poor air quality and extreme environmental conditions.

**5. Children & Infants**

Provides recommendations concerning outdoor activities and exposure to polluted environments.

**6. Outdoor Athletes / Workers**

Provides guidance for adjusting outdoor activity and exercise schedules according to pollution levels.

---

# 🧠 Machine Learning Architecture

The forecasting component uses three deep-learning architectures.

## LSTM

The Long Short-Term Memory model is designed to capture long-term dependencies within sequential air-quality observations.

The implementation uses:

* Multiple LSTM layers
* Dropout regularization
* Fully connected regression output

---

## GRU

The Gated Recurrent Unit model provides another recurrent architecture for learning temporal dependencies.

GRU can provide a computationally efficient alternative to LSTM while maintaining strong sequence-learning capabilities.

---

## Transformer

The Transformer model uses self-attention to identify relationships between different time steps in the input sequence.

The implementation includes:

* Positional encoding
* Multi-head self-attention
* Transformer encoder layers
* Regression output layer

The configured architecture uses **4 attention heads**.

---

# 📊 Dataset

The project works with historical air-quality observations collected across multiple Indian cities.

The raw dataset contains city-level and station-level observations.

### Raw data

```text
dataset/
├── city_day.csv
├── city_hour.csv
├── station_day.csv
├── station_hour.csv
└── stations.csv
```

The data-processing pipeline converts the raw information into cleaned and feature-engineered datasets suitable for machine-learning and deep-learning models.

---

# 📁 Processed Dataset

The generated datasets are stored inside:

```text
final_dataset/
```

### `city_day_cleaned.csv`

Contains cleaned daily air-quality observations.

Important fields include:

| Column          | Description               |
| --------------- | ------------------------- |
| City            | Indian city name          |
| Date            | Observation date          |
| PM2.5           | Fine particulate matter   |
| PM10            | Coarse particulate matter |
| NO              | Nitric oxide              |
| NO2             | Nitrogen dioxide          |
| NOx             | Nitrogen oxides           |
| NH3             | Ammonia                   |
| CO              | Carbon monoxide           |
| SO2             | Sulfur dioxide            |
| O3              | Ground-level ozone        |
| Benzene         | Benzene concentration     |
| Toluene         | Toluene concentration     |
| Xylene          | Xylene concentration      |
| AQI             | Calculated AQI            |
| AQI_Bucket      | AQI category              |
| Major_Pollutant | Dominant pollutant        |

---

## 🔬 Feature Engineering

The forecasting dataset contains additional temporal and statistical features.

### Temporal Features

```text
Year
Month
Day
DayOfWeek
DayOfYear
IsWeekend
Season
```

These features help the models recognize seasonal and calendar-based pollution patterns.

### Ratio Features

```text
PM2.5_PM10_ratio
NO2_NOx_ratio
```

These provide additional information about pollutant relationships.

### Lag Features

Historical values are incorporated through lag variables such as:

```text
AQI_lag_1d
AQI_lag_2d
AQI_lag_7d
PM2.5_lag_1d
NO2_lag_1d
```

### Rolling Statistics

Moving-window features include:

```text
AQI_roll_mean_3d
AQI_roll_mean_7d
AQI_roll_mean_14d
AQI_roll_std_7d
PM2.5_roll_mean_7d
```

### Forecast Targets

The model can learn to predict:

```text
AQI_target_1d
AQI_target_3d
AQI_target_7d
```

---

# 🏗️ Project Structure

```text
AI-Air-Quality-Prediction/
│
├── dataset/
│   ├── city_day.csv
│   ├── city_hour.csv
│   ├── station_day.csv
│   ├── station_hour.csv
│   └── stations.csv
│
├── final_dataset/
│   ├── city_day_cleaned.csv
│   ├── city_day_forecasting.csv
│   ├── personalized_health_advisory.csv
│   ├── health_advisory_lookup.csv
│   ├── station_day_cleaned.csv
│   └── city_hour_cleaned.csv
│
├── models/
│   ├── lstm_model.pt
│   ├── gru_model.pt
│   ├── transformer_model.pt
│   ├── scaler.pkl
│   └── model_comparison.json
│
├── scripts/
│   └── clean_and_process.py
│
├── src/
│   ├── models.py
│   ├── train.py
│   ├── eval_models.py
│   ├── advisory_agent.py
│   └── open_meteo_client.py
│
├── app.py
├── run.py
├── server.py
├── DATASET_DOCUMENTATION.md
└── README.md
```

---

# ⚙️ Technology Stack

### Programming

* Python

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Deep Learning

* PyTorch

### Visualization

* Plotly
* Streamlit

### APIs

* Open-Meteo API

### Utilities

* Joblib
* Requests

---

# 🔄 System Workflow

```text
Historical Air Quality Data
            ↓
     Data Cleaning
            ↓
    Feature Engineering
            ↓
      Dataset Creation
            ↓
     Sequence Generation
            ↓
 ┌──────────┼───────────┐
 ↓          ↓           ↓
LSTM       GRU     Transformer
 ↓          ↓           ↓
 └──────────┼───────────┘
            ↓
    Model Evaluation
            ↓
       AQI Forecast
            ↓
  Health Risk Assessment
            ↓
 Personalized Advisory
            ↓
    Streamlit Dashboard
```

---

# 🛠️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/NarendraMandada77/AI-Air-Quality-Prediction.git
cd AI-Air-Quality-Prediction
```

---

## 2. Install Required Packages

```bash
pip install torch pandas numpy scikit-learn joblib plotly streamlit requests
```

If a `requirements.txt` file is available, you can alternatively use:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

## Step 1 — Process the Dataset

Run:

```bash
python scripts/clean_and_process.py
```

This prepares the raw datasets and generates the processed feature-engineered files.

---

## Step 2 — Train the Models

Run:

```bash
python src/train.py
```

The training pipeline produces the trained model files inside:

```text
models/
```

---

## Step 3 — Evaluate the Models

Run:

```bash
python src/eval_models.py
```

This evaluates the trained models and generates comparison metrics.

---

## Step 4 — Start the Dashboard

Run:

```bash
streamlit run app.py
```

The Streamlit interface will open in your browser.

---

# 📈 Model Evaluation

The project compares the forecasting models using standard regression metrics, including:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error
* R² — Coefficient of Determination
* MAPE — Mean Absolute Percentage Error

The comparison results are stored in:

```text
models/model_comparison.json
```

---

# 🔐 Data & API Considerations

The project uses Open-Meteo for live environmental information.

Internet access is required when retrieving live data.

Historical datasets can still be processed and used for model training without requiring a paid AI API key.

---

# 🎯 Project Objectives

The primary objectives of this project are:

1. Forecast future AQI using deep-learning models.
2. Analyze historical air-pollution patterns.
3. Integrate live environmental information.
4. Calculate Indian AQI categories.
5. Identify dominant pollutants.
6. Compare different deep-learning architectures.
7. Provide personalized environmental health guidance.
8. Present results through an interactive dashboard.

---

# 👨‍💻 Author

**Narendra Mandada**

Repository:

**https://github.com/NarendraMandada77/AI-Air-Quality-Prediction**

# 📜 License

This project is available under the **MIT License**.
