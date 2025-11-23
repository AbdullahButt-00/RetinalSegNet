# 🩺 Retinal Vessel & Optic Disc Segmentation

### *TransFuse + CBAM · Streamlit Application*

This application performs **retinal blood vessel (BV)** and **optic disc
(OD)** segmentation using a lightweight **TransFuse architecture**
enhanced with **CBAM attention**.\
It supports two datasets:

-   **DRIVE**
-   **IOSTAR**

The app provides:

-   Real-time segmentation\
-   Automatic model weight downloading\
-   Prediction quality metrics\
-   Organized results saving\
-   Model benchmarking utilities

------------------------------------------------------------------------

## 📁 Project Structure

    APP/
    │── app.py                # Main Streamlit App
    │── requirements.txt
    │── weights/              # Auto-downloaded model weights & metrics
    │── utils/
    │     ├── model_utils.py
    │     ├── inference_engine.py
    │     ├── weight_manager.py
    │     └── data_utils.py
    │── data/  
    │── logs/                 # Logging & debug files

------------------------------------------------------------------------

## ⚙️ Installation & Running

### 1️⃣ Create Virtual Environment

``` bash
python3 -m venv venv
```

### 2️⃣ Activate Environment

``` bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt --default-timeout=2000
```

### 4️⃣ Run the Application

``` bash
python -m streamlit run app.py
```
