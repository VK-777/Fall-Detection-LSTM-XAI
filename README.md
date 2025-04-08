# Fall Detection Using LSTM with Explainable AI (XAI)

This project implements a fall detection system using Long Short-Term Memory (LSTM) neural networks trained on motion sensor data. 
The model is evaluated on the **KFall** dataset and compared with the **MobiFall** dataset. To enhance model interpretability, explainable AI (XAI) techniques are also applied.

---

## 📁 Project Structure
<pre>
├── Constants
    └── constants.py
├── Controller
    └── process_all_mobifall.py
    └── process_data_kfall.py
├── Dataset
    └── K-fall
    └── Mobifall_Dataset_v2.0
├── Services
    └── XAI_K_Fall.py
    └── XAI_Mobifall_LSTM.py
├── Utils
    └── Combine.py
    └── Data_loader.py
    └── Interpolate.py
├── main.py # Main script for training/testing 
├── requirements.txt # Project dependencies 
├── README.md # Project documentation
</pre>


---

## 🔍 Features

- 🧠 LSTM-based deep learning model for fall detection
- 🔁 Sensor data preprocessing and interpolation
- 📊 Evaluation and comparison with the MobiFall dataset
- 🧬 Model interpretability using Explainable AI techniques (XAI)
- 📦 Modular and extensible code structure

---

## 🧪 Datasets Used

- **[KFall Dataset (PhysioNet)](https://physionet.org/content/kfall/1.6.0/)**  
- **[MobiFall Dataset](https://www.mobilab.unina.it/mobifall/)**

> 📌 Make sure the datasets are placed in the correct folders before running.

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/VK-777/Fall-Detection-LSTM-XAI.git
cd Fall-Detection-LSTM-XAI
pip install -r requirements.txt
```

---

## 🚀 How to Run
``` bash
python main.py
```
