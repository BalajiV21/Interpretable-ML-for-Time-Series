# 🕵️‍♂️ Post-Hoc Interpretability in Time-Series Classification 

Project by **Balaji Viswanathan**, Sana Begum, Prajuvin Prabha, Sarim Ali, and Kausar Ali Ansari.

---

## 🔍 Project Overview

In safety-critical domains (healthcare, finance, etc.), **model interpretability** is as important as accuracy. We explore whether *post-hoc* attribution methods can reliably explain and validate deep-learning models on a benchmark time-series dataset.

- **Dataset**: Banknote Authentication (treated here as an ECG-style 1D time series)  
- **Models**:  
  - 1D Convolutional Neural Network (CNN)  
  - Long Short-Term Memory network (LSTM)  
- **Attribution Methods**:  
  1. Integrated Gradients  
  2. DeepLift  
  3. GradientShap  
  4. KernelShap  

---

## ⚙️ Installation & Usage

1. **Clone** this repo  
   ```bash
   git clone https://github.com/YourUsername/TimeSeries-Interpretability.git
   cd TimeSeries-Interpretability
