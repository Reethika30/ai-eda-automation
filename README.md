# 🧠 AI-Powered EDA Automation Tool

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

## 🎯 Problem
Every new dataset requires the same EDA steps: profiling, distributions, correlations, missing values. It's repetitive and time-consuming.

## 💡 Solution
A Python tool that takes any CSV, auto-generates profiling stats, distribution plots, correlation heatmaps, and a **plain-English summary using GPT-4 API** explaining key findings and recommended next steps.

## 🏗 Pipeline
```
Upload CSV → Auto-Profile → Generate Plots → GPT-4 Summary → HTML Report
```

## 🚀 How to Run
```bash
pip install pandas numpy matplotlib seaborn openai
export OPENAI_API_KEY=your_key
python src/auto_eda.py data/sample_dataset.csv
# Output: output/eda_report.html
```

## 📊 What It Generates
- Dataset shape, dtypes, memory usage
- Missing value heatmap
- Distribution plots for all numeric columns
- Correlation matrix heatmap
- Top categorical value counts
- GPT-4 plain-English summary of findings
- Recommended next steps for analysis

## 📜 License
MIT
