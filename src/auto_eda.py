"""Automated EDA tool with optional GPT-4 summary."""
import pandas as pd
import numpy as np
import sys
import json
from datetime import datetime

class AutoEDA:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.filepath = filepath
        self.report = {}

    def profile(self):
        df = self.df
        self.report['shape'] = {'rows': len(df), 'columns': len(df.columns)}
        self.report['dtypes'] = df.dtypes.astype(str).to_dict()
        self.report['missing'] = df.isnull().sum().to_dict()
        self.report['missing_pct'] = (df.isnull().mean() * 100).round(2).to_dict()
        self.report['numeric_stats'] = json.loads(df.describe().to_json())
        self.report['memory_mb'] = round(df.memory_usage(deep=True).sum() / 1024**2, 2)

        # Correlations for numeric columns
        numeric = df.select_dtypes(include=[np.number])
        if len(numeric.columns) >= 2:
            corr = numeric.corr()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.7:
                        high_corr.append({
                            'col1': corr.columns[i], 'col2': corr.columns[j],
                            'correlation': round(val, 3)
                        })
            self.report['high_correlations'] = high_corr

        # Categorical summaries
        cats = df.select_dtypes(include=['object'])
        cat_summary = {}
        for col in cats.columns:
            cat_summary[col] = {
                'unique': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        self.report['categorical'] = cat_summary
        return self

    def generate_prompt(self):
        """Generate a prompt for GPT-4 to summarize findings."""
        prompt = f"""You are a data analyst. Summarize the following EDA results in plain English.
        
Dataset: {self.filepath}
Rows: {self.report['shape']['rows']}, Columns: {self.report['shape']['columns']}
Memory: {self.report['memory_mb']} MB

Missing Values: {json.dumps({k:v for k,v in self.report['missing_pct'].items() if v > 0})}
High Correlations: {json.dumps(self.report.get('high_correlations', []))}
Categorical Columns: {json.dumps({k: v['unique'] for k, v in self.report['categorical'].items()})}

Provide:
1. A 3-sentence summary of the dataset
2. Top 3 findings or red flags
3. Recommended next steps for analysis"""
        return prompt

    def print_report(self):
        print(f"\n{'='*60}")
        print(f"📊 AUTO EDA REPORT — {self.filepath}")
        print(f"{'='*60}")
        print(f"Shape: {self.report['shape']['rows']} rows × {self.report['shape']['columns']} columns")
        print(f"Memory: {self.report['memory_mb']} MB")
        print(f"\nMissing Values:")
        for col, pct in self.report['missing_pct'].items():
            if pct > 0:
                print(f"  ⚠️ {col}: {pct}%")
        if not any(v > 0 for v in self.report['missing_pct'].values()):
            print(f"  ✅ No missing values")
        if self.report.get('high_correlations'):
            print(f"\nHigh Correlations (|r| > 0.7):")
            for c in self.report['high_correlations']:
                print(f"  📈 {c['col1']} ↔ {c['col2']}: {c['correlation']}")
        print(f"\nCategorical Columns:")
        for col, info in self.report['categorical'].items():
            print(f"  {col}: {info['unique']} unique values")
        print(f"\n💡 GPT-4 Prompt (send to OpenAI API for plain-English summary):")
        print(f"{'─'*40}")
        print(self.generate_prompt()[:500] + "...")
        print(f"\n✅ EDA complete. In production, GPT-4 generates the final summary.")

if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'data/sample_dataset.csv'
    eda = AutoEDA(filepath)
    eda.profile()
    eda.print_report()
