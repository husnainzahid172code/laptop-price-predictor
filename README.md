# AI/ML Internship — Project 01: Laptop Price Data Analysis

Python data analysis and visualization project built for Task 01 of the AI/ML Internship (Devixo Solutions). Covers data loading, cleaning, exploratory data analysis, visualization, and insight extraction on a real-world laptop pricing dataset.

## 📌 Objective

Build a strong foundation in Python programming and data analysis by loading a real-world dataset, cleaning it, performing exploratory data analysis (EDA), visualizing the data, and drawing meaningful, business-relevant insights.

## 📊 Dataset

**Laptop Price Dataset** (Kaggle) — 1,303 laptop listings with brand, specifications, and price in Euros.

- Source: [Kaggle — Laptop Price](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
- Rows: 1,303 (1,275 after removing duplicates)
- Target variable: `Price_Euros`

## 🛠️ Technologies

- Python 3
- Jupyter Notebook
- Pandas, NumPy
- Matplotlib, Seaborn

## 📁 Repository Structure

```
├── Laptop_Price_Analysis.ipynb     # Main notebook (Parts 1–5)
├── laptops_raw.csv                 # Original raw dataset
├── laptops_clean.csv               # Cleaned dataset (output of Part 2)
├── Laptop_Price_Analysis_Report.pdf # Full PDF report
├── images/                         # All 11 exported visualizations
├── screenshots/                    # Code & output screenshots
└── README.md
```

## 🔍 Project Workflow

1. **Data Loading** — Import libraries, load CSV, inspect shape/structure.
2. **Data Cleaning** — Handle missing values, remove duplicates, fix data types (units embedded in text like `"8GB"`, `"1.37kg"`, European comma-decimal prices), rename confusing columns.
3. **Feature Engineering** — Extract CPU brand/speed, GPU brand, total storage (GB), storage type, touchscreen/IPS flags from free-text fields.
4. **Exploratory Data Analysis** — Mean, median, mode, standard deviation, and correlation matrix.
5. **Visualization** — 11 charts: histogram, bar chart, pie chart, scatter plot, line chart, box plot, heatmap, count plot, pair plot, violin plot, and a bonus multi-variable scatter.
6. **Insights** — Key business questions answered (top price driver, most/least expensive brand & category, touchscreen premium, storage-type premium, etc.).

## 📈 Key Insights

- **RAM capacity** is the strongest numeric predictor of price (correlation ≈ 0.74).
- **CPU speed** is the second-strongest driver (correlation ≈ 0.43).
- **Screen size** has almost no linear effect on price (correlation ≈ 0.07).
- **Razer** is the most expensive brand on average (~€3,346); **Vero** is the cheapest (~€217).
- **Touchscreen laptops** cost ~35% more on average than non-touchscreen models.
- **SSD-only laptops** average more than double the price of HDD-only laptops.

See the full [PDF report](Laptop_Price_Analysis_Report.pdf) or the [notebook](Laptop_Price_Analysis.ipynb) for the complete analysis, statistics, and all visualizations.

## ▶️ How to Run

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook Laptop_Price_Analysis.ipynb
```

## Author

Husnain — Management & Business Computing, Beaconhouse National University (BNU)
