
# Exploratory Data Analysis of Student Performance

## 1. Project Overview
This project explores student examination scores to identify patterns, compare groups, and examine relationships among performance measures.

## 2. Objectives
- Inspect dataset structure and data quality.
- Calculate statistical summaries.
- Analyze score distributions.
- Compare scores across gender and test preparation groups.
- Examine differences by parental education and race/ethnicity.
- Explore correlations among mathematics, reading, and writing scores.

## 3. Dataset
- Records: 1,000
- Columns: 8
- Missing values: 0
- Duplicate rows: 0

### Features
- Gender
- Race/ethnicity
- Parental level of education
- Lunch
- Test preparation course
- Math score
- Reading score
- Writing score

## 4. Tools and Libraries
- Python
- Pandas
- Matplotlib
- Seaborn

## 5. Key Findings

### Subject Performance
- Mean mathematics score: 66.09
- Mean reading score: 69.17
- Mean writing score: 68.05

Reading had the highest average score, while mathematics had the lowest.

### Gender Comparison
Female students had higher average reading and writing scores in this dataset, while male students had a higher average mathematics score.

### Test Preparation
Students who completed the test preparation course had higher average scores in all three subjects than students who did not complete it.

The average writing score was 74.42 for students who completed preparation and 64.50 for those who did not.

These are observed associations and do not establish causation.

## 6. Visualizations
Graphs are saved in the `visualizations` folder:
- Mathematics score distribution
- Reading score distribution
- Writing score distribution
- Correlation heatmap
- Writing scores by test preparation
- Average scores by lunch type
- Average scores by parental education
- Average scores by race/ethnicity

## 7. How to Run

Install dependencies:

```bash
pip install pandas matplotlib seaborn
```

Run the analysis from the project root:

```bash
python eda_analysis.py
```

Ensure the dataset is located at `data/StudentsPerformance.csv`.

## 8. Conclusion
The analysis identified differences in average examination scores across subjects and student groups. Visualizations and statistical summaries provide a foundation for further investigation into student performance.

The findings describe this dataset and should not be interpreted as proof that any single demographic or educational factor causes higher or lower scores.
