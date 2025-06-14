# Career Modeling Project | 职业建模项目

This document introduces three advanced-level data modeling strategies applied to job market analysis, using NLP, regression, and similarity models.

本文介绍三种用于职业市场分析的建模策略，结合自然语言处理、回归预测与匹配度建模技术。

---

##  Model A: Keyword Topic Modeling + Clustering  
### 模型A：关键词主题建模 + 聚类分析

**Objective | 目标**:  
Use LDA (Latent Dirichlet Allocation) to extract topics from job descriptions, and apply KMeans clustering to categorize job types.

**技术路径**:
- Use `CountVectorizer` to build word frequency matrix.
- Apply `LatentDirichletAllocation` to model topics.
- Use `KMeans` to cluster job posts based on topic distribution.

**输出 | Outputs**:
- Each job post is assigned a dominant topic and a cluster.
- CSV file: `model_A_lda_kmeans_output.csv`

**应用 | Use Cases**:
- Job grouping
- Skill similarity exploration
- Job title generation

---

##  Model B: Salary Prediction with Regression  
### 模型B：岗位薪资回归预测

**Objective | 目标**:  
Predict the average salary of job postings using their keyword vectors and industry metadata.

**技术路径**:
- Use `TfidfVectorizer` to extract keyword features.
- Use `LinearRegression` to model salary.
- Evaluate prediction error and coefficients.

**输出 | Outputs**:
- Predicted salary for each job.
- CSV file: `model_B_salary_prediction.csv`

**应用 | Use Cases**:
- Evaluate skill-salary correlation.
- Estimate salary potential for a skill portfolio.

---

##  Model C: Multi-Specialty Job Matching  
### 模型C：多专业岗位匹配度建模

**Objective | 目标**:  
Match multiple academic majors to suitable jobs using cosine similarity between keyword vectors.

**技术路径**:
- Use `TfidfVectorizer` on job descriptions.
- Create major-specific keyword vectors.
- Compute similarity scores using `cosine_similarity`.

**输出 | Outputs**:
- Matching scores for each major-job pair.
- CSV file: `model_C_specialty_matching.csv`

**应用 | Use Cases**:
- Recommend jobs to students by major.
- Curriculum relevance analysis.

---

##  Output Files Summary | 输出文件汇总

| File Name | Description | 文件说明 |
|-----------|-------------|-----------|
| `model_A_lda_kmeans_output.csv` | Topic + Cluster results | 主题与聚类结果 |
| `model_B_salary_prediction.csv` | Predicted salaries | 薪资预测结果 |
| `model_C_specialty_matching.csv` | Major-job match scores | 专业岗位匹配度评分 |

---

##  Run Instructions | 使用说明

1. Place the JSON job data files in the same directory.
2. Install required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

3. Run the models:

```bash
python Model_A_Advanced_LDA_KMeans.py
python Model_B_Advanced_Salary_Prediction.py
python Model_C_Advanced_Specialty_Matching.py
```

---