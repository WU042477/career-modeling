
#  Data Field Description | 数据字段说明

This document describes the fields used across the job JSON datasets.

本文介绍各岗位JSON数据中的关键字段定义。

| Field Name               | 中文名称         | Description |
|--------------------------|------------------|-------------|
| `job_name`              | 岗位名称         | Job title |
| `company_name`          | 公司名称         | Name of the employer |
| `salary_avg`            | 平均薪资         | Average monthly salary |
| `location`              | 工作地点         | City or region |
| `degree`                | 学历要求         | Minimum education level required |
| `Word segmentation_string` | 关键词字符串     | Cleaned job description keywords |
| `industry`              | 行业类别         | Industry tag (Education, Culture, etc.) |

- All fields are extracted from web-scraped job datasets, manually cleaned.
- `Word segmentation_string` is used as NLP input for all modeling.

