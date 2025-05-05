# Insights from Apache JIRA Issue Tracking

## Project Overview

This project leverages the Apache JIRA dataset to analyze software issue tracking patterns, resolution trends, comment sentiment, and bug behaviors. By applying big data analytics and machine learning methods, the project generates actionable insights aimed at enhancing software development practices, team collaboration, and customer satisfaction.

## Project Objectives

### 1. Issue Tracking and Resolution Analysis

* Identified patterns in issue types, priority levels, resolution times, and status transitions using metadata from JIRA issues.
* Examined key factors affecting resolution timelines to facilitate better project management.

### 2. Sentiment and Comment Analysis

* Performed detailed text preprocessing including cleaning, tokenization, and removal of stopwords from user comments.
* Implemented sentiment classification models (rule-based and logistic regression) to evaluate community interactions and sentiment distributions.
* Conducted topic modeling (e.g., LDA) to uncover underlying themes in comment discussions.

### 3. Advanced Bug Analysis

* Developed engineered features such as priority, comment frequency, and resolution duration to predict the likelihood of bug reopening.
* Utilized clustering algorithms (e.g., KMeans) to identify prevalent bug patterns and thematic clusters.

### 4. Workflow and Collaboration Insights

* Analyzed changelog data and user comments to explore team collaboration dynamics, frequently altered fields, and interdependencies among issues.

## Dataset

* **Name**: Apache JIRA Issues
* **Size**: 8.78 GB
* **Format**: CSV
* **Files Included**: Issue metadata, Changelogs, Comments, and Issue Links
* **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues)

## Tools & Technologies

* **Big Data Tools**: Hadoop 3.1.2, Apache Spark 3.0.2
* **Languages and Frameworks**: Python, PySpark
* **Cluster Setup**: 5-node Hadoop cluster (2 master nodes, 3 data nodes) with Yarn Resource Manager
