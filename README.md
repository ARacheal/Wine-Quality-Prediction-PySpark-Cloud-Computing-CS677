# Cloud Computing — Wine Quality Prediction with PySpark & AWS

**Course:** CS677 Cloud Computing — NJIT  
**Tools:** Python, PySpark, Apache Spark, AWS S3, Docker, Scikit-learn

## Overview
Built a distributed machine learning pipeline to predict wine quality using 
Apache Spark on cloud infrastructure. The model trains on data stored in AWS S3, 
processes it using PySpark's distributed computing framework, and classifies 
wine quality scores using a Random Forest algorithm.

## What This Does
- Reads training and validation datasets directly from **AWS S3**
- Preprocesses and assembles features using **PySpark MLlib**
- Trains a **Random Forest Classifier** (21 trees, max depth 30, Gini impurity)
- Evaluates model performance using **F1-score and accuracy**
- Containerized using **Docker** for reproducible cloud deployment

## Results
- Random Forest with 21 trees trained on 70/30 train-test split
- Evaluated with F1-score (micro average) and classification accuracy

## Files
| File | Description |
|------|-------------|
| `TrainSet_AR2538.py` | Model training script using PySpark on AWS S3 data |
| `ValidateSet_AR2538.py` | Validation and evaluation script |
| `TrainingDataset.csv` | Wine quality training data |
| `ValidationDataset.csv` | Wine quality validation data |
| `AR2538_Dockerfile.txt` | Docker container configuration |
| `AR2538-Report-Antonita-Racheal-CC.pdf` | Full technical report |

## Tech Stack
- **Apache Spark / PySpark** — distributed data processing
- **AWS S3** — cloud data storage
- **PySpark MLlib** — Random Forest classifier
- **Docker** — containerization
- **Python** — Pandas, NumPy, Scikit-learn for evaluation
