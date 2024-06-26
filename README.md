# Graph-Data-Science--Car-Sales

# Vehicle Sales Network Analysis and Prediction

This repository hosts the project for CS343: Graph Data Science at Habib University, Spring 2024. The project involves constructing a graph database using Neo4j to model and analyze vehicle sales data, with an emphasis on link prediction and node classification through machine learning techniques.

## Project Overview

The project explores the relationships between sellers and vehicle sales across various states, using graph-based analytics to predict and classify data points within a Neo4j graph database. Insights derived from this project aim to predict potential vehicle sales and seller activities, enhancing understanding of market dynamics.

## Structure

- **Data Model**
  - `Data Model.jpeg` - Contains the data model diagram.
  
- **Data Loading Script**
  - `DataLoadingScript` - Script for loading data into the database.

- **DataSet**
  - `DataSet` - The dataset used in this project.

- **Graph Analytical Queries**
  - `Graph Analytical Queries` - Contains queries for graph analytics.
  
- **Graph Analytical Snippets**
  - `Graph Analytical Snippets.pdf` - PDF containing snippets of graph analytical queries.

- **Graph Statistical Queries**
  - `Graph Statistical Queries` – Contains statistical queries for graphs.
  
- **Graph Statistical Snippets**
  – `Graph Statistical Snippets.pdf` – PDF containing snippets of graph statistical queries.
  
## Setup Instructions

### Prerequisites

- Python 3.8+
- Neo4j 4.0+ with APOC plugin
- Virtual environment recommended (optional)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/vehicle-sales-prediction.git
   cd vehicle-sales-prediction

### Install Dependencies:
pip install pandas neo4j sklearn imblearn joblib

### Run the Script:
python color_prediction.py

# Color Prediction
The `color_prediction.py` file predicts the color of a vehicle based on its node embeddings in a Neo4j graph database. The code uses a RandomForestClassifier model for the prediction. The script fetches the vehicle embeddings and colors from the database, preprocesses the data, splits the data into features and target, oversamples minority classes using RandomOverSampler, splits the data into training and testing datasets, creates and trains a RandomForestClassifier model, evaluates the model with a classification report, saves the trained model for later use, and predicts the color of a specific vehicle using a valid embedding vector from Neo4j.
