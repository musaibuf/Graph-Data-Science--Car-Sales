# Importing necessary libraries
import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import joblib

# Function to fetch node embeddings and labels from Neo4j
def fetch_embeddings_from_neo4j(uri, user, password, query):
    # Establishing a connection with the Neo4j database
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Running the query in a session and fetching the data
    with driver.session() as session:
        result = session.run(query)
        data = result.data()
    
    # Closing the connection
    driver.close()
    
    # Returning the data as a DataFrame
    return pd.DataFrame(data)

# Function to fetch a specific embedding from Neo4j
def fetch_single_embedding(uri, user, password, vin):
    # Establishing a connection with the Neo4j database
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # Query to fetch the embedding of a specific vehicle
    query = f"""
    MATCH (v:Vehicle {{ vin: '{vin}' }})
    RETURN v.embedding AS embedding
    """
    
    # Running the query in a session and fetching the data
    with driver.session() as session:
        result = session.run(query)
        data = result.single()
    
    # Closing the connection
    driver.close()
    
    # Returning the embedding if found, else None
    return data['embedding'] if data else None

# Neo4j connection settings
uri = "bolt://localhost:7687"
user = "neo4j"
password = "12345678"

# Query to fetch the vehicle embeddings and colors
query = """
MATCH (v:Vehicle)
RETURN v.vin AS id, v.embedding AS embedding, v.color AS color
"""

# Fetching data from Neo4j
df = fetch_embeddings_from_neo4j(uri, user, password, query)

# Expanding the embeddings from list to columns
embeddings_df = pd.DataFrame(df['embedding'].tolist())
embeddings_df['color'] = df['color']

# Cleaning the data to remove null colors
embeddings_df = embeddings_df.dropna(subset=['color'])

# Splitting the data into features (X) and target (y)
X = embeddings_df.drop('color', axis=1)
y = embeddings_df['color']

# Using RandomOverSampler to oversample minority classes
oversampler = RandomOverSampler(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Splitting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Creating and training a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluating the model with a classification report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=1))

# Saving the trained model for later use
joblib.dump(model, 'best_vehicle_color_model.joblib')

# Function for new predictions
def predict_vehicle_color(model, new_embedding):
    # Predicting the color using the model and the new embedding
    return model.predict([new_embedding])

# Predicting the color of a specific vehicle using a valid embedding vector from Neo4j
vin_to_predict = '5xyktca69fg566472'  
new_vehicle_embedding = fetch_single_embedding(uri, user, password, vin_to_predict)

# If an embedding is found, load the model and predict the color
if new_vehicle_embedding:
    loaded_model = joblib.load('best_vehicle_color_model.joblib')
    predicted_color = predict_vehicle_color(loaded_model, new_vehicle_embedding)
    print('Predicted color:', predicted_color)
else:
    print(f'No embedding found for VIN: {vin_to_predict}')
