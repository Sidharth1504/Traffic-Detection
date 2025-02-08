import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model():
    try:
        with open("congestion_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: congestion_data.json not found. Make sure you run main.py first.")
        return
    except json.JSONDecodeError:
        print("Error: congestion_data.json is not a valid JSON file.")
        return

    # Prepare data for ML training.
    model_data = []
    for inter_no, time_data in data.items():
        for time_slot, congestion_values in time_data.items():
            # Use the average congestion or a sample of congestion based on the time that you have
            avg_congestion = sum(congestion_values) / len(congestion_values)
            model_data.append({
                "intersection": inter_no,
                "time_slot": time_slot,
                "congestion": avg_congestion
            })
   
    #Convert to what the model can take into account
    df = pd.DataFrame(model_data)
     # Example: Extracting hour and minute as features
    df['hour'] = df['time_slot'].apply(lambda x: int(x.split(':')[0]))
    df['minute'] = df['time_slot'].apply(lambda x: int(x.split(':')[1]))
    #The higher the congestions means that you have to pass through other congestion, so we are creating that mapping
    X = df[["congestion","hour", "minute"]]
    y = df["congestion"]
    #Splitting the test to verify that the prediction is valid or not
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #Making the data structure the model can understand, it seems it has high variance.
    model = RandomForestClassifier(n_estimators =10, random_state = 42)
    #Fit all the data
    model.fit(X_train, y_train)
    #Make the test case to verify that the prediction is valid or not
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    #Save to current directory so that we can retrieve to main
    joblib.dump(model,"traffic_model.pkl")
    print ("Traffic model dumped")

if __name__ == "__main__":
    train_model()
