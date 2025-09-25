# train.py
from sklearn.tree import DecisionTreeRegressor
from misc import load_data, preprocess_data, train_model, evaluate_model

# Main execution
if __name__ == "__main__":
    # 1. Load data
    df = load_data()

    # 2. Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # 3. Initialize model
    dt_model = DecisionTreeRegressor(random_state=42)

    # 4. Train the model
    trained_dt_model = train_model(dt_model, X_train, y_train)

    # 5. Evaluate the model
    evaluate_model(trained_dt_model, X_test, y_test)
