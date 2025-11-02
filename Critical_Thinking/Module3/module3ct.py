import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing  # replacement for load_boston


def main():

    print("------ Module 3: Linear Regression ------")
    print("\nThis program predicts house prices based on various features using Linear Regression.")
    print("Predicted prices will be compared to actual prices in a scatter plot.\n")

    # Load the California housing dataset
    housing = fetch_california_housing()

    # The three below lines were used to help visualize and understand the dataset. Uncomment if needed.
    #df = pd.DataFrame(housing.data, columns=housing.feature_names)
    #df['PRICE'] = housing.target
    #print(df.head())

    X = pd.DataFrame(housing.data, columns=housing.feature_names)
    y = pd.Series(housing.target, name='PRICE')

    # Split data into training and test sets. Randomstate=525 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=525)

    # Initialize and fit the Linear Regression model
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Evaluate model performance using R squared score
    r2_score = lr.score(X_test, y_test)
    print(f"\nModel training complete. R^2 score on test data: {r2_score * 100:.2f}%\n")

    # Plotting actual vs predicted prices
    y_pred = lr.predict(X_test)
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Actual vs Predicted House Prices (ideal prediction line in red)")
    # Draw ideal prediction line
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
    plt.show()


if __name__ == "__main__":
    main()
