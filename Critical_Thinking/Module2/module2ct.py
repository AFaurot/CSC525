import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def user_inputs():
    invalid_input = False
    # Get user input for age
    age = float(input("Enter age: "))
    # Get user input for height
    height = float(input("Enter height (inches): "))
    # Get user input for weight
    weight = float(input("Enter weight (pounds): "))
    # Get user input for gender 0 for female 1 for male
    # Gender choice is a float due to the assignment requirements but should be int ideally
    gender_choice = float(input("Select your gender\n0 Female\n1 Male\n"))
    if gender_choice not in [0, 1]:
        print("invalid_input")
        invalid_input = True
    while invalid_input:
        gender_choice = float(input("Select your gender\n0 Female\n1 Male\n"))
        if gender_choice in [0, 1]:
            invalid_input = False
        else:
            print ("invalid_input")
    return age, height, weight, gender_choice


def main():
    # set random seed CSC525!
    np.random.seed(525)
    print("------ Module 2: K-Nearest Neighbors Classifier ------")
    print("\n This program classifies video game preferences based on weight, height, age, and gender.\n")

    # Load dataset
    data = pd.read_csv('data.csv', header=None, names=['age', 'height', 'weight', 'gender', 'preference'])
    X = data[['age', 'height', 'weight', 'gender']]
    y = data['preference']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)

    # Evaluate model accuracy
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    print(f"\nModel training complete. Accuracy on test data: {score * 100:.2f}%\n")

    choice = input("Do you want to proceed? (y/n): ")
    if choice.lower() != 'y':
        print("Exiting the program.")
        return
    else:
        while choice.lower() == 'y':
            age, height, weight, gender = user_inputs()
            user_data = [[age, height, weight, gender]]
            # Create DataFrame for user input and predict preference
            user_df = pd.DataFrame(user_data, columns=['age', 'height', 'weight', 'gender'])
            prediction = knn.predict(user_df)
            print(f"Predicted preference: {prediction[0]}")
            choice = input("Do you want to classify another person? (y/n): ")


if __name__ == "__main__":
    main()
