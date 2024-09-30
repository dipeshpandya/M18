# Project to predict the likelihood that an applicant will repay their student loans
## Clustering of Crypto Currency Data

The project demonstrates use of machine learning models and neural networks.

Libraries required
```python 
# Imports
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
```
This code imports various layers and models of tensorflow in addition to sklearn scaler and metrics


Next, following code imports data for training Neural networks
```python
# Read the csv into a Pandas DataFrame
file_path = "https://static.bc-edx.com/ai/ail-v-1-0/m18/lms/datasets/student-loans.csv"
loans_df = pd.read_csv(file_path)

# Review the DataFrame
loans_df.head()
```

```python
# Review the data types associated with the columns
loans_df.dtypes
```
The code shows data types for all columns for dataframe

payment_history           float64
location_parameter        float64
stem_degree_score         float64
gpa_ranking               float64
alumni_success            float64
study_major_code          float64
time_to_completion        float64
finance_workshop_score    float64
cohort_ranking            float64
total_loan_score          float64
financial_aid_score       float64
credit_ranking              int64

Having imported preprocessed data, following code will create target (`y`) datasets. 

```python
# Check the credit_ranking value counts
loans_df["credit_ranking"].value_counts()

# Define the target set y using the credit_ranking column
y = loans_df["credit_ranking"]

# Display a sample of y
y
```
credit_ranking
1    855
0    744

```python
# Define features set X by selecting all columns but credit_ranking

X = loans_df.drop("credit_ranking", axis=1)
# Review the features DataFrame
```
This code will create data for variable X by excluding target column y

### Spliting the features and target sets into training and testing datasets and scaling training data

```python
# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_train_scaled = scaler.fit_transform(X_train)

# Fit the scaler to the features training dataset
X_test_scaled = scaler.fit_transform(X_test)

### Creatinh a deep neural network by assigning the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras.

# Define the the number of inputs (features) to the model
input_features = len(X.columns)

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 =  6

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 = 3

# Define the number of neurons in the output layer
output_neuron=1

# Create the Sequential model instance
nn_model = tf.keras.Sequential()

# Add the first hidden layer
nn_model.add(
  tf.keras.layers.Dense(units=hidden_nodes_layer1, 
                        input_dim=input_features,
                        name = "H1",
                        activation="relu")
)

# Add the second hidden layer
nn_model.add(
    tf.keras.layers.Dense(units=hidden_nodes_layer2,
                          name = "H2",
                          activation="relu")
)

# Add the output layer to the model specifying the number of output neurons and activation function
nn_model.add(tf.keras.layers.Dense(units=1, name="output", activation="sigmoid"))

# Display the Sequential model summary
nn_model.summary()
```

### Now we will compile and fit the model using the `binary_crossentropy` loss function, the `adam` optimizer, and the `accuracy` evaluation metric.
```python
# Compile the Sequential model
nn_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model using 50 epochs and the training data
nn_fit_model = nn_model.fit(X_train_scaled, y_train, epochs=50)

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled,y_test,verbose=2)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```
The code will produce this result
13/13 - 0s - 18ms/step - accuracy: 0.7425 - loss: 0.5056
Loss: 0.5055967569351196, Accuracy: 0.7425000071525574

### Save and export your model to a keras file, and name the file `student_loans.keras`.
```python
# Set the model's file path
file_path = Path("saved_model/student_loans.keras")


# Export your model to a keras file
nn_model.save(file_path)
```

Having said the model, let us predict loan repayment success by Using your Neural Network Model in 4 steps
### Step 1: Reload your saved model.
### Step 2: Make predictions on the testing data and save the predictions to a DataFrame.
### Step 3: Save the predictions to a DataFrame and round the predictions to binary results
### Step 4: Display a classification report with the y test data and predictions

```python
#Reload the model using set model's file path
file_path = Path("saved_model/student_loans.keras")

# Load the model to a new object
nn_model_imported = tf.keras.models.load_model(file_path)

# Make predictions with the test data
prediction = nn_model_imported.predict(X_test_scaled)


# Display a sample of the predictions
display(prediction[0:5])

# Save the predictions to a DataFrame and round the predictions to binary results
predictions_df = pd.DataFrame(columns=["predictions"], data=prediction).round()
predictions_df

# Print the classification report with the y test data and predictions
print(classification_report(y_test, predictions_df["predictions"].values))


```


