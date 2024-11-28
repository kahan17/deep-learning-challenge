# **Neural Network Model Report for Alphabet Soup**

## **Overview of the Analysis**
The purpose of this analysis is to design, train, and optimize a deep learning neural network model to assist **Alphabet Soup**, a philanthropic organization, in predicting the success of funding applications. By leveraging machine learning techniques, the model classifies whether an application will be successful based on various input features. The ultimate goal is to achieve a predictive accuracy of **75% or higher**.

---

## **Results**

### **Data Preprocessing**

- **Target Variable:**
  - The target variable for this model is `IS_SUCCESSFUL`, which indicates whether the funding application was successful (`1`) or not (`0`).

- **Features:**
  - The input features for the model include:
    - `APPLICATION_TYPE`
    - `AFFILIATION`
    - `CLASSIFICATION`
    - `USE_CASE`
    - `ORGANIZATION`
    - `STATUS`
    - `INCOME_AMT`
    - `SPECIAL_CONSIDERATIONS`
    - `ASK_AMT` (log-transformed for normalization)

- **Removed Variables:**
  - The following columns were removed as they were neither targets nor meaningful features:
    - `EIN` (unique identifier)
    - `NAME` (organization name, not predictive)

---

### **Compiling, Training, and Evaluating the Model**

- **Neural Network Architecture:**
  - **Input Layer:** 
    - The input layer received the scaled and one-hot-encoded features.
  - **Hidden Layers:**
    - First Hidden Layer: 256 neurons with a ReLU activation function.
    - Second Hidden Layer: 128 neurons with a ReLU activation function.
    - Third Hidden Layer: 64 neurons with a ReLU activation function.
    - A **Dropout layer** with a rate of 0.2 was added to prevent overfitting.
  - **Output Layer:**
    - 1 neuron with a Sigmoid activation function for binary classification.

- **Compilation and Training:**
  - Optimizer: `Adam`
  - Loss Function: `Binary Crossentropy`
  - Metrics: `Accuracy`
  - Epochs: Initially set to 300 with early stopping after 5 epochs of no improvement.

- **Model Performance:**
  - The model achieved:
    - **Training Accuracy:** ~74.7%
    - **Test Accuracy:** ~72.9%
    - **Loss:** 0.6096
  - Although the model showed consistent performance, it fell slightly short of the **75% accuracy goal**.

- **Steps to Improve Model Performance:**
  - Applied a **log transformation** on the `ASK_AMT` feature to reduce skewness.
  - Combined low-frequency categories in `APPLICATION_TYPE` and `CLASSIFICATION` into an `"Other"` category to simplify data representation.
  - Increased the number of neurons in hidden layers to capture more complex patterns.
  - Added a **Dropout layer** to reduce overfitting.
  - Experimented with different optimizers (`Adam`, `RMSprop`) and adjusted batch sizes (32 and 64).

---

## **Summary**

The deep learning model demonstrated reasonable performance with a test accuracy of **72.9%**, just below the target of **75%**. While further tuning of hyperparameters and feature engineering could potentially improve performance, the results suggest that this classification problem may be more effectively addressed by alternative approaches:

### **Recommendation**
- Consider using an **ensemble model** combining algorithms such as Random Forest or Gradient Boosting (e.g., XGBoost) for feature importance analysis and classification.
- These models are better at handling categorical data and may achieve higher accuracy with less preprocessing effort.
- Additionally, hyperparameter tuning tools like **GridSearchCV** or **Keras Tuner** could be used to further refine the neural network model.

---

This report highlights the iterative process of building and optimizing a deep learning model while considering practical alternatives for improved performance.
