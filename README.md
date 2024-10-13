# Machine Learning Model Comparison on Breast Cancer Dataset

This project demonstrates the application of seven different machine learning algorithms to the Breast Cancer dataset from the `sklearn` library. The goal is to compare their performance in terms of accuracy, precision, recall, and confusion matrices, and visualize the results using bar charts.

## Dataset
We use the **Breast Cancer Dataset** from the `sklearn.datasets` library, which is commonly used for binary classification tasks.

The project requires the following Python libraries:

numpy
scikit-learn
matplotlib


## Algorithms Implemented
The following machine learning algorithms are trained and evaluated:

1. Naive Bayes (GaussianNB)
2. K-Nearest Neighbors (KNN)
3. Decision Tree (DT)
4. Random Forest (RF)
5. Support Vector Machine (SVM)
6. Logistic Regression (LR)
7. Artificial Neural Networks (MLPClassifier)

## Project Workflow

1. **Data Preprocessing**
   - The dataset is loaded and split into training and test sets.
   - Feature scaling (normalization) is applied using `MinMaxScaler`.

2. **Training the Models**
   - Each of the seven algorithms is trained using the training data.
   
3. **Model Evaluation**
   - After training, each model's performance is evaluated on both training and test datasets using metrics such as:
     - Accuracy
     - Precision
     - Recall
     - Confusion Matrix
     
4. **Result Visualization**
   - Bar charts are used to visually compare the performance of the different models on the test set in terms of:
     - Accuracy
     - Precision
     - Recall

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
