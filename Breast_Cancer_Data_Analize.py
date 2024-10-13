
######################## Load the dataset from Sklearn 
from sklearn.datasets import load_breast_cancer
Breast_Cancer = load_breast_cancer()

# Optional Code (To see the description,shape and target of data)
print (Breast_Cancer.DESCR)
print (Breast_Cancer.target.shape)
print (Breast_Cancer.target[0])
print (Breast_Cancer.data.shape)
print (Breast_Cancer.data[0])




############################## Preprocessing 

# Split the train and test data
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(Breast_Cancer.data , Breast_Cancer.target , test_size=0.2)


# Optional (To be sure about the size of train and test data which now are known as Feature and Lable)
print (f"Feature Train = {x_train.shape} And Feature Test = {x_test.shape} ")
print (f"Lable Train   = {y_train.shape} And Lable Test   = {y_test.shape} ")

# Normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0 , 1))

x_train = scaler.fit_transform(x_train)
x_test  = scaler.transform(x_test)



################################ Calculate Metrics
# calculate the metrix by using a Function 
from sklearn.metrics import confusion_matrix , precision_score , recall_score, accuracy_score
def calculate_metrics(y_train,y_test,y_predict_train,y_predict_test):

    accuracy_train   = accuracy_score  (y_true = y_train , y_pred = y_predict_train)
    accuracy_test    = accuracy_score  (y_true = y_test  , y_pred = y_predict_test )
    confusion_test   = confusion_matrix(y_true = y_test  , y_pred = y_predict_test )
    precision_test   = precision_score (y_true = y_test  , y_pred = y_predict_test )
    recall_test      = recall_score    (y_true = y_test  , y_pred = y_predict_test )

    print ("confusion      :\n" , confusion_test*100 ,
        "\n precision      : "  , precision_test*100 ,"%",
        "\n recall         : "  , recall_test   *100 ,"%",
        "\n accuracy_Train : "  , accuracy_train*100 ,"%",
        "\n accuracy_Test  : "  , accuracy_test *100 ,"%",
        )
    return accuracy_train,accuracy_test,confusion_test,precision_test,recall_test




###################################### Model Selection 

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
model_gnb = GaussianNB()
model_gnb.fit(x_train,y_train)

# Predict
y_predict_train = model_gnb.predict(x_train)
y_predict_test  = model_gnb.predict(x_test)

# Show the Metrics
# Use 4 variables because we want to compare the result at the end to avoid rewrite on same variabl
accuracy_train_gnb,accuracy_test_gnb,confusion_test_gnb,precision_test_gnb,recall_test_gnb = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)


# KNeighbors
from sklearn.neighbors import KNeighborsClassifier
model_knn = KNeighborsClassifier( n_neighbors= 4 )
model_knn.fit(x_train,y_train)
# Predict
y_predict_train = model_knn.predict(x_train)
y_predict_test  = model_knn.predict(x_test)

accuracy_train_knn,accuracy_test_knn,confusion_test_knn,precision_test_knn,recall_test_knn = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)


# Decision tree
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(max_depth=64, min_samples_split=3 , criterion="gini")
model_dt.fit(x_train,y_train)


# Predict
y_predict_train = model_dt.predict(x_train)
y_predict_test  = model_dt.predict(x_test)
accuracy_train_dt,accuracy_test_dt,confusion_test_dt,precision_test_dt,recall_test_dt = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)


# Random forwst
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(n_estimators = 500 , max_depth = 85 , min_samples_split = 2 )
model_rf.fit(x_train,y_train)

# Predict
y_predict_train = model_rf.predict(x_train)
y_predict_test  = model_rf.predict(x_test)
accuracy_train_rf,accuracy_test_rf,confusion_test_rf,precision_test_rf,recall_test_rf = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# SVM
from sklearn.svm import SVC
model_svm = SVC(kernel="poly")
model_svm.fit(x_train, y_train)

# Predict
y_predict_train = model_svm.predict(x_train)
y_predict_test  = model_svm.predict(x_test)
accuracy_train_svm,accuracy_test_svm,confusion_test_svm,precision_test_svm,recall_test_svm = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# Logestic Regression
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

# Predict
y_predict_train = model_lr.predict(x_train)
y_predict_test  = model_lr.predict(x_test)
accuracy_train_lr,accuracy_test_lr,confusion_test_lr,precision_test_lr,recall_test_lr = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)

# MLP
from sklearn.neural_network import MLPClassifier
model_ann = MLPClassifier(hidden_layer_sizes=256 ,  max_iter=200 , activation= "tanh" , solver="adam" , batch_size = 32 )
model_ann.fit(x_train , y_train)

# Predict
y_predict_train = model_ann.predict(x_train)
y_predict_test  = model_ann.predict(x_test)

accuracy_train_ann,accuracy_test_ann,confusion_test_ann,precision_test_ann,recall_test_ann = calculate_metrics(y_train,y_test,y_predict_train,y_predict_test)


################################ Comparision in PLOT

# Accuracy Train 
import matplotlib.pyplot as plt
Accu_train = [accuracy_train_gnb,accuracy_train_knn,accuracy_train_dt,accuracy_train_rf,accuracy_train_svm,accuracy_train_lr,accuracy_train_ann]
tittle     = ["GNB", "KNN" , "DT" , "RF" , "SVM", "LR" , "ANN"]
colors     = ["Black", "Red", "Yellow" , "orange" , "purple" , "green" , "blue"]
plt.bar(tittle,Accu_train , color = colors)
plt.title("Acuuracy in Trained Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Accuracy")
plt.show()



# Accuracy Test 
import matplotlib.pyplot as plt
Accu_test = [accuracy_test_gnb,accuracy_test_knn,accuracy_test_dt,accuracy_test_rf,accuracy_test_svm,accuracy_test_lr,accuracy_test_ann]
tittle     = ["GNB", "KNN" , "DT" , "RF" , "SVM", "LR" , "ANN"]
colors     = ["Black", "Red", "Yellow" , "orange" , "purple" , "green" , "blue"]
plt.bar(tittle, Accu_test , color = colors)
plt.title("Acuuracy in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Accuracy")
plt.show()


# Precition 
import matplotlib.pyplot as plt
Precition = [precision_test_gnb,precision_test_knn,precision_test_dt,precision_test_rf,precision_test_svm,precision_test_lr,precision_test_ann]
tittle     = ["GNB", "KNN" , "DT" , "RF" , "SVM", "LR" , "ANN"]
colors     = ["Black", "Red", "Yellow" , "orange" , "purple" , "green" , "blue"]
plt.bar(tittle, Precition , color = colors)
plt.title("Precition in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Precition")
plt.show()


# Recall 
import matplotlib.pyplot as plt
recall = [recall_test_gnb,recall_test_knn,recall_test_dt,recall_test_rf,recall_test_svm,recall_test_lr,recall_test_ann]
tittle     = ["GNB", "KNN" , "DT" , "RF" , "SVM", "LR" , "ANN"]
colors     = ["Black", "Red", "Yellow" , "orange" , "purple" , "green" , "blue"]
plt.bar(tittle, recall , color = colors)
plt.title("Recall in Test Data")
plt.grid()
plt.xlabel("Name of Algorithms")
plt.ylabel("Recall")
plt.show()
