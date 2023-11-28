import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib 
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv('/home/lnx/Documents/Body-Language-Decoder/sign-language-detector-python/hand_landmarks_dataset.csv')

x = data[["x0", "y0","x1", "y1","x2", "y2","x3", "y3","x4", "y4","x5", "y5","x6", "y6","x7", "y7","x8", "y8","x9", "y9","x10", "y10","x11", "y11","x12", "y12","x13", "y13","x14", "y14","x15", "y15","x16", "y16","x17", "y17","x18", "y18","x19", "y19","x20", "y20"]]
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

    
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')


joblib.dump(knn, "knn_model_lab.pkl")