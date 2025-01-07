import logging
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# machine_learning_classifier.py
# Written by: Jason Parsons
# Date: December 28, 2024
#
# Input: A configuration file containing a list of .csv files corresponding to different datasets. Each dataset
#        is separated by a new line. This allows multiple datasets to be classified in one batch. This configuration
#        file is passed via command line. The config file, as well as the .csv files should be in the same directory
#        as this python script and the config file must be in .txt format.
#        Usage: python machine_learning_classifier.py <config_file>
# 
# Output: For each dataset listed in the config file, a folder containing three files will be produced as output.
#         The file names of each folder, and the names of the files inside each folder are appended with the current
#         date and time. The file contents are as follows:
#            1) A log file containing the results of each supervised and unsupervised classification method in
#               text format. Example: Log_20241228_112538.txt
#            2) A .png file containing the K-Means clustering plot. Example: kmeans_clustering_20241228_112538.png
#            3) A .png file containing the PCA plot. Example: pca_20241228_112538.png
#
# Purpose: This is a basic python script to demonstrate the use of various supervised and unsupervised machine learning
#          methods and log the results.

# Setup and initialize our logger
def setup_logging(output_dir, current_time):
    log_file = os.path.join(output_dir, f'Log_{current_time}.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

# Load the data from the .csv file. Check for column correctness and log any erros during loading
def load_data(filepath):
    try:
        data = pd.read_csv(filepath)
    except FileNotFoundError:
        logging.error(f"The file '{filepath}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return None
    except pd.errors.ParserError:
        logging.error("There was an error parsing the file.")
        return None

    if data.shape[1] < 2:
        logging.error("The dataset does not contain enough columns.")
        return None

    return data

# Encode the labels from the dataset and log any errors during encoding
def encode_labels(labels):
    try:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
    except Exception as e:
        logging.error(f"Label encoding failed with message: {e}")
        return None

    return encoded_labels

# Train each supervised classifier and log any errors during training
def train_supervised_classifiers(X_train, X_test, y_train, y_test):
    supervised_classifiers = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=3),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC(kernel='linear')
    }

    for name, clf in supervised_classifiers.items():
        try:
            clf.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Training the {name} classifier failed with message: {e}")
            continue
        
        try:
            y_pred = clf.predict(X_test)
        except Exception as e:
            logging.error(f"Prediction with the {name} classifier failed with message: {e}")
            continue

        try:
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"{name} Accuracy: {accuracy * 100:.2f}%")
        except Exception as e:
            logging.error(f"Calculating accuracy for the {name} classifier failed with message: {e}")

# Perform the two unsupervised classification methods and log any errors.
def perform_unsupervised_methods(X, y, output_dir, current_time):
    try:
        # K-Means Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(X)
        kmeans_labels = kmeans.labels_
        logging.info("K-Means Clustering labels:\n%s", kmeans_labels)

        # Plot and save K-Means Clustering results
        plt.figure(figsize=(8, 6))
        plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis')
        plt.title('K-Means Clustering')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar()
        kmeans_plot_file = os.path.join(output_dir, f'kmeans_clustering_{current_time}.png')
        plt.savefig(kmeans_plot_file)
        logging.info(f"K-Means Clustering plot saved to {kmeans_plot_file}")

        # Principal Component Analysis (PCA)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X)
        logging.info("PCA result:\n%s", pca_result)

        # Plot and save PCA results
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y, cmap='viridis')
        plt.title('PCA')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar()
        pca_plot_file = os.path.join(output_dir, f'pca_{current_time}.png')
        plt.savefig(pca_plot_file)
        logging.info(f"PCA plot saved to {pca_plot_file}")

    except Exception as e:
        logging.error(f"Unsupervised methods failed with message: {e}")

# Process the .csv input file containing the data, setup logging, and process the data
def process_file(filepath):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), current_time)
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir, current_time)
    
    data = load_data(filepath)
    if data is None:
        return

    try:
        # Assume last column is the label
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    except IndexError:
        logging.error("Failed to split the dataset into features and labels.")
        return

    y = encode_labels(y)
    if y is None:
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    except ValueError as e:
        logging.error(f"Dataset splitting failed with message: {e}")
        return

    train_supervised_classifiers(X_train, X_test, y_train, y_test)
    perform_unsupervised_methods(X, y, output_dir, current_time)
    logging.info("File processing completed.")

# Note, I am printing these messages to the console. This is a stylistic choice. It is a trivial change to log
# these messages instead
def main():
    if len(sys.argv) != 2:
        print("Usage: python machine_learning_classifier.py <config_file>")
        return

    config_file = sys.argv[1]
    if not os.path.isfile(config_file):
        print(f"The config file '{config_file}' does not exist.")
        return

    with open(config_file, 'r') as file:
        filepaths = [line.strip() for line in file if line.strip()]

    for filepath in filepaths:
        process_file(filepath)

if __name__ == "__main__":
    main()
