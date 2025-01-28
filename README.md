Purpose: This is a basic python script to demonstrate the use of various supervised and unsupervised machine learning
          methods and log the results.
          
Input: A configuration file containing a list of .csv files corresponding to different datasets. Each dataset
       is separated by a new line. This allows multiple datasets to be classified in one batch. This configuration
       file is passed via command line. The config file, as well as the .csv files should be in the same directory
       as this python script and the config file must be in .txt format.
       Usage: python machine_learning_classifier.py <config_file>

Output: For each dataset listed in the config file, a folder containing three files will be produced as output.
        The file names of each folder, and the names of the files inside each folder are appended with the current
        date and time. The file contents are as follows:
        
            1) A log file containing the results of each supervised and unsupervised classification method in
               text format. Example: Log_20241228_112538.txt
            2) A .png file containing the K-Means clustering plot. Example: kmeans_clustering_20241228_112538.png
            3) A .png file containing the PCA plot. Example: pca_20241228_112538.png
