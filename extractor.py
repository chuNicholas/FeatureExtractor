import os
import time
from bs4 import BeautifulSoup
from tkinter import Tk, filedialog
import threading
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

MAX_THREADS = 4
articles = {}

# folder selection dialog
def folder_selector():
    # folder selector
    root = Tk()
    root.withdraw()

    file_path = filedialog.askdirectory(title="Select Dataset Folder")

    if file_path == "": exit()

    file_path = os.path.abspath(file_path)

    root.destroy()

    print(file_path)

    return file_path

# convert html to text
def html_to_text(text_file):
    text = ""
    with open(text_file, encoding="utf8") as fp:
        soup = BeautifulSoup(fp, "lxml")
        for words in soup.find_all("p"):
            text += words.get_text() + " "
    
    return text

# parse all files in folder
def parse_folder(classification, fp):
    file_names = [os.path.join(fp, article) for article in os.listdir(fp)]

    for article in file_names:
        articles[classification].append(html_to_text(article))

# convert folder name to folder path
def folder_path(root, folder_name):
    return os.path.join(root, folder_name)

def main():
    # timer
    start = time.time()

    # find the dataset folder
    dataset = folder_selector()

    # create a dictionary. will store an list of the text from the articles
    global articles
    articles = {folder : [] for folder in os.listdir(dataset)}

    keys = sorted(articles)
    i = 0
    threads = []
    while i < len(articles):
        while i < len(articles) and threading.activeCount() <= MAX_THREADS:
            thread = threading.Thread(target=parse_folder, args=(keys[i], folder_path(dataset, keys[i])))
            threads.append(thread)
            thread.start()
            i += 1

    # finish threads when they are done running
    for thread in threads:
        thread.join()

    # scikit learn Feature Extraction library
    vectorizer = CountVectorizer()

    # put the articles in one array
    articles_list = []
    for key in keys:
        articles_list += articles[key]

    # create the model
    matrix = vectorizer.fit_transform(articles_list)

    # convert to numpy array
    matrix_array = np.asarray(matrix.toarray())

    # convert feature names to numpy array
    header = np.asarray(vectorizer.get_feature_names())

    # add filler blank for header
    header = np.concatenate((["Articles: {} \\Terms: {}".format(len(articles_list), len(header))], header))

    new_array = []

    # number of articles per topic = number of articles divided by the number of items in dictionary
    num_articles = len(articles_list) // len(keys)
    num_topics = len(keys)

    # loop through each row in matrix, and all article number
    for i in range(len(matrix_array)):
        new_array.append(np.concatenate((np.asarray(["{} article {}".format(keys[i // num_articles], (i % 10) + 1)]), matrix_array[i])))

    # concatenate header array with the rest of the DTM
    csv_array = np.concatenate(([header], new_array))

    # save to csv
    np.savetxt("document-term-matrix.csv", csv_array , delimiter = ",", fmt="%s")

    end = time.time()
    print("Elapsed time: {:.2f} seconds".format(end - start))

main()