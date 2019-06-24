import pandas as pd
import os
import numpy as np

# Load in the annotated atricles from "AnnotatedData.csv"
# and store it as a DataFrame named "df"
df = pd.read_csv('Annontated Data - Sheet1.csv')


# Iterates through df to:
#   1. Check whether each article exists
#   2. Save the text of each existing article in an array named "articles"
#   3. Save the label of each existing article in an array name "articles_labels"
def load_data():
    articles = []
    articles_labels = []
    relative_path = "all_articles\\all_files_aid\\"

    for index, row in df.iterrows():

        # an array meant to store the content of a single article
        a = []

        # Checks whether the file exists
        if(os.path.exists(relative_path + row.aid_eid + '.txt')):
            f = open(relative_path + row.aid_eid + '.txt')

            # Reads in the text file by line
            for line in f:
                a.append(line)

            # Removes the first and last 9 lines of the text file
            # excluding the title
            a = remove_head(a)
            a = remove_end(a)

            # Combines all of the lines into one string
            # and add the content to the articles list
            content = recombine(a)
            articles.append(content)

            # Add the label to the articles_labels list
            articles_labels.append(row.label)
            
    return articles, articles_labels


# Removes the first nine lines of an article
# excluding the title
# The parameter should be a list of strings
def remove_head(article):
    for i in range(9):
        article.pop(1)
    return article



# Removes the last nine lines of an article
# The parameter should be a list of strings
def remove_end(article):
    index = len(article)-9
    for i in range(9):
        article.pop(index)
    return article


# Combines every line of the article into one string
# The parameter should be a list of strings
def recombine(article):
    full = ""
    for line in article:
        full+=line
    return full
