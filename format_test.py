import pandas as pd
import os
import numpy as np


articles = []
articles_labels = []
relative_path = "all_articles\\all_files_aid\\"

if(os.path.exists(relative_path +"51690770_SWB2614597" + '.txt')):
    f = open(relative_path +"51690770_SWB2614597" + '.txt')
    for line in f:
        print(line)
        articles.append(line.strip())
    f.close()


def remove_head(article):
    for i in range(9):
        article.pop(1)
    return np.array(article)

def remove_end(article):
    index = len(article)-9
    for i in range(9):
        article.pop(index)
    return np.array(articles)

def recombine(article):
    full = ""
    for line in article:
        full+=line
    return full

print(len(articles))
remove_head(articles)
remove_end(articles)
print(len(articles))
print(recombine(articles))


