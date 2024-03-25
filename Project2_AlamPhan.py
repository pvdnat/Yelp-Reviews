# -*- coding: utf-8 -*-
"""
Zain Alam, Van Duc Phan
3/17/21

Project 2: Sentiment Analysis
A program that takes data from Yelp reviews and figures out which words are most commonly associated with either positive or negative reviews.
"""

import json
import nltk
from nltk.corpus import stopwords
from nltk.corpus import words as corpusWords
from collections import Counter
import csv

print("Opening file...\n")

# open JSON file
try:
    with open('yelp_academic_dataset_review_small.json') as file:
        data = json.load(file)
except:
    print("Error!")
    
print("Extracting reviews and star ratings...\n")

# extract all review texts and star ratings
list_of_stars=[]
list_of_reviews=[]
for i in data:
    list_of_stars.append(i.get('stars'))
    list_of_reviews.append(i.get('text'))

# initialize lemmatizer and corpuses
lemmatizer = nltk.WordNetLemmatizer()
cWords = set(corpusWords.words("en"))
stopwords = set(stopwords.words("english"))

print("Filtering out stopwords & lemmatizing...\n")

# break each review into words, filter out stopwords, lemmatize and store into respective star rating
list_of_1star = []
list_of_2star = []
list_of_3star = []
list_of_4star = []
list_of_5star = []

n = 0
for review in list_of_reviews:
    words = nltk.word_tokenize(review)
    words = [w.lower() for w in words]
    words = [lemmatizer.lemmatize(w) for w in words if w not in stopwords and w.isalnum()]
    if(list_of_stars[n] == 1):
        list_of_1star.append(Counter(words))
    elif(list_of_stars[n] == 2):
        list_of_2star.append(Counter(words))
    elif(list_of_stars[n] == 3):
        list_of_3star.append(Counter(words))
    elif(list_of_stars[n] == 4):
        list_of_4star.append(Counter(words))
    else:
        list_of_5star.append(Counter(words))
    n += 1
    
print("Processing dictionaries...\n")
    
# summing dicts to get total occurrences for each word
list_of_1star = sum(list_of_1star, Counter())
list_of_2star = sum(list_of_2star, Counter())
list_of_3star = sum(list_of_3star, Counter())
list_of_4star = sum(list_of_4star, Counter())
list_of_5star = sum(list_of_5star, Counter())

# all words and their total count -- doesn't actually take into account star rating
list_of_everything = list_of_1star + list_of_2star + list_of_3star + list_of_4star + list_of_5star

print("Discarding bad/unneeded data...\n")

# if a lemma is used in fewer than 10 reviews or is not in words corpus, discard it
for i in list(list_of_everything):
    if(i not in cWords or list_of_everything[i] < 10):
        del list_of_everything[i]
        if(i in set(list_of_1star)):
            del list_of_1star[i]
        if(i in set(list_of_2star)):
            del list_of_2star[i]
        if(i in set(list_of_3star)):
            del list_of_3star[i]
        if(i in set(list_of_4star)):
            del list_of_4star[i]
        if(i in set(list_of_5star)):
            del list_of_5star[i]
            
print("Calculating average star rating for each lemma...\n")
            
# weigh lemmas w/ their star ratings
for i in list_of_2star:
    list_of_2star[i] = list_of_2star.get(i) * 2
for i in list_of_3star:
    list_of_3star[i] = list_of_3star.get(i) * 3
for i in list_of_4star:
    list_of_4star[i] = list_of_4star.get(i) * 4
for i in list_of_5star:
    list_of_5star[i] = list_of_5star.get(i) * 5
    
# counter object w/ weighted totals
weighted_list = list_of_1star + list_of_2star + list_of_3star + list_of_4star + list_of_5star

# calculate average star rating for each lemma (weighted total / total)
for i in list_of_everything:
    list_of_everything[i] = float(weighted_list[i] / list_of_everything[i])
    
print("Picking out most positive and most negative lemmas...\n")
    
# pick out the 500 most positive and negative words
most_negative = list_of_everything.most_common()[:-500-1:-1]
most_positive = list_of_everything.most_common()[:500]

print("Writing file...\n")

# write csv file -- each word is paired with its average star rating
rows = zip(most_negative, most_positive)
with open("data.csv", "w", newline='') as file:
    writer = csv.writer(file, delimiter=',')
    write = writer.writerow(['Most negative lemmas', 'Most positive lemmas'])
    writer = writer.writerows(rows)
    
print("Done!")
