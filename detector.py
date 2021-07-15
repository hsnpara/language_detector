import numpy as np
import pandas as pd
import os
import time
from sklearn.model_selection import train_test_split
from sklearn import svm

# prepare the bigram matrix from line


def line_to_bigram(bigram_dict, line, alphabet):

    word_list = line.lower().split(" ")
    counter = 0
    for word in word_list:
        first_letter = "_"
        for second_letter in word:
            if second_letter not in alphabet or first_letter not in alphabet:
                first_letter = second_letter
                continue
            bigram_dict[first_letter+second_letter] += 1

            first_letter = second_letter
            counter += 1

        if first_letter in alphabet:
            bigram_dict[first_letter+"_"] += 1
    for bigram in bigram_dict.keys():
       bigram_dict[bigram] /= counter

    return bigram_dict


# reading the bigram
samples = []
y = []

for file in os.listdir('data'):
    if file[0:2] == 'tr':
        y.append(0)
    elif file[0:2] == 'en':
        y.append(1)
    else:
        y.append(2)

    data = np.load('data/'+file, allow_pickle=True).item()
    del data['__']
    samples.append(data)
start = time.time()
# prepare the matrix from data
df = pd.DataFrame(samples)
num = df.shape[0]
ls = df.values.tolist()

X_train, X_test, y_train, y_test = train_test_split(ls, y, test_size=0.2, random_state=109)  # 80% training and 20% test
clf = svm.SVC(kernel='linear')  # Linear Kernel
clf.fit(X_train, y_train)  # feed the machine wih train set
predict_matrix = clf.predict(X_test)
end = time.time()
y_pred = clf.predict(X_test)
count = 0
for i in range(len(y_pred)):
    if y_pred[i] != y_test[i]:
        count += 1
print(count)
print(100 - count/num*100)
print(end-start)

# prepare the dictionary
alphabet = "abcdefghijklmnoprstuvyz_"
bigram_dict = {}
for first_letter in range(len(alphabet)):
    for second_letter in range(len(alphabet)):
        bigram_dict[alphabet[first_letter]+alphabet[second_letter]] = 0
del bigram_dict["__"]

# take file path as input and prepare bigram matrix from document
#document = open(input("please enter the file path: "), "r")
document = "Bu akşam eve gitmek için otobüse bindim"
document = "I got on the bus to go home tonight"
document = "Ich stieg in den Bus, um heute Nacht nach Hause zu fahren"
# for line in document:
#    bigram_dict = line_to_bigram(bigram_dict, line, alphabet)
bigram_dict = line_to_bigram(bigram_dict, document, alphabet)
# give the result to prediction
text_predict = []
text_predict.append(list(bigram_dict.values()))

print(clf.predict(text_predict))
