The operating system I used to develop my system is windows 11. The training testing and cross validation program were programmed in python version 3.14. The usage is as follows.
To train: Python3 train.py traininglabels.labels modelName.json
To test: Python3 test.py modelName.json testList.list predictionLabels.labels [alpha]
To perform cross validation:
Python3 cross_validate.py trainingLabels.labels [k] [alpha] [seed] 
By default alpha is set to 0.1 because that is what gave the most accurate results
K is set to 5 and seed is 42
	The following python modules are required (all of which can be installed with pip):
json, math, os, random, re, sys, collections.Counter, collections. defaultdict
	The program itself uses multinomial naïve bayes. The tokenizer utilizes regular expressions, text is lowercased and then extracted with the regular expression [A-Za-z0-9'] dropping punctuation as well as hyphenated words. The same tokenizer was used for both training and testing. The program utilizes Laplace smoothing whose formula is as follows:
P(token | category) = (count(token, category) + alpha) / (total tokens in category + alpha * |V|)
The value for alpha chosen by default is 0.1 as it provided the best accuracy when testing. However, a stop list for commonly used words was attempted (and can be seen as a code as a comment at the top). The stop list caused worse results and was therefore left out of the final program.
	To evaluate the systems performance, a cross_validation script was made. Cross_validation employs k-fold cross-validation where examples are grouped by label shuffled and then sent to different folds such that each fold keeps around the same category proportion. The model is trained on the folds not being tested and then then tested on the target fold. This is repeated for each of the folds.
