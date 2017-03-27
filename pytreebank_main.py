__author__ = 'pva701'

import pytreebank
dataset = pytreebank.load_sst('/home/pva701/github/cnn-text-classification-tf/data/stanford_sentiment_treebank')
example = dataset["train"][0]

for node in example.all_children():
    print(node.text)
    #print(node.text + ": " + str(node.label))

# extract spans from the tree.
# for label, sentence in example.to_labeled_lines():
# 	print("%s has sentiment label %s" % (
# 		sentence,
# 		["very negative", "negative", "neutral", "positive", "very positive"][label]
# 	))