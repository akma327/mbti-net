# Author: Anthony Ma
# Date: 03/04/17
# load_data.py

import os
import sys
import numpy as np 

USAGE_STR = """

# Usage 
# python load_data.py <DATA_FILE> <PERCENTAGE_TRAIN>

# Arguments
# <DATA_FILE> Absolute path to shuffled data file with each row being a data point 
# <PERCENTAGE_TRAIN> Percentage of rows to use for training. One minus this percentage 
# for testing 

# Example 
python load_data.py /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/mbti-net/data/mbti_shuffled_data.txt 0.7

"""

K_MIN_ARG = 3

mbti_index = {"ISTJ" : 0, "ISFJ" :1, "INFJ" :2, "INTJ" :3, "ISTP" :4, "ISFP" : 5, "INFP":6, "INTP":7, "ESTP": 8, "ESFP":9, "ENFP":10, "ENTP":11, "ESTJ":12, "ESFJ":13, "ENFJ":14, "ENTJ":15}

def load_data(DATA_FILE, PERCENTAGE_TRAIN=0.7):
	f = open(DATA_FILE, 'r')
	all_data = []
	for line in f:
		linfo = line.strip().split("\t")
		mbti, sentence_str = linfo[0], linfo[3]
		mbti_one_hot = np.array([0.0]*16)
		mbti_one_hot[mbti_index[mbti]] = 1

		sentence_str = sentence_str.replace(".", " ").lower()
		words = [w for w in sentence_str.split(" ") if w != ""]

		all_data.append((words, mbti_one_hot))

	num_data_points = len(all_data)
	num_train_points = int(PERCENTAGE_TRAIN*num_data_points)

	train_data = all_data[:num_train_points]
	test_data = all_data[num_train_points:]

	return train_data, test_data





if __name__ == "__main__":
	if(len(sys.argv) < K_MIN_ARG):
		print(USAGE_STR)
		exit(1)

	DATA_FILE, PERCENTAGE_TRAIN = (sys.argv[1], float(sys.argv[2]))
	train_data, test_data = load_data(DATA_FILE, PERCENTAGE_TRAIN)
	print(train_data[0:50], len(train_data), len(test_data))





