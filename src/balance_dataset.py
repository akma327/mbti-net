# Author: Anthony Ma
# Date: 03/04/17
# balance_dataset.py

import sys
import string 
import matplotlib.pyplot as plt
from random import shuffle

USAGE_STR = """
# Purpose
# Takes a dataset of chunks, balances the frequencies of each category

# Usage 
# python balance_dataset.py <INPUT_FILE> <OUTPUT_FILE> <chunk_size>

# Arguments
# <INPUT_FILE> Absolute path to input book .txt file 
# <OUTPUT_FILE> Absolute path to output segment file with the following format 
#     <MBTI><TAB><N sentence segment>

# Example 
INPUT_FILE="/afs/ir.stanford.edu/users/g/u/gusliu/cs224n/finalProj/mbti-net/data/mbti_shuffled_data.txt"
OUTPUT_FILE="/afs/ir.stanford.edu/users/g/u/gusliu/cs224n/finalProj/mbti-net/data/mbti_balanced_shuffled_data.txt"
python balance_dataset.py $INPUT_FILE $OUTPUT_FILE

python balance_dataset.py /afs/ir.stanford.edu/users/g/u/gusliu/cs224n/finalProj/mbti-net/data/mbti_shuffled_data.txt /afs/ir.stanford.edu/users/g/u/gusliu/cs224n/finalProj/mbti-net/data/mbti_balanced_shuffled_data.txt

"""

FREQUENCY = 2939
K_MIN_ARG = 3


def balance(INPUT_FILE, OUTPUT_FILE):
  """
    Cleaning: 
      - Strips table of content
      - Concatenate into a single string 
    Segment:
      - Chunk by punctuation
  """
  mbti_count = {"ISTJ" : 0, "ISFJ" :0, "INFJ" :0, "INTJ" :0, "ISTP" :0, "ISFP" : 0, "INFP":0, "INTP":0, "ESTP": 0, "ESFP":0, "ENFP":0, "ENTP":0, "ESTJ":0, "ESFJ":0, "ENFJ":0, "ENTJ":0}
  f = open(INPUT_FILE, 'r')
  fw = open(OUTPUT_FILE, 'w')
  line_list = []
  for line in f:
    mbti = line.strip().split("\t")[0]
    if(mbti_count[mbti] >= FREQUENCY): continue
    mbti_count[mbti] += 1
    line_list.append(line)
  shuffle(line_list)

  ### Write to file
  for line in line_list:
    fw.write(line)


if __name__ == "__main__":
  num_args = len(sys.argv)
  if(num_args < K_MIN_ARG):
    print(USAGE_STR)
    exit(1)

  (INPUT_FILE, OUTPUT_FILE) = (sys.argv[1], sys.argv[2])

  balance(INPUT_FILE, OUTPUT_FILE)

