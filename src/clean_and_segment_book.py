# Author: Anthony Ma
# Date: 03/04/17
# clean_and_segment_book.py

import sys
import string 
import matplotlib.pyplot as plt

USAGE_STR = """
# Purpose
# Takes a book .txt file, performs a cleaning and filtering protocol. Segment upon 
# punctuation marks to get chunks of N sentences 

# Usage 
# python clean_and_segment_book.py <INPUT_FILE> <OUTPUT_FILE> <chunk_size>

# Arguments
# <INPUT_FILE> Absolute path to input book .txt file 
# <OUTPUT_FILE> Absolute path to output segment file with the following format 
# 		<MBTI><TAB><N sentence segment>
# <chunk_size> Optional argument to specify size of chunks. 

# Example 
INPUT_FILE="/afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/raw-book-files/ESFJ/ESFJ_AndrewCarnegie_Autobiography.txt"
OUTPUT_FILE="/afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/cleaned-segmented-files/ESFJ/ESFJ_AndrewCarnegie_Autobiography.txt"
python clean_and_segment_book.py $INPUT_FILE $OUTPUT_FILE

python clean_and_segment_book.py /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/raw-book-files/ESFJ/ESFJ_AndrewCarnegie_Autobiography.txt /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/cleaned-segmented-files/ESFJ/ESFJ_AndrewCarnegie_Autobiography.txt

"""

K_MIN_ARG = 3
DEFAULT_CHUNK_SIZE = 5 # sentences


def strip_numerics_special_chars(input_str):
	"""
		Remove numerics from string 
	"""

	processed_str = []
	inclusion_char = []
	for c in string.ascii_lowercase:
		inclusion_char.append(c)
	for c in string.ascii_uppercase:
		inclusion_char.append(c)
	inclusion_char += ["?", ".", "!", " "]
	for c in input_str:
		if(c in inclusion_char ):
			processed_str.append(c)
	return "".join(processed_str)

def remove_outlier_strings(input_str, dictionary):
	"""
		Split upon spaces to get individual "words" and get rid of following
			- Empty strings 
			- url containing www or http 
			- Gutenberg 

		Removes long strings that may associate with URLs rather than actual text. 
		Also discard "words" that are a combination of floating punctuations. 
	"""
	words = input_str.split(" ")
	filtered_words = []
	for w in words:

		if(w == "" or w == "."): continue 
		if("www" in w or "http" in w): continue 
		if(".." in w or ".," in w or ",." in w or  "..." in w): continue
		if("Gutenberg" in w or "gutenberg" in w): continue
		if(".org" in w or ".com" in w): continue
		if w not in dictionary: continue
		filtered_words.append(w)

	return " ".join(filtered_words)


def filter_sentences(input_str):
	"""
		Splits upon punctuation and segments into sentence chunks. 
	"""

	input_str = input_str.replace("?", ".")
	input_str = input_str.replace("!", ".")

	sentences = input_str.split(".")
	filtered_sentences = []
	for s in sentences:
		words = s.split(" ")
		num_words = len(words)
		if(num_words > 4 and num_words < 70):
			filtered_sentences.append(" ".join(words))

	return filtered_sentences




def chunk_sentences(filtered_sentences, CHUNK_SIZE=DEFAULT_CHUNK_SIZE):
	"""
		Group sentences of CHUNK_SIZE and output the concatenated string. 
	"""

	sentence_chunks = []
	while filtered_sentences:
		chunk = filtered_sentences[:CHUNK_SIZE]
		filtered_sentences = filtered_sentences[CHUNK_SIZE:]
		sentence_chunks.append(".".join(chunk) + ".\n")
	return sentence_chunks


def write_output(sentence_chunks, OUTPUT_FILE):
	"""
		Write to output file
	"""
	MBTI, AUTHOR, TITLE = OUTPUT_FILE.strip().split("/")[-1].strip(".txt").split("_")
	f = open(OUTPUT_FILE, 'w')
	for chunk in sentence_chunks:
		f.write(MBTI + "\t" + AUTHOR + "\t" + TITLE + "\t" + chunk + "\n")



def process_book(INPUT_FILE, OUTPUT_FILE, dictionary, CHUNK_SIZE=DEFAULT_CHUNK_SIZE):
	"""
		Cleaning: 
			- Strips table of content
			- Concatenate into a single string 
		Segment:
			- Chunk by punctuation
	"""

	f = open(INPUT_FILE, 'r')
	input_str = ""
	for line in f:
		input_str += " " + line.strip()

	processed_str = strip_numerics_special_chars(input_str)
	processed_str = remove_outlier_strings(processed_str, dictionary)
	filtered_sentences = filter_sentences(processed_str)
	sentence_chunks = chunk_sentences(filtered_sentences, CHUNK_SIZE)
	write_output(sentence_chunks, OUTPUT_FILE)


if __name__ == "__main__":
	num_args = len(sys.argv)
	if(num_args < K_MIN_ARG):
		print(USAGE_STR)
		exit(1)

	(INPUT_FILE, OUTPUT_FILE) = (sys.argv[1], sys.argv[2])
	CHUNK_SIZE = DEFAULT_CHUNK_SIZE
	if(num_args == 4):
		CHUNK_SIZE = int(sys.argv[3])

	dictionary = set([word.strip() for word in open('/usr/share/dict/words', 'r')])
	process_book(INPUT_FILE, OUTPUT_FILE, dictionary, CHUNK_SIZE)

