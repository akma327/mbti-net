# Author: Anthony Ma
# Date: 03/04/17
# batch_clean_and_segment_book.py

import os 
import sys
import glob 

USAGE_STR = """

# Purpose 
# Run clean and segmentation upon all book .txt files in the raw-book-files directory
# Output to clean-segmented-files directory 

# Usage 
# python batch_clean_and_segment_book.py <INPUT_DIR> <OUTPUT_DIR>

# Example

python batch_clean_and_segment_book.py /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/raw-book-files /afs/ir.stanford.edu/users/a/k/akma327/cs224n/project/data/cleaned-segmented-files

"""


def batch_clean_and_segment(INPUT_DIR, OUTPUT_DIR):
	"""
		Run batch clean and segment code.
	"""

	input_files = glob.glob(INPUT_DIR + "/*/*.txt")
	for i, infile in enumerate(input_files):
		print(str(i) + "). CLEANING AND SEGMENTING: " + infile + "\n")
		outfile = infile.replace("raw-book-files", "cleaned-segmented-files")
		run_command = "python clean_and_segment_book.py " + infile + " " + outfile
		os.system(run_command)



if __name__ == "__main__":
	(INPUT_DIR, OUTPUT_DIR) = (sys.argv[1], sys.argv[2])
	batch_clean_and_segment(INPUT_DIR, OUTPUT_DIR)
