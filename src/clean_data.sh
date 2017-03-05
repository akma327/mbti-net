#!/bin/bash
#Usage: have directory called "data" in the current working directory containing the subdirectories of raw data
#Output: new directory called "clean_data" in the current working directory containing the subdirectories of clean data
FILES=$(find data -type f -name '*.txt')
for f in $FILES
do
  python clean_and_segment_book.py $PWD/$f $PWD/clean_$f
  echo "Processed data into clean_data!"
done