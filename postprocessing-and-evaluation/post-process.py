import zipfile
import argparse
import xml.etree.ElementTree as ET
from colorist import Color, BrightColor
from collections import Counter
import glob
import re
import csv
import matplotlib.pyplot as plt
import data_functions





def main(args):

    ### STATISTICS FROM LAYOUT ANNOTATIONS

    # read layout annotations
    # dictionary, key: layput type, value: list of dictionaries with keys: page, direction, number of columns, headers (list)
    layout_annotations = data_functions.read_layout_annotations(args.annotations)

    # read books from annotations
    # book_name: e.g. autods_turku_muuttaneet_1900-1910_ap
    # book: a list of pages where page is (page_number, file_path, url, layout_type)
    books = {} # key: book name, value: list of pages
    for book_name, book in data_functions.yield_parish_files_and_annotations(args.annotations, warn=args.warn):
        assert book_name not in books
        books[book_name] = book
    
    print("Read", len(books), "books from annotations.")

    
    
    
    




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True, help='The excel file with book and layout annotations, first tab should be books and the second layouts.')
    parser.add_argument('--zipfile', type=str, help='The zip file to evaluate.')
    parser.add_argument('--warn', action='store_true', default=False, help='Print verbose warnings.')
    parser.add_argument('--output_file', type=str, help='Extract statistics and save to a file.')
    parser.add_argument('--parishes', type=str, default=[], nargs='*', help='List of parsihes to standardize, use "all" for everything.')
    args = parser.parse_args()
    
    main(args)