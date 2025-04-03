
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



def cumulative_histogram(counter, N):
    # cumulatice histogram of the top N values in the counter (printed layouts)

    # Get the top N values
    top_N = counter.most_common(N)
    # Extract the keys and values
    keys, values = zip(*top_N)
    keys = [f"print {i+1}" for i, key in enumerate(keys)]
    
    # Compute cumulative values
    cumulative_values = [sum(values[:i+1]) for i in range(len(values))]

    # Plot the cumulative histogram
    plt.bar(keys, cumulative_values, color='skyblue', edgecolor='black')

    # Mark the total (100%) with a vertical line
    total = sum(counter.values())
    plt.axhline(total, color='red', linestyle='--', label='100%')
    #plt.axhline(total*0.9, color='blue', linestyle='--', label='90%')
    plt.axhline(total*0.75, color='blue', linestyle='--', label='75%')
    plt.axhline(total*0.5, color='green', linestyle='--', label='50%')

    # Labeling the plot
    plt.title(f"Pre-printed layouts (top {N})")
    plt.xlabel("Layout type", fontsize=12)
    plt.xticks(rotation=90)
    plt.xlim(-0.5, len(keys) - 0.5)
    plt.ylabel("Cumulative count", fontsize=12)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.savefig("cumulative_printed_histogram.png")  # Save the plot
    plt.show()



def calculate_layout_statistics(books):
    # books is a dictionary, key: book name, value: list of pages

    # collect a list of all page layouts
    page_layouts = []
    for book_name in books.keys():
        book = books[book_name]
        for page in book:
            layout = page["layout type"]
            if layout in ["free text", "halfbook", "handrawn"]: # merge "handrawn" and "handrawn misc" into same category
                layout = layout + " misc" # add misc to these
            page_layouts.append(layout)

    total_pages = len(page_layouts)
    print(f"Total number of pages: {len(page_layouts)}")
    print("Unique layouts:", len(set(page_layouts)), "\n")

    # print statistics separately for each main category
    for main_category in ["free text", "halfbook", "handrawn", "print", "unknown", "empty", "wrong", "digital"]:
        cat_pages = [p for p in page_layouts if p.startswith(main_category)]
        print(f"Main category:")
        print(f"{main_category} & {len(cat_pages)} & {round(len(cat_pages)/total_pages*100,2)}% \\\\ ")
        for l, v in Counter(cat_pages).most_common(5):
            print(f"{l} & {v} & {round(v/len(cat_pages)*100, 2)}% \\\\ ")
        print()
    print()

    # extract all printed pages, and make a cumulative histogram of different printed layouts
    printed_layouts = Counter([l for l in page_layouts if l.startswith("print")])
    cumulative_histogram(printed_layouts, 15)



def create_colors(page_labels):
    # data is a list of tables, where table is a list of rows, where row is a list of columns
    colors = []
    for label in page_labels:
        if label == "correct":
            colors.append(BrightColor.GREEN)
        elif label == "columns off by one":
            colors.append(BrightColor.YELLOW)
        elif label == "no table":
            colors.append(BrightColor.BLACK)
        else:
            colors.append(Color.RED)
    
    #colored_pages = [f"{color} \u2588 {Color.OFF}" for color in colors]
    colors = [f"{color}|{Color.OFF}" for color in colors]
    return "".join(colors)

def evaluate_layout(layout, tables, layout_annotations):
    # one or two tables
    if len(tables) > 2:
        return "more than two tables"
    if layout_annotations[layout][0]["page"] == "opening": # only one table per opening
        if len(tables) != 1:
            return "wrong number of tables"
        # check number of columns
        predicted_columns = len(tables[0][0])
        target_columns = layout_annotations[layout][0]["columns"]
        if predicted_columns == target_columns:
            return "correct"
        elif abs(predicted_columns - target_columns) < 2:
            return "columns off by one"
        else:
            return "columns off by more than one"
    else: # right or left
        if len(tables) != 2:
            return "wrong number of tables"
        # check number of columns
        predicted_columns_left, predicted_columns_right = len(tables[0][0]), len(tables[1][0])
        target_columns_left, target_columns_right = layout_annotations[layout][0]["columns"], layout_annotations[layout][1]["columns"]
        if predicted_columns_left == target_columns_left and predicted_columns_right == target_columns_right:
            return "correct"
        elif abs(predicted_columns_left - target_columns_left) < 2 and abs(predicted_columns_right - target_columns_right) < 2:
            return "columns off by one"
        else:
            return "columns off by more than one"
     


def evaluate_book(book_name, pages, layout_annotations, layouts_to_keep, warn=False):
    # pages = list of pages, where page is a dictionary with keys: page number, file path, url, layout type
    parish, _ = book_name.split("muuttaneet", 1) # autods_turku_muuttaneet_1900-1910_ap
    try:
        zip_filename = glob.glob(f"csc_data/{parish}*.zip")[0]
    except IndexError:
        print(f"Missing .zip file for book {parish}")
        return ["missing" for p in pages]
    page_labels = []
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        predicted_files = zip_file.namelist() # each page should be in predicted files
        for page in pages:
            layout = page["layout type"]
            if layout not in layouts_to_keep:
                continue
            file = page["file path"]
            if file not in predicted_files:
                page_labels.append("missing")
                continue
            with zip_file.open(file) as xml_file:
                tables = data_functions.extract_datatables_from_xml(xml_file)
                # a list of tables, where table is a list of rows, and row is a list of columns (strings)
                if not tables:
                    page_labels.append("no table")
                    continue
                # evaluate layout
                label = evaluate_layout(layout, tables, layout_annotations)
                page_labels.append(label)
    return page_labels


def keep_book(pages, layouts_to_keep):
    # keep book if at least one page has interesting layout
    for page in pages:
        if page["layout type"] in layouts_to_keep:
            return True
    return False


def run_book_evaluation(books, layout_annotations, parishes):

    # for each book in excel file
    # 1) read layout annotations, and decide whether to evaluate the book or not
    # 2) read predictions from zip
    # 3) run evaluation
    # 4) visualize results

    print("Running page-level book evaluation for parishes:", parishes)

    layouts_to_keep = ["print 17", "print 1", "print 45", "print 3", "print 7"]
    number_of_evaluated_books = 0

    all_labels = []

    i = 0
    for book_name in books.keys():
        parish, _ = book_name.split("muuttaneet", 1) # autods_turku_muuttaneet_1900-1910_ap
        parish = parish.replace("autods_", "")
        if parish[-1] == "_":
            parish = parish[:-1]
        if parishes != ["all"] and parish not in parishes:
            continue

        i += 1
        pages = books[book_name]

        # book_name: e.g. autods_turku_muuttaneet_1900-1910_ap
        # book: a list of pages where page is (page_number, file_path, url, layout_type)
        if not keep_book(pages, layouts_to_keep):
            continue
        number_of_evaluated_books += 1
        page_labels = evaluate_book(book_name, pages, layout_annotations, layouts_to_keep)
        color_string = create_colors(page_labels)
        print("\n\n", i, book_name)
        print(color_string)
        #print(page_labels)
        all_labels += page_labels

    print(f"\nConsistency statistics for layouts {','.join(layouts_to_keep)}:")
    for l, c in Counter(all_labels).most_common():
        print(l, c, round(c/len(all_labels)*100, 2), "%")
    print("Total pages:", len(all_labels))
    print("Total books evaluated:", number_of_evaluated_books)

             


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

    # print statistics, and create a cumulative histogram of printed layouts
    calculate_layout_statistics(books)
    
    
    
    ### EVAL BIG RUN CONSISTENCY
    run_book_evaluation(books, layout_annotations, args.parishes)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True, help='The excel file with book and layout annotations, first tab should be books and the second layouts.')
    parser.add_argument('--zipfile', type=str, help='The zip file to evaluate.')
    parser.add_argument('--warn', action='store_true', default=False, help='Print verbose warnings.')
    #parser.add_argument('--output_file', type=str, help='Extract statistics and save to a file.')
    parser.add_argument('--parishes', type=str, default=[], nargs='*', help='List of parsihes to run the full book eval, use "all" for everything.')
    args = parser.parse_args()
    
    main(args)


