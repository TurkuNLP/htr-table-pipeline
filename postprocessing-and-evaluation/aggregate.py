import argparse
import csv
import zipfile
import glob
import data_functions
from evaluate_layouts import evaluate_layout

move_from_names = ["(muuttokirjan)paikka", "mistä seurakunnasta muutettiin", "tulopaikka", "mistä seurakunnasta on tullut",
    "mistä", "mistä muuttanut", "mistä muuttavat", "seurakunta, josta muutti", "seurakunta, mistä muuttanut", "seurakunta, josta muuttaa",
    "seurak nimi josta on tullut", "paikka, josta tuli", "paikka josta muutti", "från vilken församling inflyttningen skett",
    "hvarifrån kommen", "hvarifrån de inflyttat", "muuttopaikka", "församling, dit utflytning skett"]
move_to_names = ["mihin seurakuntaan on mennyt", "mihin", "mihin muuttanut", "mihin muuttavat", "mihinkä", "lähtöpaikka",
    "maa, jonne muutto ilmotettu tahi otaksutaan tapahtuneeksi", "paikka, johon muutto tapahtuu", "paikka, johon muuttaa",
    "seurakunta, johon muutettiin", "seurakunta, johon muutti", "seurakunta mihin muuttaa", "seurakunta, johon muuttaa",
    "seurak. nimi johon muuttaa", "hvart flyttat", "flyttet till", "muuttopaikka", "församling, dit utflytning skett"]
not_available = ["print 11", "print 21", "print 22", "print 25", "print 43", "print 49", "print 51"]


def process_table(annotation, table, layout):

    rows = [] # collect all processed rows here
    
    # check if the number of columns is correct
    predicted_columns = len(table[0])
    target_columns = annotation["columns"]
    if predicted_columns == target_columns:
        # find which columns are from and to in this table based on headers
        from_column = None
        to_column = None
        for column_idx, header in enumerate(annotation["headers"]):
            if (annotation["direction"] == "in" or annotation["direction"] == "both") and header in move_from_names:
                from_column = column_idx
            if (annotation["direction"] == "out" or annotation["direction"] == "both") and header in move_to_names:
                to_column = column_idx
        if from_column == None and to_column == None:
            log = "not able to locate to/from columns"
        else:
            log = "ok"

    else: # wrong number of columns
        log = "wrong number of columns"
        from_column = None
        to_column = None
  
    # fill in from and to information for each row (if available, otherwise None)    
    for row in table:
        row_data = {"direction": annotation["direction"], "layout type": layout, "log": log, "from": None, "to": None}
        if from_column != None:
            row_data["from"] = row[from_column]
        if to_column != None:
            row_data["to"] = row[to_column]
        row_data["original columns"] = row # add original columns to the row
        rows.append(row_data)
    return rows



def standardize_printed(layout, tables, layout_annotations):
    
    annotations = layout_annotations[layout] # example: [{"page": "opening", "direction": direction, "columns": number_of_columns, "headers": headers}]
    label = evaluate_layout(layout, tables, layout_annotations)

    all_rows_from_image = []

    # we have the expected number of tables, iterate annotations and tables and process one table at a time
    if len(annotations) == len(tables):
        for annotation, table in zip(annotations, tables):
            table_rows = process_table(annotation, table, layout)
            all_rows_from_image += table_rows
        return all_rows_from_image

    # wrong number of tables, return just original rows without aggregation
    # 1) if this is one table per opening, process all tables "normally"
    # 2) if page is having identical pages, process all tables "normally"
    # 3) if different pages, skip as we do not have information about which table is from which page (we would need coordinates for this)

    # if only one possible direction, select that, otherwise unknown
    if len(annotations) == 1:
        print("wrong number of tables – only one annotated layout", layout)
        for table in tables:
            table_rows = process_table(annotations[0], table, layout)
            all_rows_from_image += table_rows
    elif len(annotations) == 2 and annotations[0]["direction"] == annotations[1]["direction"] and annotations[0]["headers"] == annotations[1]["headers"]:
        print("wrong number of tables – same layout for both pages", layout)
        for table in tables:
            table_rows = process_table(annotations[0], table, layout)
            all_rows_from_image += table_rows
    else:
        direction = annotations[0]["direction"] if len(set([t["direction"] for t in annotations]))==1 else "unknown"
        print("wrong number of tables – different layout – FAIL", layout)
        for table in tables:
            for row in table:
                all_rows_from_image.append({"direction": direction, "layout type": layout, "log": "wrong number of tables", "from": None, "to": None, "original columns": row})    
    
    return all_rows_from_image
        
    



def standardize_other(layout, tables):
    rows = []
    for table in tables:
        for row in table:
            rows.append({"direction": "unknown", "layout type": layout, "log": "not implemented", "from": None, "to": None, "original columns": row})
    return rows



def standardize_page(page, layout_annotations, zip_file):
    # page: page number, file path, url, layout type
    predicted_files = zip_file.namelist() # each page should be in predicted files
    file = page["file path"]
    layout = page["layout type"]
    if file not in predicted_files:
        print(f"Warning, predicted file missing for {file}")
        return [{"direction": "unknown", "layout type": layout, "log": "predicted file missing", "from": None, "to": None, "original columns": []}]
    with zip_file.open(file) as xml_file:
        # extract list of tables from xml, each table is list of rows, headers are skipped
        tables = data_functions.extract_datatables_from_xml(xml_file)
        if not tables:
            return [{"direction": "unknown", "layout type": layout, "log": "no predicted tables", "from": None, "to": None, "original columns": []}]
        if layout.startswith("print"):
            return standardize_printed(layout, tables, layout_annotations)
        else:
            return standardize_other(layout, tables)



def aggregate_statistics(books, layout_annotations, output_filename, parishes):
    # read all books, and aggregate statistics into standardized format

    print("Standardize and aggregate:", parishes)

    master_table = []
    total_pages = 0
    collected_data_from = 0
    skipped_parishes = 0
    for book_name in books.keys():

        # extract parish name from book name
        parish, _ = book_name.split("muuttaneet", 1) # autods_turku_muuttaneet_1900-1910_ap
        parish_name = parish.replace("autods_", "")
        if parish_name[-1] == "_":
            parish_name = parish_name[:-1]

        # skip this parish or not?
        if parishes != ["all"] and parish_name not in parishes:
            skipped_parishes += 1
            continue

        # do we have a predicted zipfile for this parish? 
        try:
            zip_filename = glob.glob(f"csc_data/{parish}*.zip")[0]
        except IndexError:
            print(f"Missing .zip file for book {parish}")
            continue
        
        with zipfile.ZipFile(zip_filename, 'r') as zip_file: # open parish zip file
            pages = books[book_name] # a list of pages where page is (page number, file path, url, layout type)
            for page in pages:
                total_pages += 1

                # return standardized rows, where each row is a dictionary having keys: direction, layout type, from, to, original columns
                # log (str) is either "ok", or reason for failure
                image_rows = standardize_page(page, layout_annotations, zip_file)

                collected_data_from += 1

                # add relevant book level information to each row and add to the master table
                for row in image_rows: 
                    row["book parish"] = parish_name
                    row["book name"] = book_name
                    row["url"] = page["url"]
                    master_table.append(row)

    # overall statistics
    print(f"Total {total_pages} pages in processed parishes.")
    print(f"Collected data from {collected_data_from} pages.")
    print(f"Skipped {skipped_parishes} parishes.")
    
    # write to a file
    with open(output_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        keys = ["book name", "url", "layout type", "log", "direction", "book parish", "from", "to"]
        writer.writerow(keys+["original columns"])
        for row in master_table:
            columns = [row.get(key) for key in keys] + row["original columns"]
            writer.writerow(columns)
    print(f"Output written to {output_filename}.")


def main(args):

    # read layout annotations
    # dictionary, key: layout type, value: list of dictionaries with keys: page, direction, number of columns, headers (list)
    layout_annotations = data_functions.read_layout_annotations(args.annotations)
    print(f"Read {len(layout_annotations)} pre-printed layout annotations from {args.annotations}.")

    # read books from annotations
    # book_name: e.g. autods_turku_muuttaneet_1900-1910_ap
    # book: a list of pages where page is (page_number, file_path, url, layout_type)
    books = {} # key: book name, value: list of pages
    for book_name, book in data_functions.yield_parish_files_and_annotations(args.annotations, warn=args.warn):
        assert book_name not in books
        books[book_name] = book
    print(f"Read {len(books)} books from {args.annotations}.")
    

    ### STANDARDIZE TABLES
    aggregate_statistics(books, layout_annotations, args.output_file, parishes=args.parishes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, required=True, help='The excel file with book and layout annotations, first tab should be books and the second layouts.')
    parser.add_argument('--zipfile', type=str, help='The zip file to evaluate.')
    parser.add_argument('--warn', action='store_true', default=False, help='Print verbose warnings.')
    parser.add_argument('--output_file', type=str, default="standardized_data.tsv", help='File to save aggregated data (tsv).')
    parser.add_argument('--parishes', type=str, default=[], nargs='*', help='List of parsihes to standardize, use "all" for everything.')
    args = parser.parse_args()
    
    main(args)