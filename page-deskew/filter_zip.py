import zipfile
import os
import tqdm
import argparse

def filter_parish(input_zip_path, output_zip_path, parish):
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        with zipfile.ZipFile(output_zip_path, 'w') as output_zip:
            for entry in input_zip.infolist():
                if parish in entry.filename:
                    with input_zip.open(entry) as file:
                        data = file.read()
                    output_zip.writestr(entry, data)

def gather_names(input_zip_path):
    names = set()
    with zipfile.ZipFile(input_zip_path, 'r') as input_zip:
        for entry in input_zip.infolist():
            names.add(entry.filename.split('/')[0])
    return names

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Filter and split zip files by parish.')
    parser.add_argument('input_zip', type=str, help='Path to the input zip file')
    parser.add_argument('output_dir', type=str, help='Directory to save the output zip files')
    args = parser.parse_args()

    names = gather_names(args.input_zip)
    for n in tqdm.tqdm(names):
        filter_parish(args.input_zip, os.path.join(args.output_dir, f'{n}.zip'), n)
    