from random import sample
import openpyxl
import re
import argparse
import os
import shutil

wb = openpyxl.load_workbook('parish_norm_with_formats.xlsx')


file_dir = "/scratch/project_2005072/moving_records_htr"  #muokkaa tämä

ws = wb.active

def yield_images(file_catalogy, layout_to_sample):
    # this function reads the big excel file and yields all images which are relevant to the layout-to-sample
    # the images are yielded as filenames (same as used in puhti)
    
    # 1) iterate the rows of the excel file
    # 2) check if the book == layout-to-sample (is this print 1?)
    # 3) if yes, check the pages (are all pages in the same layout or just a subset)
    # 4) iterate the images in the book
    # 5) create filenames for the images
    # file_path = f"testikuvat/{parish}/muuttokirjoja_{years}_{source}/{parish}_muuttokirjoja_{years}_{source}_{str(i+1)}.jpg"
    # this information can be extracted from the excel
    # 6) yield the filename

    for row in range(2, ws.max_row + 1): 
        # poimi: parish, images (kuvien lukumäärä), years, source (ap, microfilm)
        # column A parish (normalized)
        a_value = ws.cell(row=row, column=1).value
        # column D IMAGES(count)
        d_value = ws.cell(row=row, column=4).value
        # column E doc(muuttokirjoja vai muuttaneet)
        e_value = ws.cell(row=row, column=5).value
        # column F URL
        f_value = ws.cell(row=row, column=6).value
        # column H years
        h_value = ws.cell(row=row, column=8).value
        # column I source
        i_value = ws.cell(row=row, column=9).value
        # column R notes
        r_value = ws.cell(row=row, column=18).value

        parish_nor = a_value
        images = d_value
        doc = e_value
        book_url = f_value
        years = h_value
        source = i_value.lower()
        notes = r_value.lower()


       

        notes = notes.split(",")
        notes = [n.strip() for n in notes]
        for note in notes:
            if ":" in note:
                print_type, pages = note.split(":")
            else:
                print_type = note
                pages = None
            

            all_printed = print_type.startswith('print')
            if (args.layout_to_sample == 'all printed' and all_printed) or print_type == args.layout_to_sample:
                if pages:
                    page_ranges = pages.split(',')
                    for page_range in page_ranges:
                        if ('-') in page_range:
                            start, end = map(int, page_range.split('-'))
                            for i in range(start, end):
                                file_path = f"images/{parish_nor}/{doc}_{years}_{source}/{parish_nor}_{doc}_{years}_{source}_{str(i+1)}.jpg"
                                yield file_path, note
                        else:
                            continue
                            
                else:
                    for i in range(images):
                        file_path = f"images/{parish_nor}/{doc}_{years}_{source}/{parish_nor}_{doc}_{years}_{source}_{str(i+1)}.jpg"
                        yield file_path, note

            else:
                #print("Do not sample this", note)
                continue

   


def main(args):

    images_to_sample = yield_images(args.file_catalogy, args.layout_to_sample)

    sampled_images = sample(list(images_to_sample), args.sample_size) # remember to sample more than needed, we can always leave images unused (sampling more later is more difficult because we need to control those already sampled earlier...)

    # sampled_images is a list of filenames in sampled order (do not modify the order after sampling, because now a subslice is also a valid random sample!)
    # 1) print theses filenames to a file (README.txt under args.output-dir eli luo uusi hakemisto mihin nämä arvot tulevat)
    # 2) copy the images to args.output-dir so that we can easily transfer the sample from puhti to local machine/transkribus

    output_dir = os.path.join(file_dir, args.output_dir)
    if not os.path.exists(output_dir):
        print("Creating dir:", output_dir)
        os.makedirs(output_dir)
    
    readme_path = os.path.join(output_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        for file, note in sampled_images:
            f.write(f'{file}  {note} \n' )

    for file, note in sampled_images:
        source_path = os.path.join(file_dir, file)
        destination_path = os.path.join(output_dir, os.path.basename(file))
        if os.path.exists(source_path):
            shutil.copy(source_path, destination_path)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Majority vote')
    parser.add_argument('--file-catalogy', type=str, help='Big excel file with all the books')
    parser.add_argument('--layout-to-sample', type=str, required=True, help='Name of the annotated layout to sample (e.g. "print 1")')
    parser.add_argument('--sample-size', type=int, required=True, help='Name of the annotated layout to sample (e.g. "print 1")')
    parser.add_argument('--output-dir', type=str, required=True, help='Directory to copy the sampled files')
    args = parser.parse_args()
    
    main(args)


    # run: python sample_images.py --file-catalogy excel.xlsx --layout-to-sample "print 1" --sample-size 100 --output-dir sample-print1