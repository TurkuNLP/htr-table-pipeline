from zipfile import ZipFile
import os
import argparse
import json
from tqdm import tqdm

class DeskewData:
    def __init__(self, content):
        """
        Parse content string with format:
        #DBG tl_old tl_new tm_old tm_new tr_old tr_new br_old br_new bm_old bm_new bl_old bl_new
        DBG 62,75 64,89 1073,70 1063,93 2084,53 2063,41 71,1646 65,1658 1082,1641 1065,1650 2089,1632 2088,1616
        """
        # skip header line, get data line
        data_line=content.strip().split('\n')[-1]
        # split into pairs and convert to tuples
        pairs=data_line.split()[1:]  # skip DBG
        
        def parse_point(s):
            try:
                return tuple(map(int, s.split(',')))
            except:
                return None
                
        self.points=[parse_point(pair) for pair in pairs]
        
        if len(self.points)!=12:
            raise ValueError(f"Expected 12 points, got {len(self.points)}")
            
        def get_point(old, new):
            return old if new is None else new
            
        # named access to points
        self.tl_old, tl_new=self.points[0:2]
        self.tm_old, tm_new=self.points[2:4] 
        self.tr_old, tr_new=self.points[4:6]
        self.br_old, br_new=self.points[6:8]
        self.bm_old, bm_new=self.points[8:10]
        self.bl_old, bl_new=self.points[10:12]
        
        self.tl_new=get_point(self.tl_old, tl_new)
        self.tm_new=get_point(self.tm_old, tm_new)
        self.tr_new=get_point(self.tr_old, tr_new)
        self.br_new=get_point(self.br_old, br_new)
        self.bm_new=get_point(self.bm_old, bm_new)
        self.bl_new=get_point(self.bl_old, bl_new)

    def get_mid_vertical_x(self):
        left=(self.tm_new[0]-self.tl_new[0])
        right=(self.tr_new[0]-self.tm_new[0])
        return left/(left+right)
    
    
## The format of the txt file is:
# #DBG tl_old tl_new tm_old tm_new tr_old tr_new br_old br_new bm_old bm_new bl_old bl_new
# DBG 62,75 64,89 1073,70 1063,93 2084,53 2063,41 71,1646 65,1658 1082,1641 1065,1650 2089,1632 2088,1616

def read_txt_from_zip(zip_path):
    with ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith('.txt'):
                # get contents and decode assuming utf-8
                content=zf.read(name).decode('utf-8')
                basename=os.path.basename(name)
                yield basename, content

def gather_mid_point_metadata_from_autods(zip_path):
    for basename, content in read_txt_from_zip(zip_path):
        deskew_data=DeskewData(content)
        print(json.dumps({"basename":basename, "mid_vertical_x_point_relative":deskew_data.get_mid_vertical_x()}))




if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('zip_files', nargs='+', help='Zip files to process')
    args=parser.parse_args()
    
    for zip_path in tqdm(args.zip_files, desc='Processing zip files'):
        gather_mid_point_metadata_from_autods(zip_path)

