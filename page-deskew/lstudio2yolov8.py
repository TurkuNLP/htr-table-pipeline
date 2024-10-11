import argparse
import json
import hashlib
import os
import yaml
import deskew
import cv2
import numpy as np
import imutils
import math
import glob
from xml.etree import ElementTree as ET

class KPoints:

    def __init__(self, tl, tm, tr, bl, bm, br, img_path=None):
        self.tl = tl
        self.tm = tm
        self.tr = tr
        self.bl = bl
        self.bm = bm
        self.br = br
        self.img_path = img_path
        self.image = None
        self.w,self.h=None,None
        self.skew_reskew_meta=None

    def possibly_read_image(self):
        if not self.image and self.img_path:
            self.image = cv2.imread(self.img_path)
            self.w,self.h=self.image.shape[1],self.image.shape[0]

            def calculate_int_coordinates(point):
                return (int(self.w * point[0] / 100), int(self.h * point[1] / 100))

            self.tl_int = calculate_int_coordinates(self.tl)
            self.tm_int = calculate_int_coordinates(self.tm)
            self.tr_int = calculate_int_coordinates(self.tr)
            self.bl_int = calculate_int_coordinates(self.bl)
            self.bm_int = calculate_int_coordinates(self.bm)
            self.br_int = calculate_int_coordinates(self.br)


    def estimate_lr_deskew(self):
        """Old code to estimate skew angle on both sides of the image, hopefully obsoleted by
        corner detection, but might still be useful as a crosscheck"""

        if not self.img_path:
            raise ValueError("No image path provided")
        
        self.possibly_read_image()
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
        x_mid_int=(self.tm_int[0]+self.bm_int[0])//2
        left_half = grayscale[:, :x_mid_int]
        right_half = grayscale[:, x_mid_int:]

        angle_left = deskew.determine_skew(left_half,min_angle=-5.0,max_angle=5.0,min_deviation=0.2)
        angle_right = deskew.determine_skew(right_half,min_angle=-5.0,max_angle=5.0,min_deviation=0.2)
        return angle_left, angle_right, x_mid_int
    
    def deskew(self, angle_left, angle_right, x_mid_int):
        """Old code to apply image deskew by its L/R angles"""
        # Draw rectangles of the two halves on the image
        cv2.rectangle(self.image, (20, 20), (x_mid_int-20, self.h-20), (0, 255, 0), 2)  # Left half rectangle
        cv2.rectangle(self.image, (x_mid_int+20, 20), (self.w-20, self.h-20), (0, 255, 0), 2)  # Right half rectangle
        
        orig_image_resized=imutils.resize(self.image.copy(), width=640)

        #1 right half first
        delta_y_right=(self.w-x_mid_int)*np.tan(np.radians(angle_right)) #how much the right side will move up
        #2) now the left side
        delta_y_left=(x_mid_int)*np.tan(np.radians(angle_left)) #how much the left side will move up

        #3) left-hand transform
        source_left=[[0,0],[x_mid_int,0],[x_mid_int,self.h],[0,self.h]]
        dest_left=[[0,+delta_y_left],[x_mid_int,0],[x_mid_int,self.h],[0,self.h+delta_y_left]]
        M_left=cv2.getPerspectiveTransform(np.array(source_left,dtype=np.float32),np.array(dest_left,dtype=np.float32))

        #3) right-hand transform
        source_right=[[x_mid_int,0],[self.w,0],[self.w,self.h],[x_mid_int,self.h]]
        dest_right=[[0,0],[x_mid_int,-delta_y_right],[x_mid_int,self.h-delta_y_right],[0,self.h]]
        M_right=cv2.getPerspectiveTransform(np.array(source_right,dtype=np.float32),np.array(dest_right,dtype=np.float32))
        
        warped_left=cv2.warpPerspective(self.image,M_left,(self.w,self.h))
        warped_right=cv2.warpPerspective(self.image,M_right,(self.w,self.h))
        combined_image = np.concatenate((warped_left[:,:x_mid_int], warped_right[:,:x_mid_int]), axis=1)

        combined_image_resized = imutils.resize(combined_image.copy(), width=640)
        cv2.imshow("original", orig_image_resized)
        cv2.imshow("combined", combined_image_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def full_deskew(self):
        """Full deskew using the 6 keypoints"""

        self.possibly_read_image()
        
        #3) left-hand transform
        source_left=[self.tl_int, self.tm_int, self.bm_int, self.bl_int]
        left_width = self.tm_int[0] - self.tl_int[0]
        left_height = self.bm_int[1] - self.tm_int[1]
        dest_left=[[0,0],[left_width,0],[left_width,left_height],[0,left_height]]
        M_left=cv2.getPerspectiveTransform(np.array(source_left,dtype=np.float32),np.array(dest_left,dtype=np.float32))
        M_left_reskew=cv2.getPerspectiveTransform(np.array(dest_left,dtype=np.float32),np.array(source_left,dtype=np.float32))
        
        #3) right-hand transform
        source_right=[self.tm_int, self.tr_int, self.br_int, self.bm_int]
        right_width = self.tr_int[0] - self.tm_int[0]
        right_height = self.bm_int[1] - self.tm_int[1]
        dest_right=[[0,0],[right_width,0],[right_width,left_height],[0,right_height]]
        M_right=cv2.getPerspectiveTransform(np.array(source_right,dtype=np.float32),np.array(dest_right,dtype=np.float32))
        M_right_reskew=cv2.getPerspectiveTransform(np.array(dest_right,dtype=np.float32),np.array(source_right,dtype=np.float32))
        
        warped_left=cv2.warpPerspective(self.image,M_left,(self.w,self.h))
        warped_right=cv2.warpPerspective(self.image,M_right,(self.w,self.h))
        combined_image = np.concatenate((warped_left[:left_height,:left_width], warped_right[:left_height,:right_width]), axis=1)

        self.skew_reskew_meta={"M_left":M_left,"M_left_reskew":M_left_reskew,"M_right":M_right,"M_right_reskew":M_right_reskew,"dest_left":dest_left,"dest_right":dest_right,"source_left":source_left,"source_right":source_right,"left_width":left_width,"right_width":right_width} 
        
        return combined_image, self.skew_reskew_meta
    
    def deskew_xml_annotations(self,xml_filename_in,xml_filename_out,scale_factor=1.0):
        #My god how I hate XML, why do they punish us with this
        all_polygons=[] #let me just gather these in case I want to make a test printout of the fig
        ns={"ns":"http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        with open(xml_filename_in,"rt") as f:
            tree = ET.parse(f)
            root = tree.getroot()
            page_elem = root.find('.//ns:Page',ns)
            if page_elem is not None:
                xml_width = int(page_elem.attrib['imageWidth'])
                xml_height = int(page_elem.attrib['imageHeight'])
            else:
                assert False, f"{xml_file} doesn't have any Page element"
            assert self.w==xml_width and self.h==xml_height, f"Image dimensions do not match: image ({self.w}x{self.h}), xml ({xml_width}x{xml_height})"
            for elem in root.findall('.//*[@points]', ns): #anything that has "points" attribute
                points = elem.attrib['points']
                new_points = []
                for point in points.split():
                    x, y = map(int, point.split(','))
                    x_new, y_new = self.deskew_point((x, y))
                    x_new=max(0,x_new)
                    x_new=min(x_new,self.w)
                    y_new=max(0,y_new)
                    y_new=min(y_new,self.h)
                    new_points.append((int(x_new*scale_factor),int(y_new*scale_factor)))
                all_polygons.append(new_points)
                elem.attrib['points'] = " ".join([f"{x},{y}" for x,y in new_points])
        tree.write(xml_filename_out)
        return(all_polygons)
        
    def deskew_point(self,point):
        #This is a bit more complex, since some of the points may fall outside the transform source by few pixels
        sources=[np.array(source,dtype=np.float32) for source in [self.skew_reskew_meta["source_left"],self.skew_reskew_meta["source_right"]]]
        transforms=[self.skew_reskew_meta["M_left"],self.skew_reskew_meta["M_right"]]
        x_offsets=[0,self.skew_reskew_meta["left_width"]]
        x,y=point
        #Which is its nearest source polygon?
        dists=[]
        for s_idx,source in enumerate(sources):
            dist=cv2.pointPolygonTest(source,(x,y),True) #the higher this number, the more inside the point is
            dists.append(dist)
        max_dist_idx = np.argmax(dists)
        M = transforms[max_dist_idx]

        #print(f"Point:{x},{y}",sources)
        
        #now apply the transform
        x_new,y_new=cv2.perspectiveTransform(np.array([[[x,y]]],dtype=np.float32),M)[0][0]
        return x_new+x_offsets[max_dist_idx],y_new
    
    def pose_dataset_line(self,box_margin=5.0,fullpage_box=False):
        #all the coordinates here are relative 0.0-100.0%
        top_y=min(y for x,y in [self.tl,self.tm,self.tr])
        bottom_y=max(y for x,y in [self.bl,self.bm,self.br])
        left_x=min(x for x,y in [self.tl,self.bl])
        right_x=max(x for x,y in [self.tr,self.br])
        center_xy=((left_x+right_x)/2,(top_y+bottom_y)/2)
        width,height=right_x-left_x+box_margin,bottom_y-top_y+box_margin #let me grow the book by few %
        if fullpage_box:
            line=f"0 0.5 0.5 1.0 1.0"
        else:
            line=f"0 {trim01(center_xy[0]/100.0)} {trim01(center_xy[1]/100.0)} {trim01(width/100.0)} {trim01(height/100.0)}"
        for x,y in [self.tl,self.tm,self.tr,self.bl,self.bm,self.br]:
            line+=f" {trim01(x/100.0)} {trim01(y/100.0)}"
        return line

    def segm_dataset_line(self):
        line="0 "
        for x,y in [self.tl,self.tm,self.bm,self.bl]:
            line+=f" {trim01(x/100.0)} {trim01(y/100.0)}"
        line+="\n"
        line+="1 "
        for x,y in [self.tm,self.tr,self.br,self.bm]:
            line+=f" {trim01(x/100.0)} {trim01(y/100.0)}"
        return line


def trim01(v):
    v=max(0.0,v)
    v=min(1.0,v)
    return v
   
def extract_kpoints_from_json(item):
    annotations=item["annotations"][0] #this is my annotation
    
    points=[(point["value"]["x"], point["value"]["y"]) for point in annotations["result"]]
    if len(points)!=6:
        raise ValueError("There should be 6 keypoints")
    
    points.sort(key=lambda x: x[1]) #sorted on Y, i.e. first 3 are the top row of points, second 3 are the bottom row of points
    points[:3] = sorted(points[:3], key=lambda x: x[0])
    points[3:] = sorted(points[3:], key=lambda x: x[0])
    
    tl, tm, tr, bl, bm, br = points
    return KPoints(tl, tm, tr, bl, bm, br)


def to_yolo(inp_files,out_dir,section,img_base2path,xml_base2path,skew_rskew_meta_dict):
    #Make sure out_dir/section/images and out_dir/section/labels exist
    os.makedirs(f"{out_dir}/{section}/images", exist_ok=True)
    os.makedirs(f"{out_dir}/{section}/labels-pose", exist_ok=True)
    os.makedirs(f"{out_dir}/{section}/labels-pose-fpbox", exist_ok=True)
    os.makedirs(f"{out_dir}/{section}/labels-segm", exist_ok=True)
    os.makedirs(f"{out_dir}/{section}/deskewed_images", exist_ok=True)
    os.makedirs(f"{out_dir}/{section}/deskewed_xmls", exist_ok=True)
    for inp_file in inp_files: #these are the jsons
        with open(inp_file, 'r') as file: #this be one json
            data = json.load(file)
            for item in data:
                if len(item["annotations"])!=1: #there should be 1 annotation, skip if not
                    print(f"Skipping {item['file_upload']} there are no annotations!")
                    continue
                
                basename=item["file_upload"].split("-",1)[1] #remove the sample number
                orig_img_path,collection_name=img_base2path[basename]
                #do we have an xml file for this?
                
                try:
                    kpoints=extract_kpoints_from_json(item)
                    kpoints.img_path=orig_img_path
                    
                except ValueError as e:
                    print(f"Skipping {item['file_upload']}: {e}")
                    kpoints=None
                if kpoints:
                    deskewed_img,skew_rskew_meta=kpoints.full_deskew()
                else:
                    deskewed_img,skew_rskew_meta=None,None

                #1) save the image into out_dir/collection/basename.jpg
                img_path=f"{out_dir}/{section}/deskewed_images/man-ds-{collection_name}"
                os.makedirs(img_path, exist_ok=True)
                
                #remove the transkribus number, if any
                parts=basename.split("_",1)
                if len(parts)==2 and parts[0].isnumeric(): #yeah...
                    basename=parts[1]
                #note we still have a .jpg at the end!
                if deskewed_img is None:
                    deskewed_img=cv2.imread(orig_img_path)

                if deskewed_img.shape[0] > 2500 or deskewed_img.shape[1] > 2500:
                    scale_factor = min(2500 / deskewed_img.shape[0], 2500 / deskewed_img.shape[1])
                    deskewed_img = cv2.resize(deskewed_img, None, fx=scale_factor, fy=scale_factor)
                else:
                    scale_factor = 1.0
                cv2.imwrite(f"{img_path}/mands-{basename}", deskewed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

                if not kpoints:
                    continue
                #1) save the image into out_dir/collection/basename.jpg (original)
                img_path=f"{out_dir}/{section}/images/{basename}"
                cv2.imwrite(img_path, cv2.imread(orig_img_path), [cv2.IMWRITE_JPEG_QUALITY, 90])
                #2) save the label into out_dir/collection/basename.txt
                with open(f"{out_dir}/{section}/labels-pose/{basename.replace('.jpg','.txt')}","wt") as lab_file:
                    print(kpoints.pose_dataset_line(box_margin=2.0), file=lab_file)
                with open(f"{out_dir}/{section}/labels-pose-fpbox/{basename.replace('.jpg','.txt')}","wt") as lab_file:
                    print(kpoints.pose_dataset_line(fullpage_box=True), file=lab_file)
                with open(f"{out_dir}/{section}/labels-segm/{basename.replace('.jpg','.txt')}","wt") as lab_file:
                    print(kpoints.segm_dataset_line(), file=lab_file)
                #3) If we have the xml for this image, now is the time to deskew it too
                xml_base=basename.replace(".jpg",".xml")
                if xml_base in xml_base2path:
                    print("Deskewing xml for",basename)
                    os.makedirs(f"{out_dir}/{section}/deskewed_xmls/man-ds-xml-{collection_name}",exist_ok=True)
                    all_polygons=kpoints.deskew_xml_annotations(xml_base2path[xml_base],f"{out_dir}/{section}/deskewed_xmls/man-ds-xml-{collection_name}/mands-{xml_base}",scale_factor=scale_factor)
                    write_debug_deskew_img(deskewed_img,all_polygons,f"{out_dir}/{section}/deskewed_xmls/man-ds-xml-{collection_name}/debug-{basename}")
                skew_rskew_meta_dict["mands-"+basename]=skew_rskew_meta

def write_debug_deskew_img(img,all_polygons,out_fname):
    overlay = img.copy()
    for polygon in all_polygons:
        cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], color=(0, 0, 255))
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.imwrite(out_fname,img,[cv2.IMWRITE_JPEG_QUALITY, 90])

#     rec_width_percent=10
#     #rec_width_percent=rec_width/annotations[0]["result"][0]["original_width"]
    
#     top_left=kpoint_0[0]-rec_width_percent/2,kpoint_0[1]
#     top_right=kpoint_0[0]+rec_width_percent/2,kpoint_0[1]
#     bottom_left=kpoint_1[0]-rec_width_percent/2,kpoint_1[1]
#     bottom_right=kpoint_1[0]+rec_width_percent/2,kpoint_1[1]

#     return top_left,top_right,bottom_left,bottom_right

# def to_yolo(inp_file,out_dir,img_source_dir):
#     with open(inp_file, 'r') as file:
#         data = json.load(file)
#         for item in data:
#             if len(item["annotations"])!=1: #there should be 1 annotation, skip if not
#                 continue
#             if len(item["annotations"][0]["result"])!=2: #there should be 2 keypoints, skip if not
#                 continue
#             top_left,top_right,bottom_left,bottom_right=extract_bb(item)
#             #Now I need to make the Yolo OBB files out of this
#             img_name=item["file_upload"].replace(".jpg","")
#             label_path=f"{out_dir}/labels/{section(img_name)}"
#             os.makedirs(label_path, exist_ok=True)
#             with open(f"{label_path}/{img_name}.txt","wt") as lab_file:
#                 print("0", end=" ", file=lab_file)
#                 for corner in top_left, top_right, bottom_left, bottom_right:
#                     print(f"{corner[0]/100} {corner[1]/100}", end=" ", file=lab_file)
#                 print("", file=lab_file)
#             img_path=f"{out_dir}/images/{section(img_name)}"
#             os.makedirs(img_path, exist_ok=True)
#             os.system(f"cp {img_source_dir}/{img_name}.jpg {img_path}/{img_name}.jpg")

#     yaml_data = {
#         'path': '.',
#         'train': 'images/train',
#         'val': 'images/val',
#         'test': 'images/test',
#         'names': {
#             0: 'midpoint'
#         }
#     }

#     with open(f'{out_dir}/dataset.yaml', 'w') as yaml_file:
#         yaml.dump(yaml_data, yaml_file)
    

def section(fname):
    hashed_value = hashlib.sha256(fname.encode("utf-8")).hexdigest()
    m = int(hashed_value, 16) % 10
    if m==0:
        return "test"
    elif m==1:
        return "val"
    else:
        return "train"
    
    
def gather_images(img_source_dir):
    all_img_files=glob.glob(f"{img_source_dir}/*/*.jpg")
    name2path={}
    for fname in all_img_files:
        path_parts=fname.split("/")
        basename=path_parts[-1]
        collection_name=path_parts[-2]
        name2path[basename]=(fname,collection_name)
    return name2path

def gather_xml_files(img_source_dirs):
    name2path={}
    all_xml_files=[]
    for img_source_dir in img_source_dirs:
        all_xml_files.extend(glob.glob(f"{img_source_dir}/*.xml"))

    for fname in all_xml_files:
        basename=os.path.basename(fname)
        name2path[basename]=fname
    return name2path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert label studio format to YOLOv8 format')
    parser.add_argument('--train-files', nargs='+', default=[], help='Path to the json annotation of the training set')
    parser.add_argument('--val-files', nargs='+', default=[], help='Path to the json annotation of the validation set')
    parser.add_argument('--test-files', nargs='+', default=[], help='Path to the json annotation of the test set')
    parser.add_argument('--img-source-dir', default="data/images", help='Path to the image source directory')
    parser.add_argument('--xml-source-dir', nargs='+', default=[], help='Path to the source directory with the transkribus xml annotation files')
    parser.add_argument('--tr-matrices',default=None,help="store the matrices into this pickle file")
    args = parser.parse_args()

    os.makedirs("midpoints-yolov8", exist_ok=True)
    img_name2path=gather_images(args.img_source_dir)
    xml_name2path=gather_xml_files(args.xml_source_dir)
    

    skew_reskew_meta={} #key:mands_basename.jpg val:reskew metadata dict
    if args.train_files:
        to_yolo(args.train_files,"midpoints-yolov8","train",img_name2path,xml_name2path,skew_reskew_meta)
    if args.val_files:
        to_yolo(args.val_files,"midpoints-yolov8","val",img_name2path,xml_name2path,skew_reskew_meta)
    if args.test_files:
        to_yolo(args.test_files,"midpoints-yolov8","test",img_name2path,xml_name2path,skew_reskew_meta)
