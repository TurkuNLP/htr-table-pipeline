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
        #these are relative 0-100
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

    def update_int_coords(self):
        #compute absolute coordinates based on the float ones
        def calculate_int_coordinates(point):
            return (int(self.w * point[0] / 100), int(self.h * point[1] / 100))
        self.tl_int = calculate_int_coordinates(self.tl)
        self.tm_int = calculate_int_coordinates(self.tm)
        self.tr_int = calculate_int_coordinates(self.tr)
        self.bl_int = calculate_int_coordinates(self.bl)
        self.bm_int = calculate_int_coordinates(self.bm)
        self.br_int = calculate_int_coordinates(self.br)
    
    def update_relative_coords(self):
        #compute relative coordinates based on the int ones
        def calculate_rel_coordinates(point):
            return (point[0] * 100 / self.w, point[1] * 100 / self.h)
        self.tl = calculate_rel_coordinates(self.tl_int)
        self.tm = calculate_rel_coordinates(self.tm_int)
        self.tr = calculate_rel_coordinates(self.tr_int)
        self.bl = calculate_rel_coordinates(self.bl_int)
        self.bm = calculate_rel_coordinates(self.bm_int)
        self.br = calculate_rel_coordinates(self.br_int)

    def possibly_read_image(self):
        if self.image is None and self.img_path:
            self.image = cv2.imread(self.img_path)
            self.w,self.h=self.image.shape[1],self.image.shape[0]
            self.update_int_coords()

    def extract_zoomed_corners(self,fuzz=0.05,corner_size=0.25,flip=False):
        """Extract the zoomed corners, with a bit of fuzz, these are given as percentages (0,1)
        
        if flip is True, the corner will always be top left and the mid will always be on the upper edge"""
        zoomed_corners=[]
        self.possibly_read_image()
        int_fuzz = int(self.w * fuzz)
        int_corner_size = int(self.w * corner_size)
        for point,flipH,flipV,point_type in [(self.tl_int,False,False,"tl"),(self.tm_int,False,False,"tm"),(self.tr_int,True,False,"tr"),(self.bl_int,False,True,"bl"),(self.bm_int,False,True,"bm"),(self.br_int,True,True,"br")]:
            #Pretend the point is randomly up to int_fuzz off
            #So this is the center of the corner area
            if fuzz>0:
                center_point = np.array([point[0] + np.random.randint(-int_fuzz, int_fuzz), point[1] + np.random.randint(-int_fuzz, int_fuzz)]).flatten().tolist()
            else:
                center_point=point
            patch, keypoint_new_coords, _ = crop_patch(self.image, center_point, point, (int_corner_size, int_corner_size))
            if flip:
                patch,flipped_points=self.flip(patch,[keypoint_new_coords],flipH,flipV)
                keypoint_new_coords=flipped_points[0]
            zoomed_corners.append((patch, center_point, keypoint_new_coords,flipH,flipV,point_type))
        return zoomed_corners

    def flip(self,img,points,flipH,flipV):
        H,W=img.shape[:2]
        if flipH:
            img=cv2.flip(img,1)
            points=[(W-x,y) for x,y in points]
        if flipV:
            img=cv2.flip(img,0)
            points=[(x,H-y) for x,y in points]
        return img,points


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
    
    def deskew_xml_annotations(self,xml_filename_in,xml_filename_out,deskewed_img_size,scale_factor=1.0,filename_prepend="mands-"):
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
                page_elem.attrib['imageFilename']=filename_prepend+page_elem.attrib['imageFilename']
            else:
                assert False, f"{xml_file} doesn't have any Page element"
            assert self.w==xml_width and self.h==xml_height, f"Image dimensions do not match: image ({self.w}x{self.h}), xml ({xml_width}x{xml_height})"
            page_elem.attrib['imageWidth']=str(int(deskewed_img_size[0]))
            page_elem.attrib['imageHeight']=str(int(deskewed_img_size[1]))
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


def vertical_line_angle(p_top,p_bottom):
    return math.degrees(math.atan2(p_bottom[1]-p_top[1],p_bottom[0]-p_top[0]))-90


def crop_patch(image,center,keypoint,patchsize):
    """
    image
    center -> center of the patch, given as (x,y)
    keypoint -> a keypoint on the patch, given as (x,y)
    patchsize -> given as (dx,dy); center-(dx,dy) will be top-left,
                center+(dx,dy) will be bottom right, i.e. the width will be (2dx,2dy)

    returns:
    (cropped_patch, -> image
     keypoint_new_coords, -> coordinates of the keypoint in the cropped patch
     bbox -> the bounding box of the patch for debug purposes)
    """
    keypoint=np.array([keypoint],int)
    center=np.array([center],int)
    dxdy=np.array([patchsize],int) #width,height

    # Bounding box, not respecting image boundaries (yet)     
    bbox=np.stack([center+dxdy*[-1,-1],center+dxdy*[1,-1],center+dxdy*[1,1],center+dxdy*[-1,1]]).squeeze()
    # amount of left and top crop, as negative values, i.e. areas of the patch which
    # fall outside of the original image
    # the coordinates in the new cropped image must be corrected by this amount
    trim_correction=np.min(bbox,axis=0) 
    trim_correction[trim_correction>0]=0

    # Now I can trim the bbox
    # the bbox is in (x,y) order but shape is (height,width)
    bbox[:, 0] = np.clip(bbox[:, 0], 0, image.shape[1]) 
    bbox[:, 1] = np.clip(bbox[:, 1], 0, image.shape[0])

    center_new_coordinates=dxdy+trim_correction #center, in the new image
    keypoint_new_coordinates=center_new_coordinates+(keypoint-center)

    # Now I can actually prepare the patch, which means I need to crop
    # the patch bounding box by the image
    # I cannot believe there ain't any function for this in CV2
    # (surely there is, just didn't find it)
    x_coords = bbox[:, 0]
    y_coords = bbox[:, 1]
    # Calculate the center of the rectangle
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    width = x_max - x_min
    height = y_max - y_min
    center_x = x_min + width // 2
    center_y = y_min + height // 2
    # Define the size of the patch to extract
    patch_size = (int(width), int(height))
    #print("patch_size",patch_size)
    #print("center",(center_x,center_y))
    cropped_img = cv2.getRectSubPix(image, patch_size, (float(center_x), float(center_y)))
    return cropped_img,keypoint_new_coordinates.flatten().tolist(),bbox

# ### Let's test
# image = cv2.imread("Test.png")
# image_copy=image.copy()

# keypoint=(500,270)
# center=(450,300)
# patchsize=(100,300)

# cropped_img,kpoint_new_coords,bbox=crop_patch(image,center,keypoint,patchsize)


# cv2.drawMarker(image_copy, center, color=(0, 255, 0), markerType=cv2.MARKER_CROSS,
#             markerSize=20, thickness=2, line_type=cv2.LINE_AA)
# cv2.drawMarker(image_copy, keypoint, color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
#             markerSize=20, thickness=2, line_type=cv2.LINE_AA)
# cv2.polylines(image_copy, [bbox], isClosed=True, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cv2_imshow(image_copy)

# cv2.drawMarker(cropped_img, kpoint_new_coords, color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
#             markerSize=20, thickness=2, line_type=cv2.LINE_AA)
# cv2_imshow(cropped_img)


def trim01(v):
    v=max(0.0,v)
    v=min(1.0,v)
    return v
   
def extract_kpoints_from_labelstudio_json(item):
    annotations=item["annotations"][0] #this is my annotation
    
    points=[(point["value"]["x"], point["value"]["y"]) for point in annotations["result"]]
    if len(points)!=6:
        raise ValueError("There should be 6 keypoints")
    
    points.sort(key=lambda x: x[1]) #sorted on Y, i.e. first 3 are the top row of points, second 3 are the bottom row of points
    #sort the top row by X
    points[:3] = sorted(points[:3], key=lambda x: x[0])
    #sort the bottom row by X
    points[3:] = sorted(points[3:], key=lambda x: x[0])
    
    tl, tm, tr, bl, bm, br = points
    return KPoints(tl, tm, tr, bl, bm, br)


def to_yolo(inp_files,section,img_basename2path,xml_basename2path,skew_rskew_meta_dict,args):
    #Make sure out_dir/section/images and out_dir/section/labels exist
    os.makedirs(f"{args.dataset_out_dir}/{section}/images", exist_ok=True)
    os.makedirs(f"{args.dataset_out_dir}/{section}/labels", exist_ok=True)
    os.makedirs(f"{args.dataset_out_dir}/{section}/deskewed_images", exist_ok=True)
    os.makedirs(f"{args.dataset_out_dir}/{section}/deskewed_xmls", exist_ok=True)

    os.makedirs(f"{args.dataset_stg2_corners_out_dir}/{section}/images", exist_ok=True)
    os.makedirs(f"{args.dataset_stg2_corners_out_dir}/{section}/debug-images", exist_ok=True)
    os.makedirs(f"{args.dataset_stg2_corners_out_dir}/{section}/labels", exist_ok=True)

    for inp_file in inp_files: #these are the jsons
        with open(inp_file, 'r') as file: #this be one json
            data = json.load(file)
            for item in data:
                #Get the basename and collection name
                
                #1) remove the sample number from labelstudio
                basename=item["file_upload"].split("-",1)[1]
                #find the original image file's path by its basename 
                orig_img_path,collection_name=img_basename2path[basename]

                #2) some images have the transkribus number by accident, trim it out if found
                parts=basename.split("_",1)
                if len(parts)==2 and parts[0].isnumeric(): #yeah...
                    basename=parts[1]
                #note we still have a .jpg at the end!
                
                if len(item["annotations"])!=1: #there should be 1 annotation, skip if not
                    print(f"{item['file_upload']}: there are no annotations!")
                    kpoints=None
                else:
                    try:
                        kpoints=extract_kpoints_from_labelstudio_json(item)
                        kpoints.img_path=orig_img_path
                    except ValueError as e:
                        print(f"{item['file_upload']}: {e}")
                        kpoints=None
                
                #3) DESKEW
                #I found the 6-point annotation and can now deskew the whole thing
                if kpoints is not None:
                    deskewed_img,skew_rskew_meta=kpoints.full_deskew()
                else: #no keypoints, just copy the image over
                    deskewed_img,skew_rskew_meta=cv2.imread(orig_img_path),None

                #1) save the image into out_dir/collection/man-ds-basename.jpg
                mands_img_pathdir=f"{args.dataset_out_dir}/{section}/mandeskewed_images/mands-{collection_name}"
                os.makedirs(mands_img_pathdir, exist_ok=True)
                
                if deskewed_img.shape[0] > 2500 or deskewed_img.shape[1] > 2500:
                    scale_factor = min(2500 / deskewed_img.shape[0], 2500 / deskewed_img.shape[1])
                    deskewed_img = cv2.resize(deskewed_img, None, fx=scale_factor, fy=scale_factor)
                else:
                    scale_factor = 1.0
                cv2.imwrite(f"{mands_img_pathdir}/mands-{basename}", deskewed_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

                #No labels to write for this one, so we're done if there are no kpoints
                if not kpoints:
                    continue
                skew_rskew_meta_dict["mands-"+basename]=skew_rskew_meta
                #4) WRITE OUT LABELS FOR TRAINING THE DESKEW RECOGNITION

                #4.1) save the image into out_dir/collection/basename.jpg (original)
                img_path=f"{args.dataset_out_dir}/{section}/images/{basename}"
                cv2.imwrite(img_path, cv2.imread(orig_img_path), [cv2.IMWRITE_JPEG_QUALITY, 90])
                #4.2) save the label into out_dir/collection/basename.txt
                with open(f"{args.dataset_out_dir}/{section}/labels/{basename.replace('.jpg','.txt')}","wt") as lab_file:
                    print(kpoints.pose_dataset_line(box_margin=2.0), file=lab_file)
                #3) If we have the xml for this image, now is the time to deskew it too
                xml_basename=basename.replace(".jpg",".xml")
                if xml_basename in xml_basename2path:
                    print("Deskewing xml for",basename)
                    xml_img_pathdir=f"{args.dataset_out_dir}/{section}/deskewed_xmls/mands-xml-{collection_name}"
                    os.makedirs(xml_img_pathdir,exist_ok=True)
                    all_polygons=kpoints.deskew_xml_annotations(xml_basename2path[xml_basename],
                                                                f"{xml_img_pathdir}/mands-{xml_basename}",
                                                                deskewed_img_size=(deskewed_img.shape[1],deskewed_img.shape[0]),
                                                                scale_factor=scale_factor)
                    debug_xml_img_pathdir=f"{args.dataset_out_dir}/{section}/deskewed_xmls-debug-images/mands-xml-{collection_name}"
                    os.makedirs(debug_xml_img_pathdir,exist_ok=True)
                    write_debug_deskew_img(deskewed_img,all_polygons,f"{debug_xml_img_pathdir}/debug-{basename}")
                
                #4) make stage2 images on the corners
                zoomed_corners=kpoints.extract_zoomed_corners(fuzz=0.05,corner_size=0.15,flip=True)
                for idx,(patch,center_point,keypoint_new_coords,flipH,flipV,point_type) in enumerate(zoomed_corners):
                    patch_width,patch_height=patch.shape[1],patch.shape[0]
                    #Let's make sure the Yolo region is square no matter what the patch size comes out to be
                    if patch_width>=patch_height:
                        w,h=0.2*patch_height/patch_width,0.2
                    else:
                        w,h=0.2,0.2*patch_width/patch_height
                    if point_type in ["tm","bm"]:
                        cls="1"
                    else:
                        cls="0"
                    cv2.imwrite(f"{args.dataset_stg2_corners_out_dir}/{section}/images/{basename.replace('.jpg','')}_{idx}.jpg",patch,[cv2.IMWRITE_JPEG_QUALITY, 90])
                    with open(f"{args.dataset_stg2_corners_out_dir}/{section}/labels/{basename.replace('.jpg','')}_{idx}.txt","wt") as lab_file:
                        print(cls, end=" ", file=lab_file)
                        print(f"{keypoint_new_coords[0]/patch_width} {keypoint_new_coords[1]/patch_height} {w} {h}", end=" ", file=lab_file)
                        print(f"{keypoint_new_coords[0]/patch_width} {keypoint_new_coords[1]/patch_height}", end=" ", file=lab_file)
                        print("", file=lab_file)
                    write_debug_corner_img(patch,keypoint_new_coords,f"{args.dataset_stg2_corners_out_dir}/{section}/debug-images/debug-{basename.replace('.jpg','')}_{idx}.jpg")

def write_debug_deskew_img(img,all_polygons,out_fname):
    overlay = img.copy()
    for polygon in all_polygons:
        cv2.fillPoly(overlay, [np.array(polygon, dtype=np.int32)], color=(0, 55, 55))
        cv2.polylines(overlay, [np.array(polygon, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)
    cv2.addWeighted(overlay, 0.2, img, 0.8, 0, img)
    cv2.imwrite(out_fname,img,[cv2.IMWRITE_JPEG_QUALITY, 90])

def write_debug_corner_img(img,keypoint_new_coords,out_fname):
    overlay = img.copy()
    cv2.drawMarker(overlay, keypoint_new_coords, color=(255, 0, 0), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=2, line_type=cv2.LINE_AA)
    cv2.imwrite(out_fname,overlay,[cv2.IMWRITE_JPEG_QUALITY, 90])

    
def gather_images(img_source_dir):
    all_img_files=glob.glob(f"{img_source_dir}/*/*.jpg")
    basename2path={}
    for fname in all_img_files:
        path_parts=fname.split("/")
        basename=path_parts[-1]
        collection_name=path_parts[-2]
        basename2path[basename]=(fname,collection_name)
    return basename2path

def gather_xml_files(img_source_dirs):
    basename2path={}
    all_xml_files=[]
    for img_source_dir in img_source_dirs:
        all_xml_files.extend(glob.glob(f"{img_source_dir}/*.xml"))

    for fname in all_xml_files:
        basename=os.path.basename(fname)
        basename2path[basename]=fname
    return basename2path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert label studio format to YOLOv8 format')
    parser.add_argument('--train-files', nargs='+', default=[], help='Path to the json annotations of the training set')
    parser.add_argument('--val-files', nargs='+', default=[], help='Path to the json annotations of the validation set')
    parser.add_argument('--test-files', nargs='+', default=[], help='Path to the json annotations of the test set')
    parser.add_argument('--img-source-dir', default="data/images", help='Path to the image source directory')
    parser.add_argument('--xml-source-dir', nargs='+', default=[], help='Path to the source directory with the transkribus xml annotation files')
    parser.add_argument('--tr-matrices',default=None,help="Store the matrices into this pickle file")
    parser.add_argument('--dataset-out-dir', default="midpoints-yolov8", help='Path to the output directory of the main dataset')
    parser.add_argument('--dataset-stg2-corners-out-dir', default="midpoints-yolov8-corners", help='Path to the output directory of the stage2 corner dataset')
    args = parser.parse_args()

    os.makedirs(args.dataset_out_dir, exist_ok=True)
    os.makedirs(args.dataset_stg2_corners_out_dir, exist_ok=True)
    img_basename2path=gather_images(args.img_source_dir)
    xml_basename2path=gather_xml_files(args.xml_source_dir)
    

    skew_reskew_meta={} #key:mands_basename.jpg val:reskew metadata dict
    if args.train_files:
        to_yolo(args.train_files,"train",img_basename2path,xml_basename2path,skew_reskew_meta,args)
    if args.val_files:
        to_yolo(args.val_files,"val",img_basename2path,xml_basename2path,skew_reskew_meta,args)
    if args.test_files:
        to_yolo(args.test_files,"test",img_basename2path,xml_basename2path,skew_reskew_meta,args)
