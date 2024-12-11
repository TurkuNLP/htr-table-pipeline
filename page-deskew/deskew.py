import cv2
import zipfile
import numpy as np
import argparse
import os
import lstudio2yolov8
from ultralytics import YOLO
import more_itertools
import sys
import glob
import traceback

#gathered from training data with python3 estimate_vangle.py ../../htr-annotations/deskew-keypoints/train_part*.json
angle_limit_data={'l': {'mean': 0.24755304277022264, 'stdev': 0.5850729424142381, 'min': -1.9078686234196738, 'max': 2.339893671562592}, 
                  'm': {'mean': -0.033317493513659334, 'stdev': 0.4102483258697504, 'min': -1.8664903884855306, 'max': 2.187532011056419}, 
                  'r': {'mean': -0.17899178022242473, 'stdev': 0.6119567565473318, 'min': -3.2559467759417657, 'max': 1.7164697009402943}}
for k in angle_limit_data:
    angle_limit_data[k]["min"]*=2 #loosen it up a bit
    angle_limit_data[k]["max"]*=2

def yield_images_from_zip(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for zinfo in zip_ref.infolist():
            if zinfo.filename.endswith('.jpg'):
                with zip_ref.open(zinfo) as file:
                    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    yield zinfo.filename,image

def stg1_keypoints(batch,yolo_stg1_pose_model,yolo_stg2_pose_model,args):
    images=[img for file_name,img in batch]
    ds_images=[]
    stg2_batch=[]
    yolo_results=yolo_stg1_pose_model.predict(images,conf=0.25,device=args.device) #this does batched inference, since we give a list on the input
    bypass=[] #images that we skip from the stg2 batch
    for (id,((file_name,img),yolo_result)) in enumerate(zip(batch,list(yolo_results))):
        _,kpoint_num,xy=yolo_result.keypoints.xy.shape
        if kpoint_num==6:
            kp=lstudio2yolov8.KPoints(*(yolo_result.keypoints.xyn*100.0).tolist()[0])
            kp.image=img
            kp.file_name=file_name
            kp.h,kp.w=img.shape[:2]
            kp.update_int_coords()
            stg2_batch.append((id,kp))
        else:
            print(f"Skipping {file_name} as it has {kpoint_num} keypoints",file=sys.stderr,flush=True)
            bypass.append(id)
    if stg2_batch:
        debug_coordinates=stg2_update_keypoints(stg2_batch,yolo_stg2_pose_model,args) #updates keypoints in-place
    else:
        debug_coordinates=[]
    assert len(stg2_batch)==len(debug_coordinates)
    for ((id1,kp),(id2,coord_list)) in zip(stg2_batch,debug_coordinates):
        assert id1==id2
        dimg,ds_meta=kp.full_deskew()
        debug_img = kp.image.copy()
        for (old_coords, new_coords) in coord_list:
            if old_coords is not None:
                cv2.circle(debug_img, tuple(old_coords), 40, (0, 0, 255), 4)
                cv2.circle(debug_img, tuple(old_coords), 5, (0, 0, 255), -1)
            if new_coords is not None:
                cv2.drawMarker(debug_img, tuple(new_coords), (0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=40, thickness=5)
        ds_images.append((id1,kp.file_name, dimg, debug_img, coord_list))
    for id,(file_name,img) in enumerate(batch):
        if id in bypass:
            ds_images.append((id,file_name, img, img, [(None,None)]*6))
    assert len(ds_images)==len(batch)
    return list(sorted(ds_images))

def highest_prediction_index(yolo_result,point_type):
    #Takes a yolo result with possibly many predictions and returns the index of the highest confidence prediction of the correct class
    if point_type in ("tm","bm"):
        cls="midpoint"
    else:
        cls="corner"
    #which class id does this correspond to?
    class_id=0 if yolo_result.names[0]==cls else 1 #ok this is dirty, but it works for 2 classes like we have
    #print(yolo_result.boxes.cls)
    #print(yolo_result.boxes.conf)
    for idx,pred_cls in enumerate(yolo_result.boxes.cls):
        if int(pred_cls)==class_id:
            return idx
    return None

def stg2_update_keypoints(batch,yolo_stg2_pose_model,args):
    stg2_batch_images=[]
    #batch of figures
    all_zoomed_corners=[]
    for id,kpoints in batch:
        zoomed_corners=kpoints.extract_zoomed_corners(fuzz=0.0,corner_size=0.15,flip=True)
        all_zoomed_corners.append(zoomed_corners)
        for (patch,center_point,keypoint_new_coords,flipH,flipV,point_type) in zoomed_corners:
            stg2_batch_images.append(patch)
    #this will now be batch times 6 predictions
    yolo_results=yolo_stg2_pose_model.predict(stg2_batch_images,max_det=5,conf=0.2,device=args.device)
    debug_output=[]
    for (id,kpoints),point_predictions_x6,corner_info_x6 in zip(batch,more_itertools.chunked(yolo_results,6),all_zoomed_corners): #there should be 6 predictions per image
        debug_output.append((id,[]))
        for point_prediction, (patch,center_point,keypoint_new_coords,flipH,flipV,point_type) in zip(point_predictions_x6,corner_info_x6):
            _,kpoint_count,two_=point_prediction.keypoints.xy.shape
            old_coords=np.array(getattr(kpoints, point_type+"_int"),int)
            if kpoint_count==0:
                #didn't find any keypoint, let's skip this one
                debug_output[-1][-1].append((old_coords,None))
                continue
            highest_idx=highest_prediction_index(point_prediction,point_type)
            if highest_idx is None:
                debug_output[-1][-1].append((old_coords,None))
                continue
            delta=point_prediction.keypoints.xy[highest_idx][0].cpu().numpy()-np.array(keypoint_new_coords,int)
            #print("xy shape",point_prediction.keypoints.xy.shape)
            if flipH:
                delta[0]*=-1
            if flipV:
                delta[1]*=-1
            new_coords=(old_coords+delta).astype(int)
            debug_output[-1][-1].append((old_coords,new_coords))
            #print("old_coords",old_coords,old_coords.__class__)
            #print("delta",delta)
            #Basically, we should only accept the correction if the corrected angle falls within the limits
            tb,lmr=point_type[0],point_type[1] #top/bottom, left/mid/right
            if tb=="t":
                top=new_coords.tolist()
                bottom=getattr(kpoints, "b"+lmr+"_int")
            else:
                bottom=new_coords.tolist()
                top=getattr(kpoints, "t"+lmr+"_int")
            angle=lstudio2yolov8.vertical_line_angle(top,bottom)
            if angle<angle_limit_data[lmr]["min"] or angle>angle_limit_data[lmr]["max"]:
                print(f"Angle {angle} for {point_type} is out of range {angle_limit_data[lmr]}",file=sys.stderr,flush=True)
            else:
                setattr(kpoints, point_type+"_int", new_coords.tolist())
            #print(f"point_type={point_type} delta={delta}")
        kpoints.update_relative_coords() #should run this since we touched

        #kpoints is the KPoints object from the batch
        #point_predictions is a list of 6 YOLO results
    return debug_output #list of [ [ (old_coords,new_coords),... ]  for each image in the batch]




def deskew(args):
    # Load models
    stg1_model = YOLO(args.stage1_model)
    if args.stage2_model is not None:
        stg2_model = YOLO(args.stage2_model)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.directory_out, exist_ok=True)
    if args.debug_directory_out is not None:
        os.makedirs(args.debug_directory_out, exist_ok=True)
    
    if args.zipfiles_in:
        all_zips=args.zipfiles_in
    elif args.zipfile_glob:
        all_zips=list(sorted(glob.glob(args.zipfile_glob)))
    else:
        all_zips=[]
    if not all_zips:
        print("No zipfiles to process",file=sys.stderr,flush=True)
        return  #nothing to do

    for counter,zipfile_in in enumerate(all_zips):
        print(f"Processing {zipfile_in} {counter+1}/{len(all_zips)}",file=sys.stderr,flush=True)
        try:
            zip_out_path = os.path.join(args.directory_out, "autods_"+os.path.basename(zipfile_in))
            if args.debug_directory_out is not None:
                zip_debug_path = os.path.join(args.debug_directory_out, "debug_"+os.path.basename(zipfile_in))

            with zipfile.ZipFile(zip_out_path, 'w') as zip_out,\
                (zipfile.ZipFile(zip_debug_path, 'w') if args.debug_directory_out is not None else None) as zip_debug:
                img_zipfile_generator=more_itertools.ichunked(yield_images_from_zip(zipfile_in),args.batchsize)
                for batch in img_zipfile_generator:
                    deskewed=stg1_keypoints(list(batch),stg1_model,stg2_model,args)
                    for id,file_name,img,img_debug,debug_coord_list in deskewed:
                        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        file_name_parts = os.path.split(file_name)
                        new_file_name = os.path.join(file_name_parts[0], "autods_" + file_name_parts[1])
                        zip_out.writestr(new_file_name, buffer.tobytes())                
                        if zip_debug is not None:
                            _, buffer = cv2.imencode('.jpg', img_debug, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                            debug_file_name = os.path.join(file_name_parts[0], "debug_" + file_name_parts[1])
                            zip_debug.writestr(debug_file_name, buffer.tobytes())
                            debug_file_name_txt = os.path.join(file_name_parts[0], "coords_" + file_name_parts[1].replace(".jpg", ".txt"))
                            debug_coords_str="#DBG tl_old tl_new tm_old tm_new tr_old tr_new br_old br_new bm_old bm_new bl_old bl_new\nDBG"
                            for old_coords, new_coords in debug_coord_list:
                                debug_coords_str+=" "
                                if old_coords is not None:
                                    debug_coords_str+=f"{old_coords[0]},{old_coords[1]}"
                                else:
                                    debug_coords_str+="None,None"
                                debug_coords_str+=" "
                                if new_coords is not None:
                                    debug_coords_str+=f"{new_coords[0]},{new_coords[1]}"
                                else:
                                    debug_coords_str+="None,None"
                            zip_debug.writestr(debug_file_name_txt, debug_coords_str.encode("utf-8"))
        except:
            print(f"FAILED {zipfile_in}",file=sys.stderr,flush=True)
            traceback.print_exc()

def parse_args():
    parser = argparse.ArgumentParser(description='Deskew images from a zip file.')
    parser.add_argument('--stage1-model', type=str, required=True, help='Path to the stage 1 YOLO model.')
    parser.add_argument('--stage2-model', type=str, default=None, help='Path to the stage 2 YOLO model.')
    parser.add_argument('--zipfiles-in', nargs="+", default=[], type=str, help='Path to the input zip file(s).')
    parser.add_argument('--zipfile-glob', type=str, default=None, help='if set and zipfiles-in is empty, will glob for zip files')
    parser.add_argument('--directory-out', type=str, default="out", help='Path to the output directory.')
    parser.add_argument('--debug-directory-out', type=str, default=None, help='Path to the output directory for debug files.')
    parser.add_argument('--batchsize', type=int, default=2, help='Batch size for processing images.')
    parser.add_argument('--device', type=str, default="cpu", help='Device to run the model on. cpu, cuda:0, cuda:1, etc.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    deskew(args)
    

    
