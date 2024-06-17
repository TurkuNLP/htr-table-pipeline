"""
Configuration
"""

import importlib.util

import configparser
import json

class Options():
    def __init__(self, cfg_file):

        package_name = 'pytesseract'
        self.package_tesseract = False
        if importlib.util.find_spec(package_name) != None:
             self.package_tesseract = True

        package_name = 'detectron2'
        self.package_detectron2 = False
        if importlib.util.find_spec(package_name) != None:
             self.package_detectron2 = True

        package_name = 'kraken'
        self.package_kraken = False
        if importlib.util.find_spec(package_name) != None:
             self.package_kraken = True

        config = configparser.ConfigParser()
        config.read(cfg_file)

        # TEXT
        self.text_recognize = None
        if "text_recognize" in config["TEXT"] : self.text_recognize = config["TEXT"]["text_recognize"]

        self.text_recognize_engine = "tesseract"

        if "text_recognize_engine" in config["TEXT"] :
            self.text_recognize_engine = config["TEXT"]["text_recognize_engine"]

        self.text_recognize_tesseract_language = "eng"
        if "text_recognize_tesseract_language" in config["TEXT"] :  self.text_recognize_tesseract_language = config["TEXT"]["text_recognize_tesseract_language"]

        self.text_recognize_tesseract_cmd = None
        if "text_recognize_tesseract_cmd" in config["TEXT"] :  
            import pytesseract
            self.text_recognize_tesseract_cmd = config["TEXT"]["text_recognize_tesseract_cmd"]
            pytesseract.pytesseract.tesseract_cmd = config["TEXT"]["text_recognize_tesseract_cmd"]

        self.text_recognize_model = None
        self.text_recognize_processor = None
        if "text_recognize_model_path" in config["TEXT"]:
            self.text_recognize_model_path = config["TEXT"]["text_recognize_model_path"]
            self.text_recognize_processor_path = "microsoft/trocr-base-handwritten"

        if self.text_recognize_engine.lower() == "tesseract":
            self.text_recognize_custom_config = r'--oem 3 --psm 7'

        # Development
        self.DEBUG = False
        if "DEBUG" in config["DEVELOPMENT"] : self.DEBUG = config["DEVELOPMENT"].getboolean("DEBUG")
        
        self.line_classes = {}
        if "line_classes" in config["CLASSIFICATION"]: self.line_classes = json.loads(config["CLASSIFICATION"]["line_classes"])
        
        self.region_classes = {}
        if "region_classes" in config["CLASSIFICATION"]: self.region_classes = json.loads(config["CLASSIFICATION"]["region_classes"])
            
        self.L_UNKNOWN = len(self.line_classes) - 1
        self.R_UNKNOWN = len(self.region_classes) - 1
        
        self.line_classLabels = [x for x in self.line_classes.keys()]
        self.region_classLabels = [x for x in self.region_classes.keys()]

        self.reverse_line_class = {v:k for k,v in self.line_classes.items()}
        self.reverse_region_class = {v:k for k,v in self.region_classes.items()}
        
        self.merged_line_elements = []
        if "merged_line_elements" in config["CLASSIFICATION"]: self.merged_line_elements = json.loads(config["CLASSIFICATION"]["merged_line_elements"])

        self.merged_region_elements = []
        if "merged_region_elements" in config["CLASSIFICATION"]: self.merged_region_elements = json.loads(config["CLASSIFICATION"]["merged_region_elements"])

        self.merge_type = "full"
        if "merge_type" in config["CLASSIFICATION"]: self.merge_type = config["CLASSIFICATION"]["merge_type"]


        RO_line_groups = []
        if "RO_line_groups" in config["CLASSIFICATION"]: self.RO_line_groups = json.loads(config["CLASSIFICATION"]["RO_line_groups"])

        RO_region_groups = []
        if "RO_region_groups" in config["CLASSIFICATION"]: self.RO_region_groups = json.loads(config["CLASSIFICATION"]["RO_region_groups"])

        one_page_per_image = True
        if "one_page_per_image" in config["CLASSIFICATION"]: self.one_page_per_image = config["CLASSIFICATION"].getboolean("one_page_per_image")

# line parameters

        self.overlap_threshold = 0.1 
        if "overlap_threshold" in config["LINE"]: self.overlap_threshold = float(config["LINE"]["overlap_threshold"])

        self.line_merge_limit = 10 
        if "line_merge_limit" in config["LINE"]: self.line_merge_limit = float(config["LINE"]["line_merge_limit"])
        
        self.line_merge = False 
        if "line_merge" in config["LINE"]: self.line_merge = config["LINE"].getboolean("line_merge")
        
        self.dist_limit = 0.1
        if "dist_limit" in config["LINE"]: self.dist_limit = float(config["LINE"]["dist_limit"])
        
        self.default_x_offset = 10
        if "default_x_offset" in config["LINE"]: self.default_x_offset = float(config["LINE"]["default_x_offset"])
        
        self.default_y_offset = 10
        if "default_y_offset" in config["LINE"]: self.default_y_offset = float(config["LINE"]["default_y_offset"])
        
        self.default_y_offset_multiplier = 0.9 
        if "default_y_offset_multiplier" in config["LINE"]: self.default_y_offset_multiplier = float(config["LINE"]["default_y_offset_multiplier"])
        
        self.baseline_sample_size = 200
        if "default_y_offset_multiplier" in config["LINE"]: self.default_y_offset_multiplier = float(config["LINE"]["default_y_offset_multiplier"])
        
        self.baseline_offset_multiplier = 0.1 
        if "default_y_offset_multiplier" in config["LINE"]: self.default_y_offset_multiplier = float(config["LINE"]["default_y_offset_multiplier"])
        
        self.line_level_multiplier = 0.2
        if "line_level_multiplier" in config["CLASSIFICATION"]: self.line_level_multiplier = float(config["CLASSIFICATION"]["line_level_multiplier"])
        
        self.line_boundary_margin = 10
        if "line_boundary_margin" in config["LINE"]: self.line_boundary_margin = int(config["LINE"]["line_boundary_margin"])
        
        self.min_line_heigth = 40
        if "min_line_heigth" in config["LINE"]: self.min_line_heigth = int(config["LINE"]["min_line_heigth"])

        self.min_line_width = 20 
        if "min_line_width" in config["LINE"]: self.min_line_width = int(config["LINE"]["min_line_width"])
    
#region detection   
        self.region_ensemble = False
        if "region_ensemble" in config["REGION DETECTION"]: self.region_ensemble = config["REGION DETECTION"].getboolean("region_ensemble")

        self.ensemble_model_path1 = None
        if "ensemble_model_path1" in config["REGION DETECTION"]: self.ensemble_model_path1 = config["REGION DETECTION"]["ensemble_model_path1"]

        self.ensemble_model_path2 = None
        if "ensemble_model_path2" in config["REGION DETECTION"]: self.ensemble_model_path2 = config["REGION DETECTION"]["ensemble_model_path2"]

        self.ensemble_combine_method = "union"
        if "ensemble_combine_method" in config["REGION DETECTION"]: self.ensemble_combine_method = config["REGION DETECTION"]["ensemble_combine_method"]

        self.region_detection_model = None
        if "region_detection_model" in config["REGION DETECTION"]: self.region_detection_model = config["REGION DETECTION"]["region_detection_model"]


        self.region_model_path = None
        if "region_model_path" in config["REGION DETECTION"]: self.region_model_path = config["REGION DETECTION"]["region_model_path"]

        self.region_config_path = None
        if "region_config_path" in config["REGION DETECTION"]: self.region_config_path = config["REGION DETECTION"]["region_config_path"]

        self.region_num_classes = len(self.region_classLabels)

        self.device = "cuda"
        if "device" in config["REGION DETECTION"]: self.device = config["REGION DETECTION"]["device"]

        self.unet_image_size = 1024
        if "unet_image_size" in config["REGION DETECTION"]: self.unet_image_size = int(config["REGION DETECTION"]["unet_image_size"])

        self.unet_score = 0.2
        if "unet_score" in config["REGION DETECTION"]: self.unet_score = float(config["REGION DETECTION"]["unet_score"])

        self.min_area = 0.0005
        if "min_area" in config["REGION DETECTION"]: self.min_area = float(config["REGION DETECTION"]["min_area"])

        self.yolo_image_size = 640
        if "yolo_image_size" in config["REGION DETECTION"]: self.unet_image_size = int(config["REGION DETECTION"]["yolo_image_size"])

        self.segformer_image_size = 1024
        if "segformer_image_size" in config["REGION DETECTION"]: self.segformer_image_size = int(config["REGION DETECTION"]["segformer_image_size"])


#line detection
        self.line_detection_model = None
        if "line_detection_model" in config["LINE DETECTION"]: self.line_detection_model = config["LINE DETECTION"]["line_detection_model"]

        self.line_config_path = None
        if "line_config_path" in config["LINE DETECTION"]: self.line_config_path = config["LINE DETECTION"]["line_config_path"]

        self.line_model_path = None
        if "line_model_path" in config["LINE DETECTION"]: self.line_model_path = config["LINE DETECTION"]["line_model_path"]

        self.line_num_classes = len(self.line_classLabels)
        
        self.kraken_model_path = None
        if "kraken_model_path" in config["LINE DETECTION"]: self.kraken_model_path = config["LINE DETECTION"]["kraken_model_path"]
# table detection
        self.table_detection_model = None
        if "table_detection_model" in config["TABLE DETECTION"]: self.table_detection_model = config["TABLE DETECTION"]["table_detection_model"]

        self.table_config_path = None
        if "table_config_path" in config["TABLE DETECTION"]: self.table_config_path = config["TABLE DETECTION"]["table_config_path"]

        self.table_column_model_path = None
        if "table_column_model_path" in config["TABLE DETECTION"]: self.table_column_model_path = config["TABLE DETECTION"]["table_column_model_path"]

        self.table_row_model_path = None
        if "table_row_model_path" in config["TABLE DETECTION"]: self.table_row_model_path = config["TABLE DETECTION"]["table_row_model_path"]

        self.table_classes = {}
        if "table_classes" in config["TABLE DETECTION"]: self.table_classes = json.loads(config["TABLE DETECTION"]["table_classes"])

        self.table_classLabels = [x for x in self.table_classes.keys()]

        self.reverse_table_class = {v:k for k,v in self.table_classes.items()}

        self.table_num_classes = len(self.table_classLabels)

        self.table_line = None
        if "table_line" in config["TABLE DETECTION"]: self.table_line = config["TABLE DETECTION"]["table_line"]
        
