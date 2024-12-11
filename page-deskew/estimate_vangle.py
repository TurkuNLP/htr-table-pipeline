import math
import lstudio2yolov8
import sys
import json
import statistics

from types import SimpleNamespace


if __name__ == "__main__":
    stats={"l": [], "m": [], "r": []}
    
    for inp_file in sys.argv[1:]: #these are the jsons
            with open(inp_file, 'r') as file: #this be one json
                data = json.load(file)
                for item in data:
                    if len(item["annotations"])!=1: #there should be 1 annotation, skip if not
                        continue
                    else:
                        try:
                            kpoints=lstudio2yolov8.extract_kpoints_from_labelstudio_json(item)
                        except ValueError as e:
                            continue
                    stats["l"].append(lstudio2yolov8.vertical_line_angle(kpoints.tl, kpoints.bl))
                    stats["m"].append(lstudio2yolov8.vertical_line_angle(kpoints.tm, kpoints.bm))
                    stats["r"].append(lstudio2yolov8.vertical_line_angle(kpoints.tr, kpoints.br))
    angle_ranges={}
    for k in stats:
        angle_ranges[k]={"mean": statistics.mean(stats[k]), "stdev": statistics.stdev(stats[k]), "min": min(stats[k]), "max": max(stats[k])}
    print(angle_ranges)