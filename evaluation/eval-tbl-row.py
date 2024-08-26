import argparse
import torch
from torchvision import ops
import xml.etree.ElementTree as ET
import numpy as np

def parse_xml(xml_file): # Parses the xml file and returns a list of table bounding boxes and a list of row bounding boxes.

    namespaces = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}
    tree = ET.parse(xml_file)
    root = tree.getroot()

    table_bboxes = []
    row_bboxes = []
    column_bboxes = []

    for region in root.findall('.//ns:TableRegion', namespaces):
        # Parse table bounding boxes
        coords = region.find('ns:Coords', namespaces).get('points')
        points = coords.split()
        x_coords = [float(point.split(',')[0]) for point in points]
        y_coords = [float(point.split(',')[1]) for point in points]
        xmin, ymin, xmax, ymax = min(x_coords), min(y_coords), max(x_coords), max(y_coords)
        table_bboxes.append([xmin, ymin, xmax, ymax])
        
        # Parse row bounding boxes
        rows = {}
        columns = {}
        cells = region.findall('.//ns:TableCell', namespaces)
        
        if len(cells) > 1:
            for cell in cells:
                coords = cell.find('ns:Coords', namespaces).get('points')
                points = coords.split()
                x_coords = [float(point.split(',')[0]) for point in points]
                y_coords = [float(point.split(',')[1]) for point in points]
                xmin = min(x_coords)
                ymin = min(y_coords)
                xmax = max(x_coords)
                ymax = max(y_coords)

                row_index = int(cell.get('row'))

                if row_index not in rows:
                    rows[row_index] = [xmin, ymin, xmax, ymax]
                else:
                    # Update the row bounding box to include this cell
                    rows[row_index][0] = min(rows[row_index][0], xmin)
                    rows[row_index][1] = min(rows[row_index][1], ymin)
                    rows[row_index][2] = max(rows[row_index][2], xmax)
                    rows[row_index][3] = max(rows[row_index][3], ymax)
                
                column_index = int(cell.get('col'))

                if column_index not in columns:
                    columns[column_index] = [xmin, ymin, xmax, ymax]
                else:
                    # Update the columns bounding box to include this cell
                    columns[column_index][0] = min(columns[column_index][0], xmin)
                    columns[column_index][1] = min(columns[column_index][1], ymin)
                    columns[column_index][2] = max(columns[column_index][2], xmax)
                    columns[column_index][3] = max(columns[column_index][3], ymax)
                
            
            for bbox in rows.values():
                row_bboxes.append(bbox)
            
            for bbox in columns.values():
                column_bboxes.append(bbox)


    return torch.tensor(table_bboxes), torch.tensor(row_bboxes), torch.tensor(column_bboxes)

def eval_iou(bboxes1, bboxes2): 

    matched_pairs = []

    num_bboxes1 = len(bboxes1)
    num_bboxes2 = len(bboxes2)
    
    box_similarities = np.zeros((num_bboxes1, num_bboxes2))
    
    for i in range(num_bboxes1):
        for j in range(num_bboxes2):
            iou = ops.box_iou(bboxes1[i].unsqueeze(0), bboxes2[j].unsqueeze(0)).item()
            box_similarities[i, j] = iou
    
    # print("IoU Matrix:")
    # print(box_similarities)


    while np.max(box_similarities) > 0.0:
        max_value = np.max(box_similarities)
        ind = np.unravel_index(np.argmax(box_similarities), box_similarities.shape)
        
        matched_pairs.append((ind[0] + 1, ind[1] + 1, max_value))
        
        box_similarities[ind[0], :] = 0.0
        box_similarities[:, ind[1]] = 0.0

   
    for i in range(len(bboxes1)):
        if not any(pair[0] == i + 1 for pair in matched_pairs):
            matched_pairs.append((i + 1, None, 0.0))
            #print(f"  No match for bbox1[{i + 1}], added None")

    return sorted(matched_pairs)

    

def calculate_precision_and_recall(matched_pairs, total_gt, total_pred, threshold):
    
    true_positives = sum(1 for pair in matched_pairs if pair[1] is not None and pair[2] >= threshold)
    false_positives = total_pred - true_positives
    false_negatives = total_gt - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    return precision, recall

def eval_table_borders(ground_truth_tables, prediction_tables, threshold): # IoU for table borders + precision and recall

    print("Evaluating Table Borders IoU:")
    results = eval_iou(ground_truth_tables, prediction_tables)

    unmatched_gt_tables = set(range(1, len(ground_truth_tables) + 1))
    unmatched_pred_tables = set(range(1, len(prediction_tables) + 1))

    iou_sum = 0
    for result in results:
        if result[1] is not None and result[2] >= threshold:
            print(f'Best IoU for ground truth table bbox {result[0]} is with predicted bbox {result[1]}: {result[2]:.4f}')
            iou_sum += result[2]
            unmatched_gt_tables.discard(result[0])
            unmatched_pred_tables.discard(result[1])

        elif result[1] is None or result[2] < threshold:
            print(f'Ground truth table bbox {result[0]} has no matching predicted bbox.')
    
    if unmatched_gt_tables:
        print(f'{len(unmatched_gt_tables)} ground truth rows have no matching predicted rows: {unmatched_gt_tables}.')
    if unmatched_pred_tables:
        print(f'{len(unmatched_pred_tables)} predicted rows have no matching ground truth rows: {unmatched_pred_tables}.')

    precision, recall = calculate_precision_and_recall(results, len(ground_truth_tables), len(prediction_tables), threshold)
    average_iou = iou_sum / len(results)
    print(f"Table Borders Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Table Borders Average IoU: {average_iou:.4f}")

def eval_table_rows(ground_truth_rows, prediction_rows, threshold): # IoU for row borders + precision and recall
  
    print("Evaluating Table Rows IoU:")
    results = eval_iou(ground_truth_rows, prediction_rows)

    unmatched_gt_rows = set(range(1, len(ground_truth_rows) + 1))
    unmatched_pred_rows = set(range(1, len(prediction_rows) + 1))

    iou_sum = 0
    for result in results:
        if result[1] is not None and result[2] >= threshold:
            print(f'Best IoU for ground truth row {result[0]} is with predicted row {result[1]}: {result[2]:.4f}')
            iou_sum += result[2]
            unmatched_gt_rows.discard(result[0])
            unmatched_pred_rows.discard(result[1])

        elif result[1] is None or result[2] < threshold:
            print(f'Ground truth row {result[0]} has no matching predicted row.')

    if unmatched_gt_rows:
        print(f'{len(unmatched_gt_rows)} ground truth rows have no matching predicted rows: {unmatched_gt_rows}.')
    if unmatched_pred_rows:
        print(f'{len(unmatched_pred_rows)} predicted rows have no matching ground truth rows: {unmatched_pred_rows}.')


    precision, recall = calculate_precision_and_recall(results, len(ground_truth_rows), len(prediction_rows), threshold)
    average_iou = iou_sum / len(results)
    print(f"Table Rows Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Table Rows Average IoU: {average_iou:.4f}")


def eval_table_columns(ground_truth_columns, prediction_columns, threshold): # IoU for column borders + precision and recall
  
    print("Evaluating Table Columns IoU:")
    results = eval_iou(ground_truth_columns, prediction_columns)

    unmatched_gt_columns = set(range(1, len(ground_truth_columns), + 1))
    unmatched_pred_columns = set(range(1, len(prediction_columns) + 1))

    iou_sum = 0
    for result in results:
        if result[1] is not None and result[2] >= threshold:
            print(f'Best IoU for ground truth column {result[0]} is with predicted column {result[1]}: {result[2]:.4f}')
            iou_sum += result[2]
            unmatched_gt_columns.discard(result[0])
            unmatched_pred_columns.discard(result[1])
            
        elif result[1] is None or result[2] < threshold:
            print(f'Ground truth column {result[0]} has no matching predicted column.')
    
    if unmatched_gt_columns:
        print(f'{len(unmatched_gt_columns)} ground truth rows have no matching predicted rows: {unmatched_gt_columns}.')
    if unmatched_pred_columns:
        print(f'{len(unmatched_pred_columns)} predicted rows have no matching ground truth rows: {unmatched_pred_columns}.')
    
    precision, recall = calculate_precision_and_recall(results, len(ground_truth_columns), len(prediction_columns), threshold)
    average_iou = iou_sum / len(results)
    print(f"Table Columns Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Table Columns Average IoU: {average_iou:.4f}")


def main(args):
    ground_truth_tables, ground_truth_rows, ground_truth_columns = parse_xml(args.groundtruth_bbox)
    prediction_tables, prediction_rows, prediction_columns = parse_xml(args.prediction_bbox)


    eval_table_borders(ground_truth_tables, prediction_tables, args.iou_threshold)
    eval_table_rows(ground_truth_rows, prediction_rows, args.iou_threshold)
    eval_table_columns(ground_truth_columns, prediction_columns, args.iou_threshold)

    num_gt_tables = len(ground_truth_tables)
    num_pred_tables = len(prediction_tables)
    num_gt_rows = len(ground_truth_rows)
    num_pred_rows = len(prediction_rows)
    num_gt_columns = len(ground_truth_columns)
    num_pred_columns = len(prediction_columns)

    
    if num_gt_tables != num_pred_tables:
        print(f'\033[1mWarning:\033[0m Number of ground truth tables ({num_gt_tables}) does not match number of predicted tables ({num_pred_tables}).')
    if num_gt_rows != num_pred_rows:
        print(f'\033[1mWarning:\033[0m Number of ground truth rows ({num_gt_rows}) does not match number of predicted rows ({num_pred_rows}).')
    if num_gt_columns != num_pred_columns:
        print(f'\033[1mWarning:\033[0m Number of ground truth columns ({num_gt_columns}) does not match number of predicted columns ({num_pred_columns}).')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare table and row bounding boxes with IoU threshold')
    parser.add_argument('--groundtruth-bbox', type=str, help='Annotated XML file', required=True)
    parser.add_argument('--prediction-bbox', type=str, help='Predicted XML file', required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.0, help='IoU threshold for matching (default is 0.0)')

    args = parser.parse_args()

    main(args)

     # run: python eval-tbl-row.py --groundtruth-bbox annotated xml-file --prediction-bbox predicted xml-file --iou-threshold 0.6 (optional default is 0.0)

"""
To do:

Design Choice - ratkaisuja

- Rivit prosessoidaan taulukoista irrallisina, jolloin rivievaluoinnissa ei ole väliä kuinka monta taulukkoa on annotoitu/ennustettu. 
- Rivi rakennetaan nyt ottamalla solujen (min x, max x) ja (min y, max y), jolloin esim. paljon vinossa olevasta kapeasta rivistä muodostuu hyvin paksu laatikko. 

"""