import os
import json
import numpy as np
import pandas as pd

class LabelStudioTools:
    def __init__(self, dataset_location = None, make_folders = False):
        if make_folders:
            # If folder doesn't exist, create it
            if not os.path.exists(dataset_location): os.makedirs(dataset_location)
            self.dataset_location = dataset_location

            # Create the data and predictions folders
            if not os.path.exists(dataset_location + "/data"): os.makedirs(dataset_location + "/data")
            self.data_location = dataset_location + "/data"

            if not os.path.exists(dataset_location + "/predictions"): os.makedirs(dataset_location + "/predictions")
            self.prediction_location = dataset_location + "/predictions"


    def save_data_as_csv(self, data, location, timestamp, session_id):
        # Lets create a dataframe with the data, first column will be "time" starting with 0, and incrementing by 1
        # Second column will be lead_ecg2 and third column will be lead_ecg3
        # data is a numpy array with 2 rows, first row is lead_ecg2 and second row is lead_ecg3
        # data.shape[1] is the number of columns in the data array
        # data.shape[0] is the number of rows in the data array
        
        df = pd.DataFrame(columns=["time", "lead_ecg2", "lead_ecg3"])
        df["time"] = np.arange(0, data.shape[1])
        df["lead_ecg2"] = data[0,:]
        df["lead_ecg3"] = data[1,:]
        
        # Save the dataframe to a csv file
        df.to_csv(self.data_location + f"/{location}_fgre_{timestamp}_{session_id}.csv", index=False)


    def save_predictions_as_json(self, s3_data_folder, location, timestamp, session_id, predictions,detections = None, vcg_trigger_detections = None):
        # Create JSON file with predictions in format for Label Studio
        # predictions is a numpy array with indices of the peaks
        # in the prediction json file we should give a range of peak -2 to peak +2 for start and end time
        peak_offset = 10 

        # Construct the json object.
        # The json object should be a list of dictionaries
        
        pred = {
            "data" : {
                "location": location,
                "timestamp": timestamp,
                "session_id": session_id,
                "csv": "s3://ucsd-ecg-dataset/" + s3_data_folder + "/" + location + "_fgre_" + timestamp + "_" + session_id + ".csv"
            },
            "predictions": [
                ]
        }

        if(predictions is not None):
            pred_entry = {
                    "model_version": "CNN_Detector",
                    "result": []
            }
            for p in predictions:
                pred_entry["result"].append(
                    {
                        "from_name": "label",
                        "to_name": "ts",
                        "type": "timeserieslabels",
                        "value": {
                            "start": p,
                            "end": p,
                            "instant": True,
                            "timeserieslabels": [
                                "R-Peak"
                            ]
                        }
                    }
                )
            pred["predictions"].append(pred_entry)

        if(detections is not None):
            pred_entry = {
                    "model_version": "CNN_Detector",
                    "result": []
            }
            for p in detections:
                pred_entry["result"].append(
                    {
                        "from_name": "label",
                        "to_name": "ts",
                        "type": "timeserieslabels",
                        "value": {
                            "start": p,
                            "end": p,
                            "instant": True,
                            "timeserieslabels": [
                                "R-Peak"
                            ]
                        }
                    }
                )
            pred["predictions"].append(pred_entry)

        if(vcg_trigger_detections is not None):
            pred_entry = {
                    "model_version": "CNN_Detector",
                    "result": []
            }
            for p in vcg_trigger_detections:
                pred_entry["result"].append(
                    {
                        "from_name": "label",
                        "to_name": "ts",
                        "type": "timeserieslabels",
                        "value": {
                            "start": p,
                            "end": p,
                            "instant": True,
                            "timeserieslabels": [
                                "R-Peak"
                            ]
                        }
                    }
                )
                
        # Save the json object to a json file
        with open(self.prediction_location + f"/{location}_fgre_{timestamp}_{session_id}.json", "w") as f: json.dump(pred, f)


    def convert_short_json_annotations_to_dataframe(self, json_file):
        # Load the json file
        with open(json_file) as f: data = json.load(f)

        # Create the dataframe
        df = pd.DataFrame(columns=["location", "timestamp", "session_id", "annotations", "task_id"])

        # Iterate through the results and add the annotations to the dataframe
        for task in data:
            # Get the location, timestamp and session_id from the data
            # if task has patientid key, then it is in the old format and needs to be changed
            if("patientid" in task):
                location = task["location"]
                timestamp = task["patientid"] + "_" + task["timestamp"].split("_")[0] + "_" + task["timestamp"].split("_")[1]
                session_id = task["timestamp"].split("_")[2]
            else:
                location = task["location"]
                timestamp = task["timestamp"]
                session_id = task["session_id"]

            # Annotations to a Pandas Series
            annotations_series = self.LabelToSeries(task["label"])

            is_mwi_filter = False
            is_paced_signal = False
            
            if "ecg2_filter" in task:
                # If ecg2_filter has choices key then it is in the new format
                if("choices" in task["ecg2_filter"]):
                    # If ecg2_filter is a list
                    for filter in task["ecg2_filter"]["choices"]:
                        if(filter == "MWI Filter"): is_mwi_filter = True
                        if(filter == "Paced"): is_paced_signal = True
                else:
                    is_mwi_filter = (task["ecg2_filter"] == "MWI Filter")
                    is_paced_signal = (task["ecg2_filter"] == "Paced")

            # Add the annotations to the dataframe
            new_row = {"location": location, "timestamp": timestamp, "session_id": session_id, "is_mwi_filter": is_mwi_filter,"is_paced": is_paced_signal, "annotations": annotations_series, "task_id": task["id"]}
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

        return df
    

    def convert_json_annotations_to_dataframe(self, json_file):
        # Load the json file
        with open(json_file) as f: data = json.load(f)

        # Create the dataframe
        df = pd.DataFrame(columns=["location", "timestamp", "session_id", "annotations"])

        # Iterate through the results and add the annotations to the dataframe
        for task in data:
            if 'label' not in task:
                print(f"Key 'label' not found in task: {task}") 
                continue  # Skip this task if 'label' key is missing

            # if task has patientid key, then it is in the old format and needs to be changed
            if("patientid" in task):
                location = task["location"]
                timestamp = task["patientid"] + "_" + task["timestamp"].split("_")[0] + "_" + task["timestamp"].split("_")[1]
                session_id = task["timestamp"].split("_")[2]
            else:
                location = task["location"]
                timestamp = task["timestamp"]
                session_id = task["session_id"]

            # # If the ecg2_quality is unusable, then skip the task
            # if("ecg2_quality" in task and task["ecg2_quality"] == "Unusable"): continue
            
            # Annotations to a Pandas Series
            annotations_series = self.LabelToSeries(task["label"])

            is_mwi_filter = False
            is_paced_signal = False

            if "ecg2_filter" in task:
                # If ecg2_filter has choices key then it is in the new format
                if("choices" in task["ecg2_filter"]):
                    # If ecg2_filter is a list
                    for filter in task["ecg2_filter"]["choices"]:
                        if(filter == "MWI Filter"): is_mwi_filter = True
                        if(filter == "Paced"): is_paced_signal = True
                else:
                    is_mwi_filter = (task["ecg2_filter"] == "MWI Filter")
                    is_paced_signal = (task["ecg2_filter"] == "Paced")
        
            # Add the annotations to the dataframe
            new_row = {"location": location, "timestamp": timestamp, "session_id": session_id, "is_mwi_filter": is_mwi_filter, "is_paced": is_paced_signal,  "annotations": annotations_series, "task_id": task["id"]}
            new_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_df], ignore_index=True)

        return df
    

    def isPacedSignal(self, annotations):
        # Check if the task has annotations that have results that contain "value" that contain "choices" that contains "MWI Filter"
        if("result" in annotations[0]):
            # For each result in the annotations
            for result in annotations[0]["result"]:
                if("choices" in result["value"] and result["value"]["choices"][0] == "Paced"): return True
        return False
    

    def isMWIFilter(self, annotations):
        # Check if the task has annotations that have results that contain "value" that contain "choices" that contains "MWI Filter"
        if("result" in annotations[0]):
            # For each result in the annotations
            for result in annotations[0]["result"]:
                if("choices" in result["value"] and result["value"]["choices"][0] == "MWI Filter"): return True
        return False
    

    def LabelToSeries(self, labels):
        # Convert the annotations to a Pandas Series
        marked_peaks = []
        for label in labels:
            # If annotation_value doesn't contain "start" key, then it is not a valid annotation
            if("start" not in label): continue
            # Get the annotation start time
            annotation_start = label["start"]
            # Get the annotation end time
            if(label["instant"] == False): annotation_end = label["end"]
            # If the annotation is an R-Peak or PVC-Peak, then add it to the list
            if("R-Peak" in label["timeserieslabels"] or "PVC-Peak" in label["timeserieslabels"]): marked_peaks.append(annotation_start)
        return pd.Series(marked_peaks, copy=True)
    

    def AnnotationsToSeries(self, annotations):
        # Convert the annotations to a Pandas Series
        marked_peaks = []
        for annotation in annotations[0]["result"]:
            # Get the annotation value
            annotation_value = annotation["value"]
            # If annotation_value doesn't contain "start" key, then it is not a valid annotation
            if("start" not in annotation_value): continue
            # Get the annotation start time
            annotation_start = annotation_value["start"]
            # Get the annotation end time
            if(annotation_value["instant"] == False): annotation_end = annotation_value["end"]
            # If the annotation is an R-Peak or PVC-Peak, then add it to the list
            if("R-Peak" in annotation_value["timeserieslabels"] or "PVC-Peak" in annotation_value["timeserieslabels"]): marked_peaks.append(annotation_start)
        return pd.Series(marked_peaks, copy=True)

