import os
import csv
import math
import numpy as np

class MotionData:
    def __init__(self):
        # Set up the data folder
        cwd = os.path.dirname(__file__)
        self._data_path = os.path.join(cwd, "data_GP")
        self._data_name_start = "block"
        self._data_name_end   = "-UNWEIGHTED-SLOW-NONDOMINANT-RANDOM"

        # Names of the folders for each of the participants in the motion study
        self._participants = [name for name in os.listdir(self._data_path) if os.path.isdir(os.path.join(self._data_path, name))]
        self._participant_data = [None]*len(self._participants)
        self.participant_count = len(self._participants)

    def participant(self, idx: int, run: int = None):
        # Make sure index is valid
        if idx >= self.participant_count:
            print("Invalid index")
            return None

        # If we haven't read this folder yet, load the data
        if self._participant_data[idx] is None:
            print(f"Extracting data for participant {idx}")
            self._participant_data[idx] = [None]*5
            for run_instance in range(5):
                # Get the full path to the data file
                folder_name = self._data_name_start + str(run_instance+1) + self._data_name_end
                folder_path = os.path.join(self._data_path, self._participants[idx], folder_name)
                file_name = [file for file in os.listdir(folder_path) if file.endswith(".csv")][0]
                full_path = os.path.join(folder_path, file_name)

                # Open the file and copy it's contents
                with open(full_path) as data_file:
                    data = csv.DictReader(data_file)
                    self._participant_data[idx][run_instance] = list(data)
                    data_file.close()

        if run is None:
            return self._participant_data[idx]
        elif run < 0 or run > 4:
            print(f"Run must be between 0 and 5; {run} was given instead")
            return None
        else:
            return self._participant_data[idx][run]

    """
    Return the data from a number of runs from the GP data set

    dir   : The axis on which to load the data. Can be 'x', 'y', or 'z'
    count : The number of runs to take. Will roll over to the next participant
            if needed 
    start : The index of the first run to record
    """
    def fingerData(self, dir: str, count: int, start: int = 0):
        data = np.zeros((0, 2000))
        min_len = np.inf

        for i in range(start, start + count, 1):
            new_data_set = self.participant(i//5, i - math.floor(i/5)*5)
            min_len = min(min_len, len(new_data_set))
            new_row = np.zeros([1, 2000])

            for frame, frame_dict in enumerate(new_data_set):
                new_row[0,frame] = frame_dict['finger_' + dir]

            data = np.vstack([data, new_row])

        return data[:,:min_len]
