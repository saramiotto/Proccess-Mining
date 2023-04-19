from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np

class ImagePPMiner:
    def __init__(self, eventlog):
        """
        Initialize the ImagePPMiner class with the eventlog file name.
        Parameters:
        eventlog (str): name of the eventlog file.
        """
        self._eventlog = eventlog

    def import_log(self):
        """
        This method imports the XES log using the xes importer and converts the log to a pandas dataframe.
        Then it maps the names of activities to unique integers, select the columns "case:concept:name", "concept:name", "time:timestamp" from the dataframe and returns the modified dataframe.
        """
        # Import the XES log using the xes importer
        log = xes_importer.apply('ImagePPMiner-master/dataset/'+self._eventlog+'.xes')
        # Convert the log to a pandas dataframe
        dataframe = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)

        # Get the unique names of activities in the log
        unique = dataframe['concept:name'].unique()
        # Create a dictionary mapping the names of activities to unique integers
        dictOfWords = { i : unique[i] for i in range(0, len(unique) ) }
        dictOfWords = {v: k for k, v in dictOfWords.items()}
        for k in dictOfWords:
            dictOfWords[k] += 1
        # Replace the original names of activities with the unique integers
        dataframe['concept:name'] = [dictOfWords[item] for item in dataframe['concept:name']]
        # Select the columns "case:concept:name", "concept:name", "time:timestamp" from the dataframe
        dataframe = dataframe[["case:concept:name", "concept:name", "time:timestamp"]]
        # Return the modified dataframe
        return dataframe

    def generate_prefix_trace(self, log, n_caseid):
        
        # grouping the log dataframe by case:concept:name and taking the concept:name column
        # and applying the list function to it, to convert the values into a list
        act = log.groupby('case:concept:name', sort=False).agg({'concept:name': lambda x: list(x)})
        
        # grouping the log dataframe by case:concept:name and taking the time:timestamp column
        # and applying the list function to it, to convert the values into a list
        temp = log.groupby('case:concept:name', sort=False).agg({'time:timestamp': lambda x: list(x)})
        
        # calculating the size of the train set, using n_caseid as 2/3 of the total dataframe size
        size = int((n_caseid / 3) * 2)
        
        # splitting the act dataframe into train and test sets
        train_act = act[:size]
        
        # splitting the temp dataframe into train and test sets
        train_temp = temp[:size]
        
        # splitting the act dataframe into train and test sets
        test_act = act[size:]
        
        # splitting the temp dataframe into train and test sets
        test_temp = temp[size:]
        
        # returning the train_act, train_temp, test_act, test_temp variables
        return train_act, train_temp, test_act, test_temp

    @staticmethod
    def generate_image(act_val, time_val, max_trace, n_activity):
        
        # Initiate loop variables and image matrix
        i = 0
        matrix_zero = [max_trace, n_activity, 2]
        image = np.zeros(matrix_zero)
        list_image = []
        
        # Iterate over the length of time_val
        while i < len(time_val):
            j = 0
            list_act = []
            list_temp = []
            
            # Create a list of integers from 1 to n_activity
            a = list(range(1, n_activity + 1))
            dict_act = dict.fromkeys(a, 0)
            dict_time = dict.fromkeys(a, 0)
            
            # Iterate over the length of activity value
            while j < (len(act_val.iat[i, 0]) - 1):
                
                # Get start time of the trace
                start_trace = time_val.iat[i, 0][0]
                
                # Increase the value of the activity in the dictionary
                dict_act[act_val.iat[i, 0][0 + j]] += 1
                
                # Calculate duration of the activity
                duration = time_val.iat[i, 0][0 + j] - start_trace
                
                # calculate days
                days = (duration.total_seconds())/86400
                dict_time[act_val.iat[i, 0][0 + j]] = days
                l_act = list(dict_act.values())
                l_time = list(dict_time.values())
                list_act.append(l_act)
                list_temp.append(l_time)
                j = j + 1
                cont = 0
                lenk = len(list_act) - 1
                
                # Create image
                while cont <= lenk:
                    z = 0
                    while z < n_activity:
                        image[(max_trace - 1) - cont][z] = [list_act[lenk - cont][z], list_temp[lenk - cont][z]]
                        z = z + 1
                    cont = cont + 1
                if cont > 1:
                    list_image.append(image)
                    image = np.zeros(matrix_zero)
            i = i + 1
        return list_image



    @staticmethod
    def get_label(act):
        i = 0 
        
        # initialize an empty list to store labels
        list_label = []
        
        # iterate through all activities 
        while i < len(act):
            j = 0
            
            # iterate through each activity
            while j < (len(act.iat[i, 0]) - 1):
               
                # if j>0, append the next label
                if j > 0:
                    list_label.append(act.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        
        # return the generated label list
        return list_label


    @staticmethod
    def dataset_summary(log):
        
        # Print the distribution of activities
        print("Activity Distribution\n", log['concept:name'].value_counts())
        
        # Calculate the unique numbers for cases
        n_caseid = log['case:concept:name'].nunique()
        print("Number of CaseID", n_caseid)
        
        # Calculate the unique number for activities
        n_activity = log['concept:name'].nunique()
        print("Number of Unique Activities", n_activity)
        
        # Calculate the total number of activities
        print("Number of Activities", log['concept:name'].count())
        
        # Calculate the total number of events
        cont_trace = log['case:concept:name'].value_counts(dropna=False)
        
        # Calculate the maximum length of the trace
        max_trace = max(cont_trace)
        print("Max lenght trace", max_trace)
        
        # Calculate the mean length of the trace
        print("Mean lenght trace", np.mean(cont_trace))
        
        # Calculate the minimum length of the trace
        print("Min lenght trace", min(cont_trace))
        return max_trace, n_caseid, n_activity