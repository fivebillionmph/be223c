"""
Author: James Go
"""

import csv

class Labels:
    """
    This class saves the labels of multiple patient label csv files
    It's used to save the state on the server for use by the hash similarity function

    Attributes:
        labels: dictionary of labels where patient name maps to 1 or 0 for progression/no progression
    """
    def __init__(self, labels_csv_files):
        self.labels = {}
        for lcf in labels_csv_files:
            with open(lcf) as f:
                reader = csv.reader(f)
                next(reader) # throw away first line
                for line in reader:
                    self.labels[line[0]] = int(line[1])

    def add_labels_to_similarity_list(self, sim_list):
        """
        modifies a list of similar images to add the response label for each image in it

        Args:
            sim_list: a list of similar image names

        Returns:
            the sim_list modified in place with the labels of progression/no progression added
        """
        for sl in sim_list:
            if sl["name"].endswith(".png"):
                name = sl["name"][:-4]
            else:
                name = sl["name"]
            label = self.labels[name]
            sl["response"] = label
        return sim_list
