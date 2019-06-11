import csv

class Labels:
    def __init__(self, labels_csv_files):
        self.labels = {}
        for lcf in labels_csv_files:
            with open(lcf) as f:
                reader = csv.reader(f)
                next(reader) # throw away first line
                for line in reader:
                    self.labels[line[0]] = int(line[1])

    def add_labels_to_similarity_list(self, sim_list):
        for sl in sim_list:
            if sl["name"].endswith(".png"):
                name = sl["name"][:-4]
            else:
                name = sl["name"]
            label = self.labels[name]
            sl["response"] = label
        return sim_list
