from csv import reader


class PoseReader:

    def __init__(self, filepath) -> None:
        self.path = filepath

    def read(self):
        import csv
        datarows = []
        with open(self.path, 'w', newline='') as f:
            reader = csv.reader()
            datarows = [row for row in reader]
        data = datarows[3:]
