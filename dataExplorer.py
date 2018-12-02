import csv
import sys

csv.field_size_limit(sys.maxsize)

def readData():
    data = []
    with open("stories.csv") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            print(row["Status"])



readData()