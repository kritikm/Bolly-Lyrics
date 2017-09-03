import os
import csv

skip_lines = 4
data_root = "../roman/"
output_root = "../roman_cleaned/"

songs = []

for file in os.listdir(data_root):
    with open(data_root + file) as f:
        for _ in range(skip_lines):
            next(f)
        songs.append(f.read())
outname = output_root + "roman.csv"

with open(outname, 'w') as target:
    writer = csv.writer(target, delimiter = ',')
    writer.writerows(zip(songs))
