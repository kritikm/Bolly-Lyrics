import csv
dataroot = '../four_class_roman/'
with open('../roman_cleaned/roman.csv') as data:
    reader = csv.reader(data, delimiter = ',', quotechar = '"')
    count = 1
    for row in reader:
        if row[1] == "":
            count += 1
            continue

        dataClass = row[1]
        filePath = dataroot + dataClass + "/" + str(count)
        with open(filePath, 'w') as song:
            song.write(row[0])
        count += 1
