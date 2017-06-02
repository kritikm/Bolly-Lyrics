import csv
dataroot = '../four_class_data/'
with open('../dataset/258.csv') as data:
    reader = csv.reader(data, delimiter = ',', quotechar = '"')
    count = 258
    for row in reader:
        dataClass = row[2]
        filePath = dataroot + dataClass + "/" + str(count)
        with open(filePath, 'w') as song:
            song.write(row[1])
        count += 1
