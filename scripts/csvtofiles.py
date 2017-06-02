import csv
dataroot = '../four_class_data/'
with open('../dataset/772.csv') as data:
    reader = csv.reader(data, delimiter = ',', quotechar = '"')
    count = 772
    for row in reader:
        dataClass = row[2]
        filePath = dataroot + dataClass + "/" + str(count)
        with open(filePath, 'w') as song:
            song.write(row[1])
        count += 1
