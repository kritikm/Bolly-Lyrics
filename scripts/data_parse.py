import os

skip_lines = 5

def cleanData():
    data_root = "/media/sf_IASNLP/dataset/neg-pos/"
    data_dir = ["neg/", "pos/"]
    output_root = "/home/rkb/Documents/cleaned/"

    for dir in data_dir:
        for file in os.listdir(data_root + dir):
            with open(data_root + dir + file) as f:
                    for _ in range(skip_lines):
                        next(f)
                    with open(output_root + dir + file, 'w') as out:
                        for line in f:
                            out.write(line)

cleanData()
