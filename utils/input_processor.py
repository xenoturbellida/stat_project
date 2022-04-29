import csv
import re


def beautify_title(title: str) -> str:
    return "".join(re.findall("[a-zA-Z]", title))


def read_csv(file_name: str):

    with open(f'assets/{file_name}.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = []
        for row in reader:
            data.append(row)

        data_located = {}
        for coll_no in range(len(data[0])):
            coll_name = ''
            for row_no in range(len(data)):
                if row_no == 0:
                    coll_name = beautify_title(data[row_no][coll_no])
                    data_located[coll_name] = []
                else:
                    value = data[row_no][coll_no]
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    data_located[coll_name].append(value)

        return data_located
