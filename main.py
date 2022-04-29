from calculation.calc_functions import (
    process_data_maternal_risks
)
from utils.input_processor import read_csv
from vision.output_processor import output_data


if __name__ == '__main__':
    # data = read_csv('test')
    # results = process_data(data)
    data = read_csv('Maternal Health Risk Data Set')
    results = process_data_maternal_risks(data)

    output_data(results)
