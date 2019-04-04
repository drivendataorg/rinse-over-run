import os
import time
from logger import Logger


def write_results_in_my_format(all_names, all_outputs, results_file_name):
    logger = Logger(__name__).logger
    logger.info('Starting to write output results in my format...')
    start = time.time()
    rows = []
    with open(results_file_name, 'w') as csv_file:
        csv_file.write('process_id,final_rinse_total_turbidity_liter\n')

        for (name, ouput) in zip(all_names, all_outputs):
            result = ouput.item() * 1000000.0
            if result < 0.0:
                print('process_id ', name.item(), end=' ')
                result = 0.0
            row = str(name.item()) + ',' + str(result) + '\n'
            rows.append(row)
        rows[-1] = rows[-1].replace('\n', '')
        csv_file.writelines(rows)
    end = time.time()
    logger.info('Results are written in my format in {}'.format(end - start))


def remove_directory(top):
    for root, dirs, files in os.walk(top, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
