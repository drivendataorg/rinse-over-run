import pandas as pd
import numpy as np

data = pd.read_csv('submission_test.csv')
data = np.array(data[['process_id', 'final_rinse_total_turbidity_liter']])

process_id = int(data[0, 0]) % 100000
mean_value = data[0, 1]
counter = 1.0
ids = []
values = []
for i in range(1, data.shape[0]):
    data[i, 0] = int(data[i, 0]) % 100000
    if data[i, 0] == process_id:
        mean_value = mean_value + data[i, 1]
        counter = counter + 1.0
    else:
        ids.append(process_id)
        values.append(mean_value / counter)

        process_id = int(data[i, 0]) % 100000
        mean_value = data[i, 1]
        counter = 1.0

ids.append(process_id)
values.append(mean_value / counter)

rows = []
with open('submission_test_tta.csv', 'w') as csv_file:
    csv_file.write('process_id,final_rinse_total_turbidity_liter\n')

    for (name, output) in zip(ids, values):
        row = str(int(name)) + ',' + str(output) + '\n'
        rows.append(row)
    rows[-1] = rows[-1].replace('\n', '')
    csv_file.writelines(rows)
