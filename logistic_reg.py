import numpy as np
import csv
import datetime
from pm25_beijing.build_network import L_layer_model, forward_propagation

# X =

train_data = list(list())
train_result = list()

fileName = "./pm25_train.csv"

with open(fileName) as f:
    reader = csv.reader(f)
    for row in reader:
        if reader.line_num > 1:
            tmp_arr = []
            for index in range(len(row)):
                if index == 0:
                    month_day = str(row[index]).split('-')
                    # 转为日期类型
                    date_type = datetime.datetime.strptime("".join(month_day), "%Y%m%d")
                    # 获取在一年中的天数
                    today = date_type.timetuple().tm_yday
                    tmp_arr.append(float(today))
                elif index != 2:
                    tmp_arr.append(float(row[index]))
            train_data.append(tmp_arr)
            train_result.append(row[2])
    f.close()
X = np.array(train_data, dtype=float, copy=True).T
Y = np.array(train_result[:], dtype=float)/1000.0
Y = Y.reshape((1, Y.shape[0]))
# print(max(Y[0,:]))

layer_dims = [X.shape[0], 7, 1]
parameters = L_layer_model(X, Y, layer_dims, learning_rate=0.0075, iteration=1000, print_cost=True)


AL, caches = forward_propagation(X, parameters)
print(AL)

out = open("res.csv", 'a', newline='')
csv_writer = csv.writer(out, dialect='excel')
AL = np.multiply(AL, 1000)
AL = AL.T
for i in range(len(AL)):
    csv_writer.writerow(AL[i])
out.close()

#
# dZ = np.array(layer_dims, copy=True)
# Z = np.array([-1, 2, -4])
# dZ[Z <= 0] = 0
#
# print(dZ)
