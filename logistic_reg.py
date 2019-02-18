import numpy as np
import csv
import datetime
from pm25_beijing.build_network import L_layer_model, forward_propagation
from pm25_beijing.build_logistic_regression import predict, model
from pm25_beijing.sklearn_lr import use_function

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
X = np.abs(X)
max_X = np.amax(X, axis=1)
for i in range(X.shape[0]):
    X[i, :] = np.divide(X[i, :], max_X[i])
Y = np.array(train_result[:], dtype=float)
# max_Y = np.max(Y)
# Y = np.divide(Y, max_Y)
# Y = Y.reshape((1, Y.shape[0]))


# print(max(Y[0,:]))

# layer_dims = [X.shape[0], 7, 1]
# parameters = L_layer_model(X, Y, layer_dims, learning_rate=0.0075, iteration=2000, print_cost=True)
'''
train_set_x = X
test_set_x = X
test_set_y = Y
train_set_y = Y

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=0.005, print_cost=True)

# AL, caches = forward_propagation(X, parameters)
# print(AL)
'''
test_file = './pm25_test.csv'
test_data = []
with open(test_file) as t:
    t_reader = csv.reader(t)
    for row in t_reader:
        tmp_data = []
        if t_reader.line_num > 1:
            for index in range(len(row)):
                if index == 0:
                    month_day = str(row[index]).split('-')
                    # 转为日期类型
                    date_type = datetime.datetime.strptime("".join(month_day), "%Y%m%d")
                    # 获取在一年中的天数
                    today = date_type.timetuple().tm_yday
                    tmp_data.append(float(today))
                else:
                    tmp_data.append(float(row[index]))
            test_data.append(tmp_data)
    t.close()
test_X = np.array(test_data, dtype=float, copy=True).T
test_X = np.abs(test_X)
max_test_X = np.amax(test_X, axis=1)
for i in range(test_X.shape[0]):
    test_X[i, :] = np.divide(test_X[i, :], max_test_X[i])

# A = predict(d["w"], d["b"], test_X)
# A = np.multiply(A, max_Y)
# A = A.T
result = use_function(X, Y, test_X)
out = open("res.csv", 'w', newline='')
csv_writer = csv.writer(out, dialect='excel')
csv_writer.writerow(["pm2.5"])
result = list(map(lambda x: [x], result))
for i in result:
    csv_writer.writerow(i)
out.close()

'''
test_X = np.divide(test_X, 1000)
assert test_X.shape[0] == X.shape[0]
AL, ZL = forward_propagation(X, parameters)
print(AL)

AL = AL.T


#
# dZ = np.array(layer_dims, copy=True)
# Z = np.array([-1, 2, -4])
# dZ[Z <= 0] = 0
#
# print(dZ)
'''
