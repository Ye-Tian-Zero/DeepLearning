from HandWritenRecognize import LeNet
import numpy as np
import csv
le_net = LeNet((20, 50))
csv_file = file('test.csv', 'r')
reader = csv.reader(csv_file)
test_set = np.array([line for line in reader])
test_set = test_set[1:, :]
test_set = test_set.astype('float32') / 255
result = [le_net.predict(ret.reshape(1,784)) for ret in test_set]
result = [str(i + 1) + ',' + str(s[0]) for i, s in zip(range(len(result)),result)]
f = open('kaggle_result.csv', 'w')
f.write('ImageId,Label\n')
f.write('\n'.join(result))



