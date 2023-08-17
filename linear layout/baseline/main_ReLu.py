import os.path
import csv
import numpy as np
from dnn_ReLu import Dnn

num_epoch = 500

lr = np.array([3e-6])
# lr = np.array([1e-5])
# lr = np.array([3e-5])
# lr = np.array([1e-4])
# lr = np.array([3e-4])

N0 = 1e-9
pl_exponent = 3.5

num_cell = 6
num_user_per_cell = 10

W = 1e6

edge_frf = 3
center_frf = 3

num_edge = int(num_user_per_cell/2)
num_center = int(num_user_per_cell/2)

bs_range = 0.5
shadowing_var = -1

# Cthr = 4e5
Cthr = 5e5
# Cthr = 6e5

num_sample_train = 8000
train_user_seed = 15

if shadowing_var != -1:
    if bs_range == 0.5:
        filename_h = '../data/irregular_' + str(num_sample_train) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(train_user_seed) + '_a' + str(pl_exponent) + '_' + str(shadowing_var) +'dB.dat'
    else:
        filename_h = '../data/irregular_' + str(num_sample_train) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(train_user_seed) + '_a' + str(pl_exponent) + '_' + str(shadowing_var) + 'dB_bs' + str(bs_range) +'.dat'

else:
    if bs_range == 0.5:
        filename_h = '../data/irregular_' + str(num_sample_train) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(train_user_seed) + '_a' + str(pl_exponent) +'.dat'
    else:
        filename_h = '../data/irregular_' + str(num_sample_train) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(train_user_seed) + '_a' + str(pl_exponent) + '_bs' + str(bs_range) +'.dat'

with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')

input_h_train = np.reshape(input_h_temp, [num_sample_train,-1])
print('input_h_train', input_h_train)

num_sample_test = 2000
test_user_seed = 25

if shadowing_var != -1:
    if bs_range == 0.5:
        filename_h = '../data/irregular_' + str(num_sample_test) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(test_user_seed) + '_a' + str(pl_exponent) + '_' + str(shadowing_var) +'dB.dat'
    else:
        filename_h = '../data/irregular_' + str(num_sample_test) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(test_user_seed) + '_a' + str(pl_exponent) + '_' + str(shadowing_var) + 'dB_bs' + str(bs_range) +'.dat'

else:
    if bs_range == 0.5:
        filename_h = '../data/irregular_' + str(num_sample_test) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(test_user_seed) + '_a' + str(pl_exponent) +'.dat'
    else:
        filename_h = '../data/irregular_' + str(num_sample_test) + 'chs_' + str(num_cell) + 'cell_' + str(num_user_per_cell) + 'users_seed' + str(test_user_seed) + '_a' + str(pl_exponent) + '_bs' + str(bs_range) +'.dat'


with open(filename_h, 'r') as f:
    rdr = csv.reader(f, delimiter='\t')
    temp = np.array(list(rdr))
    input_h_temp = np.delete(temp, -1, axis=1).astype('float64')

input_h_test = np.reshape(input_h_temp, [num_sample_test,-1])
print('input_h_test', input_h_test)

batch_size = 1
Layer_dim_list = [128, num_cell]

all_beta = 1
# all_beta = 2
# all_beta = 4

alpha = 50000

dnn0bj = Dnn(batch_size=batch_size, n_epoch=num_epoch, layer_dim_list=Layer_dim_list, num_cell=num_cell,num_user_per_cell=num_user_per_cell,
             N0=N0, W=W, all_beta=all_beta, Cthr =Cthr, alpha = alpha)


file_path = '.\\dnn_result\\train_sample='+ str(num_sample_train) + ',test_sample=' + str(num_sample_test) + '//' + str(num_cell) +'cell,' + str(num_user_per_cell) +'users(seed='+ str(train_user_seed)  + ','+ str(test_user_seed) +')\\pl_exponent='  + str(pl_exponent) +',var=' + str(shadowing_var)  + '\\Cthr=' + str(int(Cthr)) +',all_beta=' + str(all_beta) +  '//alpha=' + str(alpha) + '//'

if not os.path.exists(file_path):
    os.makedirs(file_path)

for l_idx in range(lr.shape[0]):
    lr_path = file_path + 'lr=' + str(lr[l_idx]) +'\\'
    if not os.path.exists(lr_path):
        os.makedirs(lr_path)
    with open(lr_path + '\dnn_iscenter_train.dat', 'w') as f_i:
        pass
    with open(lr_path + '\dnn_sum_capa_train.dat', 'w') as f_sc:
        pass
    with open(lr_path + '\dnn_total_power_train.dat', 'w') as f_p:
        pass

for j in range(lr.shape[0]):
    dnn0bj.train_dnn(input_h_train, lr[j], file_path)

for l_idx in range(lr.shape[0]):
    lr_path = file_path + 'lr=' + str(lr[l_idx]) +'\\'
    with open(lr_path + '\dnn_iscenter_test.dat', 'w') as f_i:
        pass
    with open(lr_path + '\dnn_sum_capa_test.dat', 'w') as f_sc:
        pass
    with open(lr_path + '\dnn_total_power_test.dat', 'w') as f_p:
        pass

for j in range(lr.shape[0]):
    dnn0bj.test_dnn(input_h_test, lr[j], file_path)