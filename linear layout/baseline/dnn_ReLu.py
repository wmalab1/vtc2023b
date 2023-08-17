import tensorflow as tf
import numpy as np
import datetime
import os

print("===== np version: %s =====" %np.__version__)
print("===== tf version: %s =====" %tf.__version__)
print("===== Is GPU available?: %s =====" %tf.test.is_gpu_available())

def log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def file_write(file_name, write_type, data):
    with open(file_name, write_type) as f:
        if data.shape[0] == data.size:
            for i in range(data.shape[0]):
                f.write('%10.10g\n' % (data[i]))
        else:
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    f.write('%10.10g    ' % (data[i, j]))
                f.write('\n')
    f.close()

def file_read(file_name, data_type):
    f = open(file_name, 'r')
    data = f.readlines()
    w = np.zeros_like(data_type)
    for i, line in enumerate(data):
        line = line.rstrip("\n")
        line = line.rstrip()
        if data_type.shape[0] == data_type.size:
            w[i] = float(line)
        else:
            try:
                a = line.split()
                a = [float(j) for j in a]

                w[i,:] = a

            except ValueError as e:
                print(e)
                print ("on line %d" %i)
                print(data_type.shape)
                print("a.shape: %s" %len(a))
                print("w.shape: %s" %w[i,:].shape)
    f.close()
    return w

class Dnn:
    def __init__(self, batch_size=1, n_epoch=500, layer_dim_list=[16, 16, 3], num_cell=7, num_user_per_cell=16,
                 N0=1e-9, W=1e7, all_beta=1, Cthr=1e6, alpha=1):

        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.layer_dim_list = layer_dim_list

        self.num_cell = num_cell
        self.num_user_per_cell = num_user_per_cell

        self.num_edge = int(num_user_per_cell/2)
        self.num_center = int(num_user_per_cell/2)

        self.center_FRF = 3
        self.edge_FRF = 3

        self.N0 = N0
        self.W = W

        self.all_beta = all_beta

        self.Cthr = Cthr
        self.alpha = alpha

    def get_itfc_indicator_linear(self, is_center):

        BP_green = np.array([2, 5])
        BP_green = BP_green[BP_green <= self.num_cell]
        BP_green_idx = BP_green - 1
        bin_BP_green = np.zeros([self.num_cell])
        bin_BP_green[BP_green_idx] = 1
        bin_BP_green_complement = np.ones([self.num_cell])
        bin_BP_green_complement[BP_green_idx] = 0

        BP_blue = np.array([1, 4])
        BP_blue = BP_blue[BP_blue <= self.num_cell]
        BP_blue_idx = BP_blue - 1
        bin_BP_blue = np.zeros([self.num_cell])
        bin_BP_blue[BP_blue_idx] = 1
        bin_BP_blue_complement = np.ones([self.num_cell])
        bin_BP_blue_complement[BP_blue_idx] = 0

        BP_red = np.array([3, 6])
        BP_red = BP_red[BP_red <= self.num_cell]
        BP_red_idx = BP_red - 1
        bin_BP_red = np.zeros([self.num_cell])
        bin_BP_red[BP_red_idx] = 1
        bin_BP_red_complement = np.ones([self.num_cell])
        bin_BP_red_complement[BP_red_idx] = 0

        indicator_itfc_bp = np.zeros([2, self.num_cell, self.num_cell])
        indicator_itfc_p = np.zeros([2, self.num_cell, self.num_cell])


        if is_center == 'center':
            for sv_cell in range(self.num_cell):
                if np.any(BP_red_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_blue
                    indicator_itfc_bp[1, sv_cell, :] = bin_BP_green
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_blue_complement
                    indicator_itfc_p[1, sv_cell, :] = bin_BP_green_complement

                elif np.any(BP_green_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_blue
                    indicator_itfc_bp[1, sv_cell, :] = bin_BP_red
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_blue_complement
                    indicator_itfc_p[1, sv_cell, :] = bin_BP_red_complement

                elif np.any(BP_blue_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_green
                    indicator_itfc_bp[1, sv_cell, :] = bin_BP_red
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_green_complement
                    indicator_itfc_p[1, sv_cell, :] = bin_BP_red_complement

                indicator_itfc_p[:, sv_cell, sv_cell] = 0
                indicator_itfc_bp[:, sv_cell, sv_cell] = 0

        elif is_center == 'edge':
            for sv_cell in range(self.num_cell):
                if np.any(BP_red_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_red
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_red_complement
                elif np.any(BP_green_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_green
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_green_complement
                elif np.any(BP_blue_idx == sv_cell):
                    indicator_itfc_bp[0, sv_cell, :] = bin_BP_blue
                    indicator_itfc_p[0, sv_cell, :] = bin_BP_blue_complement

                indicator_itfc_p[0, sv_cell, sv_cell] = 0
                indicator_itfc_bp[0, sv_cell, sv_cell] = 0
                indicator_itfc_p[1, sv_cell, :] = np.NaN
                indicator_itfc_bp[1, sv_cell, :] = np.NaN

        return indicator_itfc_bp, indicator_itfc_p

    def Calc_capa(self, input_h, Ptotal):

        input_h_not_scaled = np.reshape(input_h, [self.num_cell, self.num_user_per_cell, self.num_cell])

        [all_c_indicator_itfc_bp, all_c_indicator_itfc_p] = self.get_itfc_indicator_linear('center')
        [all_e_indicator_itfc_bp, all_e_indicator_itfc_p] = self.get_itfc_indicator_linear('edge')

        c_capa_per_cell = np.empty([self.num_cell, self.num_user_per_cell])
        e_capa_per_cell = np.empty([self.num_cell, self.num_user_per_cell])
        delta_capa_per_cell = np.empty([self.num_cell, self.num_user_per_cell])

        k0 = 1
        for c_idx in range(self.num_cell):
            c_indicator_itfc_bp = np.squeeze(all_c_indicator_itfc_bp[:, c_idx, :])
            c_indicator_itfc_p = np.squeeze(all_c_indicator_itfc_p[:, c_idx, :])

            e_indicator_itfc_bp = np.squeeze(all_e_indicator_itfc_bp[0, c_idx, :])
            e_indicator_itfc_p = np.squeeze(all_e_indicator_itfc_p[0, c_idx, :])

            for u_idx in range(self.num_user_per_cell):
                Ic = 0
                Ie = 0
                h = input_h_not_scaled[c_idx, u_idx, :]

                for i in range(self.num_cell):
                    if i != c_idx:
                        itfc_C = Ptotal[i] / (self.num_center * (2 + self.all_beta))
                        c_tmp1 = h[i] * itfc_C * c_indicator_itfc_p[:, i]
                        c_tmp2 = h[i] * self.all_beta * itfc_C * c_indicator_itfc_bp[:, i]
                        Ic = Ic + (c_tmp1 + c_tmp2)

                        e_tmp1 = h[i] * itfc_C * e_indicator_itfc_p[i]
                        e_tmp2 = h[i] * self.all_beta * itfc_C * e_indicator_itfc_bp[i]
                        Ie = Ie + (e_tmp1 + e_tmp2)

                itfc_power_c = k0 * Ic
                itfc_power_e = k0 * Ie
                noise_power_c = self.N0 * (self.W / self.center_FRF)
                noise_power_e = self.N0 * (self.W / self.edge_FRF)

                sv_C = Ptotal[c_idx] / (self.num_center * (2 + self.all_beta))

                SINR_c = k0 * sv_C * h[c_idx] / (noise_power_c + itfc_power_c)
                SINR_e = k0 * self.all_beta * sv_C * h[c_idx] / (noise_power_e + itfc_power_e)

                capacity_c = (self.W / self.center_FRF) * np.log2(1 + SINR_c)
                capacity_c = np.sum(capacity_c)
                capacity_e = (self.W / self.edge_FRF) * np.log2(1 + SINR_e)

                c_capa_per_cell[c_idx, u_idx] = capacity_c
                e_capa_per_cell[c_idx, u_idx] = capacity_e
                delta_capa_per_cell[c_idx, u_idx] = capacity_c - capacity_e

        all_is_center = np.zeros([self.num_cell, self.num_user_per_cell])

        for c_idx in range(self.num_cell):
            sorted_idx = np.argsort(delta_capa_per_cell[c_idx, :])[::-1]
            tmp_frf = np.zeros([self.num_user_per_cell])
            tmp_frf[sorted_idx[0:self.num_center]] = np.repeat(1, self.num_center)
            tmp_frf[sorted_idx[self.num_center:]] = np.repeat(0, self.num_edge)
            all_is_center[c_idx, :] = tmp_frf

        sum_capa = np.zeros([self.num_cell, self.num_user_per_cell])
        for c_idx in range(self.num_cell):
            for u_idx in range(self.num_user_per_cell):
                if all_is_center[c_idx, u_idx] == 1:
                    sum_capa[c_idx, u_idx] = c_capa_per_cell[c_idx, u_idx]
                elif all_is_center[c_idx, u_idx] == 0:
                    sum_capa[c_idx, u_idx] = e_capa_per_cell[c_idx, u_idx]

        return sum_capa, all_is_center


    def train_dnn(self, input_h, lr, file_path):

        self.weights = []
        self.biases = []

        num_batch = int(input_h.shape[0] / self.batch_size)

        seed_weight = 1000
        seed_shuffle = 2000
        np.random.seed(seed_shuffle)

        input_h_not_scaled = input_h
        input_h_log = np.log10(input_h)

        avg_h_log = np.mean(input_h_log)
        std_h_log = np.std(input_h_log)

        input_h = (input_h_log - avg_h_log) / std_h_log

        with open(file_path + '\input_scaling_param.dat', 'w') as f:
            f.write('   %.20g\n   %.20g\n' % (avg_h_log, std_h_log))

        with tf.device('/CPU:0'):
            tf.reset_default_graph()

            x_ph = tf.placeholder(tf.float64, shape=[None, input_h.shape[1]])
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input_h.shape[1]
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i - 1]
                    out_dim = self.layer_dim_list[i]

                weight = tf.Variable(tf.random_normal([in_dim, out_dim], stddev=tf.sqrt(2.0 / tf.cast(in_dim, tf.float64)),seed=seed_weight * (i * i + 1), dtype=tf.float64), dtype=tf.float64)
                bias = tf.Variable(tf.zeros(out_dim, dtype=tf.float64), dtype=tf.float64)
                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list) - 1:
                    out_layer = tf.nn.relu(mult)

                else:
                    output_Ptotal = tf.nn.sigmoid(mult) + 1e-100

                self.weights.append(weight)
                self.biases.append(bias)

            H = tf.reshape(x_ph, [-1, self.num_cell, self.num_user_per_cell,self.num_cell])

            temp = H * std_h_log + avg_h_log
            H = pow(10, temp)

            [all_c_indicator_itfc_bp, all_c_indicator_itfc_p] = self.get_itfc_indicator_linear('center')
            [all_e_indicator_itfc_bp, all_e_indicator_itfc_p] = self.get_itfc_indicator_linear('edge')

            c_capa_per_cell = []
            e_capa_per_cell = []
            delta_capa_per_cell = []
            k0 = 1
            for c_idx in range(self.num_cell):
                c_indicator_itfc_bp = np.squeeze(all_c_indicator_itfc_bp[:, c_idx, :])
                c_indicator_itfc_p = np.squeeze(all_c_indicator_itfc_p[:, c_idx, :])

                e_indicator_itfc_bp = np.squeeze(all_e_indicator_itfc_bp[0, c_idx, :])
                e_indicator_itfc_p = np.squeeze(all_e_indicator_itfc_p[0, c_idx, :])

                tmp_c_capa_per_user = []
                tmp_e_capa_per_user = []
                tmp_delta_capa_per_user = []

                for u_idx in range(self.num_user_per_cell):
                    Ic = 0
                    Ie = 0
                    h = H[:, c_idx, u_idx, :]


                    for i in range(self.num_cell):
                        if i != c_idx:
                            itfc_C = output_Ptotal[:, i] / (self.num_center * (2 + self.all_beta))
                            c_tmp1 = tf.expand_dims(h[:, i], axis=1) * tf.expand_dims(itfc_C, axis=1) * tf.reshape(c_indicator_itfc_p[:, i], [1, -1])
                            c_tmp2 = self.all_beta * tf.expand_dims(itfc_C,axis=1) * tf.expand_dims(h[:, i], axis=1) * tf.reshape(c_indicator_itfc_bp[:, i], [1,-1])
                            Ic = Ic + (c_tmp1 + c_tmp2)

                            e_tmp1 = h[:, i] * itfc_C * e_indicator_itfc_p[i]
                            e_tmp2 = self.all_beta * itfc_C * h[:, i] * e_indicator_itfc_bp[i]
                            Ie = Ie + (e_tmp1 + e_tmp2)

                    itfc_power_c = k0 * Ic
                    itfc_power_e = k0 * Ie
                    noise_power_c = self.N0 * (self.W / self.center_FRF)
                    noise_power_e = self.N0 * (self.W / self.edge_FRF)

                    sv_C = output_Ptotal[:, c_idx] / (self.num_center * (2 + self.all_beta))

                    SINR_c = tf.expand_dims(k0 * sv_C * h[:, c_idx], axis=1) / (noise_power_c + itfc_power_c)
                    SINR_e = (k0 * self.all_beta * sv_C * h[:, c_idx]) / (noise_power_e + itfc_power_e)

                    capacity_c = (self.W / self.center_FRF) * log2(1 + SINR_c)
                    capacity_c = tf.reduce_sum(capacity_c, axis=1)
                    capacity_e = (self.W / self.edge_FRF) * log2(1 + SINR_e)

                    tmp_c_capa_per_user.append(capacity_c)
                    tmp_e_capa_per_user.append(capacity_e)
                    tmp_delta_capa_per_user.append(capacity_c - capacity_e)

                tmp_c_capa_per_user = tf.convert_to_tensor(tmp_c_capa_per_user)
                tmp_e_capa_per_user = tf.convert_to_tensor(tmp_e_capa_per_user)
                tmp_delta_capa_per_user = tf.convert_to_tensor(tmp_delta_capa_per_user)

                c_capa_per_cell.append(tmp_c_capa_per_user)
                e_capa_per_cell.append(tmp_e_capa_per_user)
                delta_capa_per_cell.append(tmp_delta_capa_per_user)

            c_capa_per_cell = tf.convert_to_tensor(c_capa_per_cell)
            e_capa_per_cell = tf.convert_to_tensor(e_capa_per_cell)
            delta_capa_per_cell = tf.convert_to_tensor(delta_capa_per_cell)

            tst1 = tf.concat(
                [tf.stack([np.arange(self.num_center), np.repeat(j, self.num_center)], axis=1)
                 for j in range(self.batch_size)
                 ], axis=0
            )

            tst2 = tf.concat(
                [tf.stack([np.arange(self.num_center, self.num_user_per_cell), np.repeat(j, self.num_edge)], axis=1)
                 for j in range(self.batch_size)
                 ], axis=0
            )

            tmp_center_idx = []
            tmp_edge_idx = []
            sum_capa = 0

            center_capa = []
            edge_capa = []

            for c_idx in range(self.num_cell):
                sorted_idx = tf.argsort(delta_capa_per_cell[c_idx, :, :], axis=0,direction='DESCENDING')

                center_idx = tf.reshape(tf.gather_nd(sorted_idx, tst1), [-1, self.num_center])
                tmp_center_idx.append(center_idx)

                tst3 = tf.concat(
                    [tf.stack([center_idx[j, :], np.repeat(j, self.num_center)], axis=1)
                     for j in range(self.batch_size)
                     ], axis=0
                )

                tmp_c_capa = tf.reshape(tf.gather_nd(c_capa_per_cell[c_idx, :, :], tst3),[-1, self.num_center])
                center_capa.append(tmp_c_capa)

                edge_idx = tf.reshape(tf.gather_nd(sorted_idx, tst2), [-1, self.num_edge])
                tmp_edge_idx.append(edge_idx)

                tst3 = tf.concat(
                    [tf.stack([edge_idx[j, :], np.repeat(j, self.num_edge)], axis=1)
                     for j in range(self.batch_size)
                     ], axis=0
                )

                tmp_e_capa = tf.reshape(tf.gather_nd(e_capa_per_cell[c_idx, :, :], tst3),[-1, self.num_edge])
                edge_capa.append(tmp_e_capa)

                tmp_sum_capa = tf.reduce_sum(tmp_c_capa, axis=1, keepdims=True) + tf.reduce_sum(tmp_e_capa, axis=1,keepdims=True)
                sum_capa = sum_capa + tmp_sum_capa


            edge_capa = tf.convert_to_tensor(edge_capa)
            center_capa = tf.convert_to_tensor(center_capa)

            edge_capa_per_batch = []
            center_capa_per_batch = []
            for j in range(self.batch_size):
                tmp_e = tf.reshape(edge_capa[:, j, :], [self.num_cell * self.num_edge])
                edge_capa_per_batch.append(tmp_e)

                tmp_c = tf.reshape(center_capa[:, j, :], [self.num_cell * self.num_center])
                center_capa_per_batch.append(tmp_c)

            edge_capa_per_batch = tf.convert_to_tensor(edge_capa_per_batch)
            center_capa_per_batch = tf.convert_to_tensor(center_capa_per_batch)

            capacity_prime = sum_capa - self.alpha * (tf.nn.relu(self.Cthr - center_capa_per_batch) + tf.nn.relu(self.Cthr - edge_capa_per_batch))  # batch_size, num_cell* num_center

            loss = tf.reduce_sum(-1 * capacity_prime)

            optimizer = tf.compat.v1.train.AdamOptimizer(lr)
            train = optimizer.minimize(loss)

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)


            start_time = datetime.datetime.now()
            print('======== Start Time: %s ========\n' % start_time)

            loss_per_batch = np.zeros([num_batch])
            loss_per_epoch = np.zeros([self.n_epoch])

            for e in range(self.n_epoch):
                for j in range(num_batch):
                    input_batch_h = input_h[j * self.batch_size: (j + 1) * self.batch_size,:]
                    sess.run(train, feed_dict={x_ph: input_batch_h})
                    loss_per_batch[j] = sess.run(loss, feed_dict={x_ph: input_batch_h})
                loss_per_epoch[e] = np.mean(loss_per_batch)

                if (e + 1) % 10 == 0:
                    now_time = datetime.datetime.now()
                    remain_time = (now_time - start_time) * self.n_epoch / (e + 1) - (now_time - start_time)
                    print('epoch= %6d | lr=%g | loss= %8.10g | remain = %s(h:m:s)' % (e + 1, lr, loss_per_epoch[e], remain_time))

            ww, bb = sess.run([self.weights, self.biases])
            dnn_Ptotal_per_sample = sess.run(output_Ptotal,feed_dict={x_ph: input_h})

            dnn_sum_capa_per_sample = np.zeros([input_h.shape[0], self.num_cell, self.num_user_per_cell])
            dnn_is_center_per_sample = np.zeros([input_h.shape[0], self.num_cell, self.num_user_per_cell])

            for m in range(input_h.shape[0]):
                dnn_sum_capa_per_sample[m, :, :], dnn_is_center_per_sample[m, :, :] = self.Calc_capa(input_h_not_scaled[m, :], dnn_Ptotal_per_sample[m, :])

        sess.close()

        lr_path = file_path + '\lr=' + str(lr)
        if not os.path.exists(lr_path):
            os.makedirs(lr_path)

        wb_path = file_path + '\w_b'
        if not os.path.exists(wb_path):
            os.makedirs(wb_path)

        for i in range(len(self.layer_dim_list)):
            file_write(wb_path + '\W' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '.dat', 'w', ww[i][:, :])
            file_write(wb_path + '\B' + str(i + 1) + '_lr' + str(format(lr, "1.0e")) + '.dat', 'w', bb[i][:])


        epoch = np.arange(0, self.n_epoch, 1)
        with open(lr_path + '\loss_per_epoch_train.dat', 'w') as f:
            for j in range(self.n_epoch):
                f.write('%d\t %8.10g\n' % (epoch[j] + 1, loss_per_epoch[j]))
            f.close()

        for m in range(input_h.shape[0]):
            with open(lr_path + '\dnn_total_power_train.dat', 'a') as f_p:
                f_p.write('%g\t' % (lr))
                for c in range(self.num_cell):
                    f_p.write('%10.10g  ' % dnn_Ptotal_per_sample[m, c])
                f_p.write('\n')

            with open(lr_path + '\dnn_sum_capa_train.dat', 'a') as f_sc:
                for c in range(self.num_cell):
                    f_sc.write('%g\t' % (lr))
                    for k in range(self.num_user_per_cell):
                        f_sc.write('%20.20g  ' % dnn_sum_capa_per_sample[m, c, k])
                    f_sc.write('\n')

            with open(lr_path + '\dnn_iscenter_train.dat', 'a') as f_i:
                for c in range(self.num_cell):
                    f_i.write('%g\t' % (lr))
                    for k in range(self.num_user_per_cell):
                        f_i.write('%10.10g  ' % dnn_is_center_per_sample[m, c, k])
                    f_i.write('\n')


        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now() - start_time))

    def test_dnn(self, input_h, lr, file_path):

        start_time = datetime.datetime.now()
        print('======== Start Time: %s ========\n' % start_time)

        lr_path = file_path + '\lr=' + str(lr)

        with open(file_path + '\input_scaling_param.dat', 'r') as f:
            lines = f.readlines()

        avg_h_log = float(lines[0].strip())
        std_h_log = float(lines[1].strip())

        input_h_not_scaled = input_h
        input_h_log = np.log10(input_h)

        input_h = (input_h_log - avg_h_log) / std_h_log

        with tf.device('/CPU:0'):
            tf.reset_default_graph()

            x_ph = tf.placeholder(tf.float64, shape=[None, input_h.shape[1]])
            for i in range(len(self.layer_dim_list)):
                if i == 0:
                    in_layer = x_ph
                    in_dim = input_h.shape[1]
                    out_dim = self.layer_dim_list[i]
                else:
                    in_layer = out_layer
                    in_dim = self.layer_dim_list[i - 1]
                    out_dim = self.layer_dim_list[i]

                weight = np.zeros([in_dim, out_dim], dtype=np.float64)
                bias = np.zeros(out_dim, dtype=np.float64)

                wb_path = file_path + '\w_b'
                weight = file_read(wb_path + "\W" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", weight)
                bias = file_read(wb_path + "\B" + str(i + 1) + "_lr" + str(format(lr, "1.0e")) + ".dat", bias)

                weight = tf.convert_to_tensor(weight)
                bias = tf.convert_to_tensor(bias)

                mult = tf.matmul(in_layer, weight) + bias

                if i < len(self.layer_dim_list) - 1:
                    out_layer = tf.nn.relu(mult)

                else:
                    output_Ptotal = tf.nn.sigmoid(mult) + 1e-100

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            dnn_Ptotal_per_sample = sess.run(output_Ptotal,feed_dict={x_ph: input_h})

            dnn_sum_capa_per_sample = np.zeros([input_h.shape[0], self.num_cell, self.num_user_per_cell])
            dnn_is_center_per_sample = np.zeros([input_h.shape[0], self.num_cell, self.num_user_per_cell])

            for m in range(input_h.shape[0]):
                dnn_sum_capa_per_sample[m, :, :], dnn_is_center_per_sample[m, :, :] = self.Calc_capa(input_h_not_scaled[m, :], dnn_Ptotal_per_sample[m, :])

        sess.close()

        for m in range(input_h.shape[0]):
            with open(lr_path + '\dnn_total_power_test.dat', 'a') as f_p:
                f_p.write('%g\t' % (lr))
                for c in range(self.num_cell):
                    f_p.write('%10.10g  ' % dnn_Ptotal_per_sample[m, c])
                f_p.write('\n')

            with open(lr_path + '\dnn_sum_capa_test.dat', 'a') as f_sc:
                for c in range(self.num_cell):
                    f_sc.write('%g\t' % (lr))
                    for k in range(self.num_user_per_cell):
                        f_sc.write('%20.20g  ' % dnn_sum_capa_per_sample[m, c, k])
                    f_sc.write('\n')

            with open(lr_path + '\dnn_iscenter_test.dat', 'a') as f_i:
                for c in range(self.num_cell):
                    f_i.write('%g\t' % (lr))
                    for k in range(self.num_user_per_cell):
                        f_i.write('%10.10g  ' % dnn_is_center_per_sample[m, c, k])
                    f_i.write('\n')
        print('======== Elapsed Time: %s (h:m:s) ========\n' % (datetime.datetime.now() - start_time))



















