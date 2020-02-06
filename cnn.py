import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from PIL import Image

#Path to the input images
directory = '/home/hansonxufyp/balanced_images_heart_diabetes/'

def get_files(file_dir):
    #path to the file, the label of the file and path into each class directory
    healthy = []
    label_healthy = []
    healthy_directory = file_dir + "healthy/"
    diabetes = []
    label_diabetes = []
    diabetes_directory = file_dir + "diabetes/"
    heart_disease = []
    label_heart_disease = []
    heart_disease_directory = file_dir + "heart_disease/"

    #Extract the image in healthy class, note that the label has to start with 0
    for file in os.listdir(healthy_directory):
        healthy.append(healthy_directory+file)
        label_healthy.append(0)
    num_healthy=len(healthy)

    for file in os.listdir(diabetes_directory):
        diabetes.append(diabetes_directory+file)
        label_diabetes.append(1)
    num_dia=len(diabetes)


    for file in os.listdir(heart_disease_directory):
        heart_disease.append(heart_disease_directory+file)
        label_heart_disease.append(2)
    num_hd=len(heart_disease)

    #print the number of images in each class, checking the correctness.
    print('%d healthy\n %d diabetes\n %d heart_disease\n'\
          %(num_healthy,num_dia,num_hd))

    imgls = np.hstack((healthy,diabetes,heart_disease))
    lbls = np.hstack((label_healthy,label_diabetes,label_heart_disease))

    arratem = np.array([imgls,lbls])
    arratem = arratem.transpose()
    np.random.shuffle(arratem)

    imgls = list(arratem[:,0])
    lbls = list(arratem[:,1])
    lbls = [int(i) for i in lbls]

    return  imgls,lbls


def get_batch(img, lb, sz_bat, cap):
    #type conversion
    img = tf.cast(img,tf.string)
    lb = tf.cast(lb,tf.int32)
    
    quein = tf.train.slice_input_producer([img,lb])

    lb = quein[1]
    imgcontent = tf.read_file(quein[0])
    img = tf.image.decode_bmp(imgcontent,channels=3)
    img.set_shape([192,256,3])
    img = tf.cast(img, tf.float32)

    img_bat,lb_bat = tf.train.batch([img,lb], sz_bat, num_threads=16, capacity=cap)
    lb_bat = tf.reshape(lb_bat,[sz_bat])
    return img_bat, lb_bat

def inference(imgs, sz_bat, num_clas, keep_prob):
    # 1ST CONVOLUTION LAYER
    with tf.variable_scope("con1", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[3, 3, 3, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        layer_c = tf.nn.conv2d(imgs, para_w, strides=[1, 1, 1, 1], padding="SAME")
        act_before = tf.nn.bias_add(layer_c, para_b)
        con1 = tf.nn.relu(act_before, name="con1")

    
    with tf.variable_scope("pooling1_lrn") as scope:
        pool1 = tf.nn.max_pool(con1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling1")
        layer_normal1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='layer_normal1')

    # 2 CONVOLUTION LAYER
    with tf.variable_scope("con2", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        layer_c = tf.nn.conv2d(layer_normal1, para_w, strides=[1, 1, 1, 1], padding="SAME")
        act_before = tf.nn.bias_add(layer_c, para_b)
        con2 = tf.nn.relu(act_before, name="con2")

    
    with tf.variable_scope("pooling2_lrn") as scope:
        pool2 = tf.nn.max_pool(con2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling2")
        layer_normal2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='layer_normal2')

    # 3 CONVOLUTION LAYER
    with tf.variable_scope("con3", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        layer_c = tf.nn.conv2d(layer_normal2, para_w, strides=[1, 1, 1, 1], padding="SAME")
        act_before = tf.nn.bias_add(layer_c, para_b)
        con3 = tf.nn.relu(act_before, name="con3")

    
    with tf.variable_scope("pooling3_lrn") as scope:
        pool3 = tf.nn.max_pool(con3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling3")
        layer_normal3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='layer_normal3')

    # 4 CONVOLUTIONAL LAYER
    with tf.variable_scope("con4", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[3, 3, 50, 50],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=1, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[50],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        layer_c = tf.nn.conv2d(layer_normal3, para_w, strides=[1, 1, 1, 1], padding="SAME")
        act_before = tf.nn.bias_add(layer_c, para_b)
        con4 = tf.nn.relu(act_before, name="con4")

    
    with tf.variable_scope("pooling4_lrn") as scope:
        pool4 = tf.nn.max_pool(con4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding="SAME", name="pooling4")
        layer_normal4 = tf.nn.lrn(pool4, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          beta=0.75, name='layer_normal4')

    # 1 FULLLYCONNECTED LAYER
    with tf.variable_scope("fu1", reuse=tf.AUTO_REUSE) as scope:
        changeshap = tf.reshape(layer_normal4, shape=[sz_bat, -1])
        weidu = changeshap.get_shape()[1].value
        para_w = tf.get_variable("para_w",
                                  shape=[weidu, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        fu1 = tf.nn.relu(tf.matmul(changeshap, para_w) + para_b, name="fu1")
        #dropout
        fu1_drop = tf.nn.dropout(fu1, keep_prob=keep_prob)
    # 2 FULLLYCONNECTED LAYER
    with tf.variable_scope("fu2", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        fu2 = tf.nn.relu(tf.matmul(fu1_drop, para_w) + para_b, name="fu2")
        #dropout
        fu2_drop = tf.nn.dropout(fu2, keep_prob=keep_prob)

    # SOFTMAXLINEAR_LAYER
    with tf.variable_scope("softmaxlinear", reuse=tf.AUTO_REUSE) as scope:
        para_w = tf.get_variable("para_w",
                                  shape=[128, num_clas],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))
        para_b = tf.get_variable("para_b",
                                 shape=[num_clas],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(1))
        softmaxlinear = tf.add(tf.matmul(fu2_drop, para_w), para_b, name="softmaxlinear")
    return softmaxlinear

def los(lgts, lbs):
    with tf.variable_scope("funclos") as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=lgts,
                                                                       labels=lbs, name="xentropy_per_example")
        funclos = tf.reduce_mean(cross_entropy, name="funclos")
        tf.summary.scalar(scope.name + "funclos", funclos)
    return funclos

def traininng(funclos, para_lr):
    with tf.name_scope("func_opt"):
        func_opt = tf.train.AdamOptimizer(learning_rate=para_lr)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optra = func_opt.minimize(funclos, global_step=global_step)
    return optra

def func_eva(lgts, lbs):
    with tf.variable_scope("accu") as scope:
        zhengque = tf.nn.in_top_k(lgts, lbs, 1)
        zhengque = tf.cast(zhengque, tf.float16)
        accu = tf.reduce_mean(zhengque)
        tf.summary.scalar(scope.name + "accu", accu)
    return accu

N_CLASSES = 3
SZBAT = 32
CAP = 947
MOST_STP = 14000
para_lr = 0.0001

def runcnn():
    directory = '/home/hansonxufyp/balanced_images_heart_diabetes/'
    logs_directory = './newlog/'
    data, lb = get_files(directory)
    num_example = 947
    ratio=0.8
    STEPS_ONE_EPOCH_TRA = int(round((num_example * ratio) / SZBAT))
    STEPS_ONE_EPOCH_TEST = int(round((num_example * (1-ratio)) / SZBAT))
    s=np.int(num_example*ratio)
    traindata=data[:s]
    trainlb=lb[:s]
    test=data[s:]
    test_label=lb[s:]

    battra,battralb = get_batch(traindata,trainlb,SZBAT,CAP)
    test_batch,test_label_batch = get_batch(test,test_label,SZBAT,CAP)
    keep_prob_tra=0.85 
    train_lgts =inference(battra,SZBAT,N_CLASSES, keep_prob_tra)
    trainlos = los(train_lgts,battralb)
    optra = traininng(trainlos,para_lr)
    accutra = func_eva(train_lgts,battralb)
    
    test_lgts = inference(test_batch, SZBAT,N_CLASSES, 1)
    test_loss = los(test_lgts, test_label_batch)
    test_acc = func_eva(test_lgts, test_label_batch)


    oop_sum = tf.summary.merge_all()
    sess = tf.Session()
    writetra = tf.summary.FileWriter(logs_directory,sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    multithread = tf.train.start_queue_runners(sess = sess,coord = coord)
    tlosstrainn_epoch = []
    taccutrainu_epoch = []
    test_lossn_epoch = []
    test_accu_epoch = []

	
    try:
        for stps in np.arange(MOST_STP):
            if coord.should_stop():
                break
            _,tlosstrain,taccutrain = sess.run([optra,trainlos,accutra])
            test_loss, test_acc = sess.run([test_loss,test_acc])
            if stps %  10 == 0:
                print('Step %d Train_loss %.2f accuracy %.2f ,test_loss %.2f accuracy %.2f'%(stps,tlosstrain,taccutrain,test_loss,test_acc))
                st_sum = sess.run(oop_sum)
                writetra.add_summary(st_sum,stps)
        

            if stps % 200 ==0 or (stps +1) == MOST_STP:
                tcheckpointadd = os.path.join(logs_directory,'model.ckpt')
                saver.save(sess,tcheckpointadd,global_step = stps)
        
        # Get the overall training loss and accuracy, calculated using one epoch
        for stps in np.arange(STEPS_ONE_EPOCH_TRA):
            if coord.should_stop():
                break
            tlosstrainn, taccutrainu = sess.run([trainlos, accutra])
            tlosstrainn_epoch.append(tlosstrainn)
            taccutrainu_epoch.append(taccutrainu)

        # Get the overall testing accuracy and testing accuracy, calculated using one epoch
        for stps in np.arange(STEPS_ONE_EPOCH_TEST):
            if coord.should_stop():
                break
            test_lossn, test_accu = sess.run([test_loss, test_acc])
            test_lossn_epoch.append(test_lossn)
            test_accu_epoch.append(test_accu)

        tlosstrain_avg = sum(tlosstrainn_epoch) / len(tlosstrainn_epoch)
        taccutrainu_avg = sum(taccutrainu_epoch) / len(taccutrainu_epoch)
        test_loss_avg = sum(test_lossn_epoch) / len(test_lossn_epoch)
        test_accu_avg = sum(test_accu_epoch) / len(test_accu_epoch)
        print('Overall_Train_loss %.2f accuracy %.2f ,Overall_test_loss %.2f accuracy %.2f'%(tlosstrain_avg, taccutrainu_avg, test_loss_avg, test_accu_avg))
    except tf.errors.OutOfRangeError:
        print('error: theres too many eepochess!')
    finally:
        coord.request_stop()

    coord.join(multithread)
    sess.close()

runcnn()
