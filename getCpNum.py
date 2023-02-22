import cpIdentify.province_test as province
import tensorflow._api.v2.compat.v1 as tf
import glob

province_list=['京','闽','鄂','苏','沪','浙']
numLetter_list=[
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','J','K',
    'L','M','N','P','Q','R','S','T','U','V',
    'W','X','Y','Z']
province_classes=6
num_classes=34
def create_weights(shape):
    # 定义权重和偏置  stddev 标准差
    return tf.Variable(initial_value = tf.random_normal(shape = shape,stddev = 0.01))
def create_model(x,prob,classes):
    # 构建卷积神经网络模型
    # x: [None,20,80,3]
    # 1、第一卷积大层
    with tf.variable_scope('conv1'):
        conv1_weights=create_weights(shape = [5,5,1,32])
        conv1_bias=create_weights(shape = [32])
        conv1_x=tf.nn.conv2d(input = x,filter = conv1_weights,strides = [1,1,1,1],padding="SAME")+conv1_bias
        relu1_x=tf.nn.relu(conv1_x)
        pool1_x=tf.nn.max_pool(value =relu1_x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
        drop1_x = tf.nn.dropout(pool1_x, keep_prob=prob)
    with tf.variable_scope('conv2'):
        conv2_weights = create_weights(shape = [5, 5, 32, 64])
        conv2_bias = create_weights(shape = [64])
        conv2_x = tf.nn.conv2d(input = drop1_x, filter = conv2_weights, strides = [1, 1, 1, 1],padding = "SAME") + conv2_bias
        # 激活层
        relu2_x = tf.nn.relu(conv2_x)
        # 池化层
        pool2_x = tf.nn.max_pool(value = relu2_x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        drop2_x = tf.nn.dropout(pool2_x, keep_prob=prob)
    # 全连接层
    with tf.variable_scope('full_connection'):
        # 2、构建模型 - 全连接
        x_fc=tf.reshape(drop2_x,shape = [-1,8*10*64])
        weights_fc=create_weights(shape = [8*10*64,classes])
        bias_fc=create_weights(shape = [classes])
        y_predict=tf.matmul(x_fc,weights_fc)+bias_fc

    return y_predict
# def read_pic():
#     '''
#     0-京 1-闽 2-鄂 3-苏 4-沪 5-浙
#     :return:
#     '''
#     filename='./train_images/training-set/chinese-characters'
#     filenames=[]
#     for i in range(classes):
#         dir = './train_images/training-set/chinese-characters/%s/' % i
#         file_names=glob.glob(dir+'*.bmp')
#         for file in file_names:
#             filenames.append(file)
#     file_queue = tf.train.string_input_producer(filenames)
#     # 2、读取与解码
#     reader = tf.WholeFileReader()
#     filename, image = reader.read(file_queue)
#     # 解码阶段
#     print(f"image={image}")
#     decoded_image = tf.image.decode_bmp(image)
#     # 更新形状，将图片形状确定下来
#     decoded_image.set_shape([40,32,1])
#     # 修改图片的类型
#     cast_image = tf.cast(decoded_image, tf.float32)
#     # 3、批处理
#     filename_batch,image_batch = tf.train.batch([filename,cast_image], batch_size=50, num_threads=1,
#                                                      capacity=100)
#     return filename_batch, image_batch

def read_num_test():
    '''
         0-9表示数字0-9
        10-A 11-B 12-C 13-D 14—E 15-F 16-G 17-H 18-J 19-K 20-L 21-M 22-N 23-P 24-Q 25-R 26-S 27-T 28-U
        29-V 30-W 31-X 32-Y 33-Z
        :return:
    '''
    file_names=['E:/pictures/cp/chars/1.bmp','E:/pictures/cp/chars/2.bmp','E:/pictures/cp/chars/3.bmp',
                'E:/pictures/cp/chars/4.bmp','E:/pictures/cp/chars/5.bmp','E:/pictures/cp/chars/6.bmp']
    # file_names=['E:/pictures/cp/chars/4.bmp']
    file_queue = tf.train.string_input_producer(file_names,shuffle=False)
    # 2、读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)
    # 解码阶段
    decoded_image = tf.image.decode_bmp(image)
    # 更新形状，将图片形状确定下来
    decoded_image.set_shape([40, 32, 1])
    # 修改图片的类型
    cast_image = tf.cast(decoded_image, tf.float32)
    filename_batch,image_batch = tf.train.batch([filename,cast_image], batch_size=6, num_threads=1,
                                                 capacity=100)
    return filename_batch,image_batch
def read_province_test():
    '''
        0-京 1-闽 2-鄂 3-苏 4-沪 5-浙
        :return:
    '''
    file_names=['E:/pictures/cp/chars/0.bmp']
    file_queue = tf.train.string_input_producer(file_names)
    # 2、读取与解码
    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)
    # 解码阶段
    decoded_image = tf.image.decode_bmp(image)
    # 更新形状，将图片形状确定下来
    decoded_image.set_shape([40, 32, 1])
    # 修改图片的类型
    cast_image = tf.cast(decoded_image, tf.float32)
    filename_batch,image_batch = tf.train.batch([filename,cast_image], batch_size=1, num_threads=1,
                                                 capacity=100)
    return filename_batch,image_batch
def filename2label_test(filenames:list):
    labels=[]
    for filename in filenames:
        file=str(filename)
        # print(file)
        start_index=file.rfind('/')
        end_index = file.rfind('.')
        labels.append(int(file[start_index+1:end_index]))
    return labels
def try_pic():
    g1=tf.Graph()
    with g1.as_default():
        filenames, images = read_province_test()
        x = tf.placeholder(tf.float32, shape=[None, 40, 32, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, province_classes])
        prob=tf.placeholder(tf.float32)
        y_predict = create_model(x,prob,province_classes)
        loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        y_p=tf.nn.softmax(y_predict)
        # print(f"y_p={y_p}")
        loss = tf.reduce_mean(loss_list)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))

        accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))
        # 初始化变量

        init = tf.global_variables_initializer()
    cp_str=""
    with tf.Session(graph=g1) as sess:
        # 初始化变量
        sess.run(init)
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coor)
        saver=tf.train.Saver()
        saver.restore(sess,save_path='D:/python/try/tf_car_license_dataset/model/province.ckpt')
        filenames_value, images_value = sess.run([filenames, images])
        labels =filename2label_test(filenames_value)
        labels_one_hot = tf.reshape(tf.one_hot(labels, depth=province_classes), [-1, province_classes]).eval()
        y_p_value,y_predict_value,loss_value, accuracy_value=sess.run([y_p,y_predict,loss, accuracy],feed_dict={x: images_value, y_true: labels_one_hot,prob:1.0})
        # print(labels_one_hot)
        # print("损失为%f，准确率为%f" % (loss_value, accuracy_value))
        province_index = tf.argmax(y_predict_value[0]).eval();
        # print(f"预测province序号={province_index}")
        # for i in range(province_classes):print("%.6f"%y_p_value[0][i],end=" ")
        print(province_list[province_index],end=" ")
        print("%.6f" % y_p_value[0][province_index])
        coor.request_stop()
        coor.join(threads)
        sess.close()
        cp_str=cp_str+province_list[province_index]
    #============================================================
    new_graph=tf.Graph()
    num_classes=34

    with new_graph.as_default():
        filenames, images = read_num_test()
        x = tf.placeholder(tf.float32, shape=[None, 40, 32, 1])
        y_true = tf.placeholder(tf.float32, shape=[None,num_classes])
        prob = tf.placeholder(tf.float32)
        y_predict = create_model(x, prob,num_classes)
        loss_list = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict)
        y_p = tf.nn.softmax(y_predict)
        # print(f"y_p={y_p}")
        loss = tf.reduce_mean(loss_list)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
        bool_list = tf.equal(tf.argmax(y_true, axis=1), tf.argmax(y_predict, axis=1))

        accuracy = tf.reduce_mean(tf.cast(bool_list, tf.float32))
        # 初始化变量
        init = tf.global_variables_initializer()
    with tf.Session(graph=new_graph) as sess:
        # 初始化变量
        sess.run(init)
        coor = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coor)
        saver = tf.train.Saver()
        saver.restore(sess, save_path='D:/python/try/tf_car_license_dataset/model2/num_letter.ckpt')
        filenames_value, images_value = sess.run([filenames, images])
        labels = filename2label_test(filenames_value)
        labels_one_hot = tf.reshape(tf.one_hot(labels, depth=num_classes), [-1,num_classes]).eval()

        y_p_value, y_predict_value, loss_value, accuracy_value = sess.run([y_p, y_predict, loss, accuracy],
                                                                          feed_dict={x: images_value,
                                                                                     y_true: labels_one_hot, prob: 1.0})
        # print(labels_one_hot)
        # print("损失为%f，准确率为%f" % (loss_value, accuracy_value))
        # print(f"预测province序号={tf.argmax(y_predict_value[0]).eval()}")
        for i in range(6):
            # for j in range(num_classes): print("%.6f" % y_p_value[i][j], end=" ")
            num_index = tf.argmax(y_predict_value[i]).eval();
            print(f'{numLetter_list[num_index]}的y_p={y_p_value[i][num_index]}')
            cp_str = cp_str + (numLetter_list[num_index])
        coor.request_stop()
        coor.join(threads)
        sess.close()
    return cp_str
# if __name__=='__main__':
#     tf.disable_eager_execution()
#     province=try_pic()
#     print(province)
    # test()
def getCpNum():
    tf.disable_eager_execution()
    return try_pic()
if __name__ =='__main__':
    tf.disable_eager_execution()
    print(try_pic())

