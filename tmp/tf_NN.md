# TensorFlow 学习之神经网络的构建

- ### [Install tensorflow](https://github.com/luanxxys/env/blob/master/tensorflow.md)

- ### 实例:创建一个简单的两层神经网络(三输入*四输出)

    ```python
    #训练一个二次函数
    import tensorflow as tf
    import numpy as np

    def add_layer(inputs , in_size , out_size , activate = None):

        #随机初始化
        Weights = tf.Variable(tf.random_normal([in_size,out_size]))

        #可以随机但是不要初始化为0，都为固定值比随机好点
        baises = tf.Variable(tf.zeros([1,out_size])+0.1)

        #matmul:矩阵乘法，multipy:一般是数量的乘法
        y = tf.matmul(inputs, Weights) + baises
        if activate:
            y = activate(y)
        return y
    if __name__ == '__main__':

        #创建-1,1的300个数，此时为一维矩阵，后面转化为二维矩阵===[1,2,3]-->>[[1,2,3]]
        x_data = np.linspace(-1,1,300,dtype=np.float32)[:,np.newaxis]

        #噪声是(1,300)格式,0-0.05大小
        noise = np.random.normal(0,0.05,x_data.shape).astype(np.float32)

        #带有噪声的抛物线
        y_data = np.square(x_data) - 0.5 + noise

        #外界输入数据
        xs = tf.placeholder(tf.float32,[None,1])
        ys = tf.placeholder(tf.float32,[None,1])

        l1 = add_layer(xs,1,10,activate=tf.nn.relu)
        prediction = add_layer(l1,10,1,activate=None)

        #误差
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),reduction_indices=[1]))

        #对误差进行梯度优化，步伐为0.1
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

        sess = tf.Session()
        sess.run( tf.global_variables_initializer())
        for i in range(1000):

            #训练
            sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
            if i%50 == 0:

                #查看误差
                print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    ```

- ### reference

    + [理解 Python 中的 if __name__ == '__main__'](https://github.com/luanxxys/code/blob/master/python/if%20__name__%20%3D%3D%20'__main__'%20.md)
