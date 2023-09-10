import tensorflow as tf

from custom import *


class Attention(layers.Layer):
    def __init__(self,
                 in_channel: int,
                 filters: int,
                 kernel_size: tuple,
                 groups: int,
                 num_kernel: int,
                 reduction: float,
                 min_attention_channel: int = 16,
                 **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.in_channel = in_channel
        self.filters = filters
        self.kernel_size = kernel_size
        self.groups = groups
        self.num_kernel = num_kernel
        self.reduction = reduction
        self.min_attention_channel = min_attention_channel

        self.temperature = 1.0

    def build(self, input_shape):
        attention_channel = max(int(self.in_channel * self.reduction), self.min_attention_channel)
        self.avgpool = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc = layers.Conv2D(filters=attention_channel,
                                kernel_size=(1, 1),
                                use_bias=False)  # 全连接层使用1*1的卷积层代替
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()

        # channel方面注意力
        self.channel_fc = layers.Conv2D(filters=self.in_channel,
                                        kernel_size=(1, 1),
                                        use_bias=True)
        self.func_channel = self.get_channel_attention

        # filter方面注意力
        if self.in_channel == self.groups and self.in_channel == self.filters:
            '''如果odconv中使用的是depth-wise convolution,则不计算filter的注意力'''
            self.func_filter = self.skip
        else:
            self.filter_fc = layers.Conv2D(filters=self.filters,
                                           kernel_size=(1, 1),
                                           use_bias=True)
            self.func_filter = self.get_filter_attention

        # spatial方面注意力，卷积空间
        if self.kernel_size == (1, 1):
            '''如果odconv中使用的是point-wise convolution,则不计算spatial的注意力'''
            self.func_spatial = self.skip
        else:
            self.spatial_fc = layers.Conv2D(filters=self.kernel_size[0] ** 2,
                                            kernel_size=(1, 1),
                                            use_bias=True)
            self.func_spatial = self.get_spatial_attention

        # kernel核之间的注意力
        if self.num_kernel == 1:
            '''如果odconv中只使用一个卷积核，则不计算kernel的注意力'''
            self.func_kernel = self.skip
        else:
            self.kernel_fc = layers.Conv2D(filters=self.num_kernel,
                                           kernel_size=(1, 1),
                                           use_bias=True)
            self.func_kernel = self.get_kernel_attention

    @staticmethod
    def skip(_):
        return 1.0

    def update_temperature(self, temperature):
        '''
        设置注意力分数权重权重
        :param temperature:
        :return:
        '''
        self.temperature = temperature

    def get_channel_attention(self, x):
        '''
        获取输入通道注意力
        :param x:
        :return:
        '''
        channel_attention = tf.reshape(self.channel_fc(x), (tf.shape(x)[0], 1, 1, -1))
        channel_attention = tf.nn.sigmoid(channel_attention / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        '''
        获取卷积滤波器/卷积核之间的注意力
        :param x:
        :return:
        '''
        filter_attention = tf.reshape(self.filter_fc(x), (tf.shape(x)[0], 1, 1, -1))
        filter_attention = tf.nn.sigmoid(filter_attention / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        '''
        获取卷积核内部空间的注意力
        :param x:
        :return:
        '''
        spatial_attention = tf.reshape(self.spatial_fc(x),
                                       (tf.shape(x)[0], 1, self.kernel_size[0], self.kernel_size[1], 1, 1))
        spatial_attention = tf.nn.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        '''
        获取卷积核之间的注意力
        :param x:
        :return:
        '''
        kernel_attention = tf.reshape(self.kernel_fc(x), (tf.shape(x)[0], -1, 1, 1, 1, 1))
        kernel_attention = tf.nn.softmax(kernel_attention / self.temperature, axis=1)
        return kernel_attention

    def call(self, inputs, *args, **kwargs):
        x = self.avgpool(inputs)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        channel_attention = self.func_channel(x)
        filter_attention = self.func_filter(x)
        spatial_attention = self.func_spatial(x)
        kernel_attention = self.func_kernel(x)
        return channel_attention, filter_attention, spatial_attention, kernel_attention


class ODConv2D(layers.Layer):
    def __init__(self,
                 filters: int,
                 kernel_size: tuple,
                 strides: tuple,
                 groups: int,
                 num_kernel: int,
                 reduction: float = 0.0625,
                 **kwargs
                 ):
        super(ODConv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.groups = groups
        self.num_kernel = num_kernel
        self.reduction = reduction

    def build(self, input_shape):
        in_channel = input_shape[-1]
        self.attention = Attention(in_channel=in_channel,
                                   filters=self.filters,
                                   kernel_size=self.kernel_size,
                                   groups=self.groups,
                                   num_kernel=self.num_kernel,
                                   reduction=self.reduction)
        kernel_shape = (self.num_kernel,) + self.kernel_size + (in_channel // self.groups, self.filters)
        self.kernels = self.add_weight(name='kernels',
                                       shape=kernel_shape,
                                       initializer=initializers.get('random_normal'),
                                       regularizer=regularizers.get('l2'),
                                       constraint=constraints.get(None),
                                       dtype=self.dtype,
                                       trainable=True)
        if self.kernel_size == (1, 1) and self.num_kernel == 1:
            '''如果卷积核大小为1*1，并且卷积核个数为1'''
            self._call = self._call_pw1x
        else:
            self._call = self._call_common

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _call_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        inputs = tf.split(x, num_or_size_splits=self.groups, axis=-1)
        kernels = tf.split(tf.squeeze(self.kernels,axis=0), num_or_size_splits=self.groups, axis=-1)
        feats = [tf.nn.conv2d(input=input,
                              filters=kernels[i],
                              strides=self.strides,
                              padding='SAME')
                 for i, input in enumerate(inputs)]
        output = tf.concat(feats, axis=-1)
        output = output * filter_attention
        return output

    def _call_common(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, height, width, in_channel = tf.shape(x)
        x = x * channel_attention
        x = tf.reshape(x, (1, height, width, -1))
        aggregate_weight = spatial_attention * kernel_attention * tf.expand_dims(self.kernels, axis=0)
        aggregate_weight = tf.reshape(tf.reduce_sum(aggregate_weight, axis=1), (
            (self.kernel_size[0], self.kernel_size[1], in_channel // self.groups, -1)))

        inputs = tf.split(x, num_or_size_splits=int(self.groups * batch_size), axis=-1)
        kernels = tf.split(aggregate_weight, num_or_size_splits=int(self.groups * batch_size), axis=-1)
        feats = [tf.nn.conv2d(input=input,
                              filters=kernels[i],
                              strides=self.strides,
                              padding='SAME')
                 for i, input in enumerate(inputs)]
        output = tf.concat(feats, axis=-1)
        output = tf.reshape(output, (batch_size, tf.shape(output)[-2], tf.shape(output)[-3], self.filters))
        output = output * filter_attention
        return output

    def call(self, inputs):
        return self._call(inputs)


if __name__ == '__main__':
    ips = tf.random.normal(shape=(4, 64, 64, 16))
    odconv = ODConv2D(filters=32,
                      kernel_size=(3, 3),
                      strides=(1, 1),
                      groups=2,
                      num_kernel=1)
    ops = odconv(ips)
    1
