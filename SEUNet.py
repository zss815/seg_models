import tensorflow as tf

def GroupNorm(x,name_scope,group=8,esp=1e-5):
    with tf.variable_scope(name_scope,reuse=tf.AUTO_REUSE):
        # tranpose: [bs,h,w,c] to [bs,c,h,w] 
        x = tf.transpose(x,[0,3,1,2])
        N,C,H,W = x.get_shape().as_list()
        G = min(group, C)
        x = tf.reshape(x,[-1,G,C//G,H,W])
        mean, var = tf.nn.moments(x,[2,3,4],keep_dims=True)  #calculate mean and var
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma',[C],initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta',[C],initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma,[1,C,1,1])
        beta = tf.reshape(beta,[1,C,1,1,])
        output = tf.reshape(x,[-1,C,H,W]) * gamma + beta
        output = tf.transpose(output,[0,2,3,1])
    return output


def SELayer(x,reduction,name_scope):
    with tf.variable_scope(name_scope):
        _,H,W,C = x.get_shape().as_list()
        y=tf.layers.average_pooling2d(x,pool_size=(H,W),strides=1,name='global_ave_pool')
        y=tf.reshape(y,[-1,C])
        #y=tf.reduce_mean(x,[1,2],name='global_ave_pool',keep_dims=False)
        y=tf.layers.dense(y,units=C//reduction,name='l1')
        y=tf.nn.leaky_relu(y)
        y=tf.layers.dense(y,units=C,activation=tf.nn.sigmoid,name='l2')
        y=tf.reshape(y,[-1,1,1,C])
    return x*y

def InputLayer(x,multiplier,name_scope):
    with tf.variable_scope(name_scope):
        _,H,W,C = x.get_shape().as_list()
        y=tf.layers.average_pooling2d(x,pool_size=(H,W),strides=1,name='input_global_pool')
        y=tf.reshape(y,[-1,C])
        y=tf.layers.dense(y,units=int(C*multiplier),name='l1')
        y=tf.nn.leaky_relu(y)
        y=tf.layers.dense(y,units=C,activation=tf.nn.sigmoid,name='l2')
        y=tf.add(y,tf.constant(0.5))
        y=tf.reshape(y,[-1,1,1,C])
    return x*y

def SEResBlock(x,filters,reduction,stride,name_scope):
    with tf.variable_scope(name_scope):
        N,H,W,C = x.get_shape().as_list()
        residual=x
        out = tf.layers.conv2d(x,filters,kernel_size=(3,3),strides=stride,padding='same',name='conv1')
        out = GroupNorm(out,name_scope='gn1',group=8)
        out = tf.nn.leaky_relu(out)
        out = tf.layers.conv2d(out,filters,kernel_size=(3,3),strides=1,padding='same',name='conv2')
        out = GroupNorm(out,name_scope='gn2',group=8)
        out = SELayer(out,reduction,name_scope='SE')
        if C!=filters or stride==2:
            y = tf.layers.conv2d(x,filters,kernel_size=(3,3),strides=stride,padding='same',name='conv3')
            residual=GroupNorm(y,name_scope='gn3',group=8)
        out+=residual
        out=tf.nn.leaky_relu(out)
        return out
    
def UpSEResBlock(x1,x2,filters,reduction,stride,name_scope):
    with tf.variable_scope(name_scope):
        _,_,_,C1 = x1.get_shape().as_list()
        _,_,_,C2 = x2.get_shape().as_list()
        if stride==2:
            x1=tf.layers.conv2d_transpose(x1,C1,kernel_size=(3,3),strides=2,padding='same',name='trans_conv')
        x=tf.concat([x1,x2],axis=-1)
        if (C1+C2)!=filters:
            y = tf.layers.conv2d(x,filters,kernel_size=(3,3),strides=1,padding='same',name='conv3')
            residual = GroupNorm(y,name_scope='gn3',group=8)
        else:
            residual=x
        out = tf.layers.conv2d(x,filters,kernel_size=(3,3),strides=1,padding='same',name='conv1')
        out = GroupNorm(out,name_scope='gn1',group=8)
        out = tf.nn.leaky_relu(out)
        out = tf.layers.conv2d(out,filters,kernel_size=(3,3),strides=1,padding='same',name='conv2')
        out = GroupNorm(out,name_scope='gn2',group=8)
        out = SELayer(out,reduction,name_scope='SE')
        out+=residual
        out=tf.nn.leaky_relu(out)
        return out 
    
def SEUNet(x,num_classes,reduction,name_scope):
    with tf.variable_scope(name_scope):
        num_channels=[16,32,64,128,256]
        x=InputLayer(x,multiplier=3,name_scope='Input')
        x=tf.layers.conv2d(x,num_channels[0],kernel_size=(3,3),strides=1,padding='same',name='conv1')
        x = GroupNorm(x,name_scope='gn1',group=8)
        down1_map=tf.nn.leaky_relu(x)   #16 1
        x=SEResBlock(down1_map,num_channels[0],reduction,stride=2,name_scope='Down1') #16 1/2
        down2_map=SEResBlock(x,num_channels[1],reduction,stride=1,name_scope='SERes1') #32 1/2
        x=SEResBlock(down2_map,num_channels[1],reduction,stride=2,name_scope='Down2') #32 1/4
        down3_map=SEResBlock(x,num_channels[2],reduction,stride=1,name_scope='SERes2') #64 1/4
        x=SEResBlock(down3_map,num_channels[2],reduction,stride=1,name_scope='Down3') #64 1/4
        down4_map=SEResBlock(x,num_channels[3],reduction,stride=1,name_scope='SERes3') #128 1/4
        x=SEResBlock(down4_map,num_channels[3],reduction,stride=1,name_scope='Down4') #128 1/4
        down5_map=SEResBlock(x,num_channels[4],reduction,stride=1,name_scope='SERes4') #256 1/4
        x=SEResBlock(down5_map,num_channels[3],reduction,stride=1,name_scope='Down5') #128 1/4
        x=UpSEResBlock(x,down4_map,num_channels[3],reduction,stride=1,name_scope='Up1') #128 1/4
        x=SEResBlock(x,num_channels[2],reduction,stride=1,name_scope='SERes5') #64 1/4
        x=UpSEResBlock(x,down3_map,num_channels[2],reduction,stride=1,name_scope='Up2') #64 1/4
        x=SEResBlock(x,num_channels[1],reduction,stride=1,name_scope='SERes6') #32 1/4
        x=UpSEResBlock(x,down2_map,num_channels[1],reduction,stride=2,name_scope='Up3') #32 1/2
        x=SEResBlock(x,num_channels[0],reduction,stride=1,name_scope='SERes7') #16 1/2
        x=UpSEResBlock(x,down1_map,num_channels[0],reduction,stride=2,name_scope='Up4') #16 1
        x=tf.layers.conv2d(x,num_classes,kernel_size=(1,1),strides=1,padding='same',name='conv_final')
        return x
    
    
        
#x = tf.placeholder(tf.float32,shape=(None,224,224,3))
#y=SEResUNet(x,num_classes=2,reduction=8,name_scope='Model')
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    y=sess.run(y,feed_dict={x:np.random.normal(size=(1,224,224,3))})
#    print(y.shape)
    
    
    
        
    
    
    




    
    
    
        
    
    
    



