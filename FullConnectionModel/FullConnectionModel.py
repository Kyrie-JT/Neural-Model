import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import trange

# 用MNIST数据集做训练和测试

# 载入 MNIST 数据集，如果指定路径 dataset/ 下没有已经下载好的数据，那么 TensorFlow 会自动从数据集所在网址下载数据
mnist = input_data.read_data_sets("dataset/", one_hot=True)

print("Shape of train data:", mnist.train.images.shape)
print("Type of train data:", type(mnist.train.images))
print("Shape of train labels: ", mnist.train.labels.shape)
print("Type of train labels:", type(mnist.train.labels))
'''输出结果为：
Shape of train data: (55000, 784)
Type of train data: <class 'numpy.ndarray'>
Shape of train labels:  (55000, 10)
Type of train labels: <class 'numpy.ndarray'>'''

print("Shape of validation data:", mnist.validation.images.shape)
print("Type of validation data:", type(mnist.validation.images))
print("Shape of validation labels: ", mnist.validation.labels.shape)
print("Type of validation labels:", type(mnist.validation.labels))
'''输出结果为：
Shape of validation data: (5000, 784)
Type of validation data: <class 'numpy.ndarray'>
Shape of validation labels:  (5000, 10)
Type of validation labels: <class 'numpy.ndarray'>'''

print("Shape of test data:", mnist.test.images.shape)
print("Type of test data:", type(mnist.test.images))
print("Shape of test labels: ", mnist.test.labels.shape)
print("Type of test labels:", type(mnist.test.labels))
'''输出结果为：
Shape of test data: (10000, 784)
Type of test data: <class 'numpy.ndarray'>
Shape of test labels:  (10000, 10)
Type of test labels: <class 'numpy.ndarray'>'''

#sigmoid 激活函数
def sigmoid_function(X):
    fx = 1/(1 + np.exp(-X))
    return fx  

class sigmoid():
    def __init__(self):
        self.mem = {}

    def forward(self, X):
            self.mem["X"] = X
            return sigmoid_function(X)

    def backward(self, y_grad):
            X = self.mem["X"]
            return sigmoid_function(X) * (1 - sigmoid_function(X)) * y_grad
        
#Loss 函数
class Loss():
    def __init__(self):
        self.mem = {}
        self.len = 0

    def forward(self, y_pred, y):
        self.mem["y_pred"] = y_pred
        self.len = np.size(y_pred,0)
        return np.mean(np.sum((y_pred-y)**2,axis=1))
    
    def backward(self,y):
        y_pred = self.mem["y_pred"]
        return -2 * (y-y_pred) / self.len
    
#全连接层
class FullConnectionLayer():
    def __init__(self):
        self.mem = {}       #保存前向传输过程的参数，输入，权重

    def forward(self,X,W):
        self.mem["X"] = X
        self.mem["W"] = W
        return np.matmul(X,W)
    
    def backward(self,H_grad):
        X = self.mem["X"]
        W = self.mem["W"]
        X_grad = np.matmul(H_grad,W.T)
        W_grad = np.matmul(X.T,H_grad)
        return X_grad,W_grad

class FullConnectionModel():
    def __init__(self, dims):            #dims 表示创建的隐藏层有几个节点
        self.dims = dims
        self.W1 = np.random.normal(loc=0,scale=1,size=[28 * 28 + 1,dims]) /np.sqrt((28 * 28 + 1)/2)    #加一层偏置
        self.W2 = np.random.normal(loc=0,scale=1,size=[dims + 1,10]) /np.sqrt((dims + 1) / 2)
        self.node_h1 = FullConnectionLayer()
        self.node_h2 = FullConnectionLayer()
        self.sigmoid1 = sigmoid()
        self.sigmoid2 = sigmoid()
        self.Loss = Loss()
        
    def forward(self, X, labels):
        bias1 = np.ones(shape=[X.shape[0],1])       #增加偏置
        X = np.concatenate([X, bias1], axis=1)
        self.h1 = self.node_h1.forward(X,self.W1)
        self.h1_out = self.sigmoid1.forward(self.h1)
        bias2 = np.ones(shape=[self.h1_out.shape[0],1])
        self.h2_in =np.concatenate([self.h1_out, bias2], axis=1) #增加第二层的偏置
        self.h2 = self.node_h2.forward(self.h2_in,self.W2)
        self.y_pred = self.sigmoid2.forward(self.h2)
        self.Lossp = self.Loss.forward(self.y_pred,labels)

    def backward(self, labels):
        self.grad_loss = self.Loss.backward(labels)
        self.grad_h2_sigmoid = self.sigmoid2.backward(self.grad_loss)
        self.grad_h2, self.grad_W2 = self.node_h2.backward(self.grad_h2_sigmoid)
        self.grad_h1_sigmoid_pre = self.grad_h2[:,0:self.dims] 
        self.grad_h1_sigmoid = self.sigmoid1.backward(self.grad_h1_sigmoid_pre)
        self.grad_h1, self.grad_W1 = self.node_h1.backward(self.grad_h1_sigmoid)


def computeAccuracy(prob, labels):   #数据集labels为长度为10的向量，其中有9个为0，一个为1，故比较最大值的位置
    predicitions = np.argmax(prob, axis=1)
    truth = np.argmax(labels, axis=1)
    return np.mean(predicitions == truth)


# 训练一次模型
def trainOneStep(model, x_train, y_train, learning_rate=1e-5):
    model.forward(x_train, y_train)
    model.backward(y_train)
    model.W1 += -learning_rate * model.grad_W1
    model.W2 += -learning_rate * model.grad_W2
    loss = model.Lossp
    accuracy = computeAccuracy(model.y_pred, y_train)
    return loss, accuracy


# 训练模型和寻优
def train(x_train, y_train, x_validation, y_validation):
    epochs = 1000
    learning_rate = 1e-1
    latent_dims_list = list(range(10,100,10))
    best_accuracy = 0
    best_latent_dims = 0

    # 在验证集上寻优
    print("Start seaching the best parameter...\n")
    for latent_dims in latent_dims_list:
        model = FullConnectionModel(latent_dims)

        bar = trange(20)  # 使用 tqdm 第三方库，调用 tqdm.std.trange 方法给循环加个进度条
        for epoch in bar:
            loss, accuracy = trainOneStep(model, x_train, y_train, learning_rate) 
            bar.set_description(f'Parameter latent_dims={latent_dims: <3}, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
        bar.close()

        validation_loss, validation_accuracy = evaluate(model, x_validation, y_validation)
        print(f"Parameter latent_dims={latent_dims: <3}, validation_loss={validation_loss}, validation_accuracy={validation_accuracy}.\n")

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_latent_dims = latent_dims

    # 得到最好的参数组合，训练最好的模型
    print(f"The best parameter is {best_latent_dims}.\n")
    print("Start training the best model...")
    best_model = FullConnectionModel(best_latent_dims)
    x = np.concatenate([x_train, x_validation], axis=0)
    y = np.concatenate([y_train, y_validation], axis=0)
    bar = trange(epochs)
    for epoch in bar:
        loss, accuracy = trainOneStep(best_model, x, y, learning_rate)
        bar.set_description(f'Training the best model, epoch={epoch + 1: <3}, loss={loss: <10.8}, accuracy={accuracy: <8.6}')  # 给进度条加个描述
    bar.close()

    return best_model


# 评估模型
def evaluate(model, x, y):
    model.forward(x, y)
    loss = model.Lossp
    accuracy = computeAccuracy(model.y_pred, y)
    return loss, accuracy


mnist = input_data.read_data_sets("dataset/", one_hot=True)
model = train(mnist.train.images, mnist.train.labels, mnist.validation.images, mnist.validation.labels)
loss, accuracy = evaluate(model, mnist.test.images, mnist.test.labels)
print(f'Evaluate the best model, test loss={loss:0<10.8}, accuracy={accuracy:0<8.6}.')
