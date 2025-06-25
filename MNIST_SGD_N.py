import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
from scipy.special import softmax

#活性化関数の定義（ラムダ式）
relu_fn = lambda Z: np.maximum(Z,0)  #ReLU関数      
drelu_fn = lambda Z: np.array(Z > 0, dtype=int)  #ReLU関数の微分（勾配算出時に使用）
softmax_fn = lambda Z: softmax(Z,axis=1)  #ソフトマックス関数
dummy_grad = lambda Z: 1  #ソフトマックス関数の微分は使用しない為，仮の関数を定義

#Layerクラス（ニューラルネットワークの層1つ分）
class Layer:
  def __init__(self, prev_layer_size, layer_size, activation_fn, dactivation_fn):
    np.random.seed()
    self.W = np.random.randn(prev_layer_size, layer_size) / np.sqrt(prev_layer_size)
    self.b = np.zeros(layer_size)
    self.activation_fn = activation_fn
    self.dactivation_fn = dactivation_fn

  def forward(self, A_prev):
    self._input = A_prev
    self.Z = np.dot(A_prev, self.W) + self.b
    return self.activation_fn(self.Z)

#誤差逆伝播法
  def backward(self, delta, W_next):
    return np.dot(delta, W_next.T) * self.dactivation_fn(self.Z)

#各パラメータ（重み，バイアス）の勾配算出
  def grad(self, delta):
    dW = np.dot(self._input.T, delta)
    db = np.sum(delta, axis=0)
    return dW, db

#SimpleClassifierクラス（ニューラルネットワーク（多層パーセプトロン）全体）
class SimpleClassifier:
  def __init__(self,input_layer_size,output_layer_size,
               hidden_layers_sizes,activation_fn=relu_fn):
    layer_sizes = [input_layer_size, *hidden_layers_sizes]
    self.layers = (
        [Layer(layer_sizes[i],layer_sizes[i+1],activation_fn,drelu_fn)
        for i in range(len(layer_sizes)-1)])
    output_layer = Layer(layer_sizes[-1],output_layer_size, softmax_fn,dummy_grad)
    self.layers.append(output_layer)

#層ごとの出力
  def forward(self,A0):
    A = A0
    for layer in self.layers:
      A = layer.forward(A)
    Y_hat = A
    return Y_hat

#出力層における出力
  def predict(self,X):
    Y_hat = self.forward(X)
    return Y_hat

#精度算出
  def evaluate_accuracy(self,X,Y):
    predict = np.argmax(self.predict(X),axis=1)
    actual = np.argmax(Y, axis=1)
    num_corrects = len(np.where(predict==actual)[0])
    accuracy = num_corrects / len(X)
    return accuracy
  

(x_train, y_train), (x_test, y_test) = mnist.load_data()  #MNISTデータのロード
x_train, x_test = x_train/255.0, x_test/255.0  #画像データの正規化

x_train_flat,x_test_flat = x_train.reshape(-1,28*28), x_test.reshape(-1,28*28)  #画像データのフラット化（入力できる形に変形）
num_classes=10  #分類するクラス数
y_train_ohe = np.eye(num_classes)[y_train]  #学習データのOne-Hotエンコーディング
y_test_ohe = np.eye(num_classes)[y_test]  #テストデータのOne-Hotエンコーディング

mnist_classifier = SimpleClassifier(x_test_flat.shape[1],num_classes,[64,32])  #ニューラルネットワークの実装（隠れ層は2つ，ノード数はそれぞれ64，32）

#パラメータを1回更新する関数
def train_step(x, t):
  y = mnist_classifier.predict(x)
  y = np.clip(y, 1e-10, 1 - 1e-10)
  for i, layer in enumerate(mnist_classifier.layers[::-1]):
    if i == 0:
      delta = y - t
    else:
      delta = layer.backward(delta, W)
    dW ,db = layer.grad(delta)
    layer.W = layer.W - 0.0001 * dW
    layer.b = layer.b - 0.0001 * db

    W = layer.W

epochs = 20

#SGDによる学習
for epoch in range(epochs):
    indices = np.arange(x_train_flat.shape[0])
    np.random.shuffle(indices)
    x_train_flat_shuffled = x_train_flat[indices]
    y_train_ohe_shuffled = y_train_ohe[indices]

    for j in range(x_train_flat_shuffled.shape[0]):
        x = x_train_flat_shuffled[j:j+1]
        t = y_train_ohe_shuffled[j:j+1]
        train_step(x, t)

    print(f"Epoch {epoch+1} 完了")  #学習進捗の確認用
    #重みが0になっていないかの確認用
    for i, layer in enumerate(mnist_classifier.layers):
      if np.isnan(layer.W).any():
        print(f"Layer {i} の重みに nan が含まれています")

#N番目の分類確認
N = 1
y_hat = mnist_classifier.predict(x_test_flat[N-1:N])
print("推論結果：\n" , y_hat[0])
print("分類結果：\n" , np.argmax(y_hat[0]))

#全テストデータの分類精度
accuracy = mnist_classifier.evaluate_accuracy(x_test_flat,y_test_ohe)
print("正解率：{:.2f}%".format(accuracy*100))