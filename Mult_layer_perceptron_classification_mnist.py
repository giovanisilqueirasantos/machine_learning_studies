from utils.loadMNIST import load_mnist
import matplotlib.pyplot as plt
from models.NeuralNetMLP import NeuralNetMLP
import numpy as np

X_train, y_train = load_mnist()
X_test, y_test = load_mnist(kind='t10k')

fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

nn = NeuralNetMLP(n_output=10, n_features=X_train.shape[1], n_hidden=50, l1=0.0, l2=0.1,
                  epochs=1000, eta=0.001, alpha=0.001, decrease_const=0.00001, shuffle=True,
                  minibatches=50, random_state=1)

nn.fit(X_train, y_train, print_progress=True)

y_train_pred = nn.predict(X_train)
acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]
print(f'Training accuracy: {(acc * 100):.2f}')

y_test_pred = nn.predict(X_test)
acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]
print(f'Test accuracy: {(acc * 100):.2f}')

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
plt.show()

miscl_img = X_test[y_test != y_test_pred][:25]
correct_lab = y_test[y_test != y_test_pred][:25]
miscl_lab = y_test_pred[y_test != y_test_pred][:25]

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(25):
    img = miscl_img[i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[i].set_title(f'{i+1}) t: {correct_lab[i]} p: {miscl_lab[i]}')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()