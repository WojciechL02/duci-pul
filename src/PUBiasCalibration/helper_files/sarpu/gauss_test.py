# %%
import torch

n = 1000
n_pos = int(n/2)

X1_neg = torch.normal(-3, 1, size=(n_pos, 1))
X2_neg = torch.normal(0, 1, size=(n_pos, 1))
X_neg = torch.cat([X1_neg, X2_neg], axis=1)

X1_pos = torch.normal(3, 1, size=(n_pos, 1))
X2_pos = torch.normal(0, 1, size=(n_pos, 1))
X_pos = torch.cat([X1_pos, X2_pos], axis=1)

X = torch.cat([X_neg, X_pos], axis=0)
y = torch.cat([torch.zeros(n_pos), torch.ones(n_pos)], axis=0)

X, y

# %%
import torch
import os
import re
from scipy.io import arff
import numpy as np
import pandas as pd


dir_path = os.path.dirname(os.path.realpath(__file__))


def read_names_file(filename):
    with open(filename, 'r') as f:
        columns = []
        while True:
            s = f.readline()
            if s == '':
                break

            match = re.match(r'([^:]+):\s+[a-zA-Z]+\.', s)
            
            if match is not None:
                column_name = match.groups()[0]
                columns.append(column_name)
            
        return columns


def get_datasets():
    names = [
        'Adult',
        'BreastCancer',
        'credit-a',
        'credit-g',
        'diabetes',
        'heart-c',
        'spambase',
        'vote',
        'wdbc',
    ]

    return {name: load_dataset(name) for name in names}


def load_dataset(name):
    data = arff.loadarff(os.path.join(dir_path, 'data', f'{name}.arff'))
    df = pd.DataFrame(data[0])

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return X.to_numpy(), y.to_numpy()

X, y = load_dataset('diabetes')
#Obtain mean of columns as you need, nanmean is convenient.
col_mean = np.nanmean(X, axis=0)
#Find indices that you need to replace
inds = np.where(np.isnan(X))
#Place column means in the indices. Align the arrays using take
X[inds] = np.take(col_mean, inds[1])

X, y = torch.tensor(X), torch.tensor(y)

n = len(y)
X, y

# %%
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

# %%
from sklearn.linear_model import LogisticRegression
import numpy as np

# clf = LogisticRegression(tol=1e-3, max_iter=3)
clf = LogisticRegression()
clf.fit(X, y)

y_pred = clf.predict(X)
np.mean(y_pred == y.numpy())

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

plt.show()


# # %%
# from model import LogisticClassifier

# nn_clf = LogisticClassifier(2)
# criterion = torch.nn.BCELoss()

# optimizer = torch.optim.Adam(nn_clf.parameters())

# for epoch in range(100):
#     optimizer.zero_grad() # Setting our stored gradients equal to zero
#     outputs = nn_clf(X)
#     loss = criterion(torch.squeeze(outputs), y) # [200,1] -squeeze-> [200]

#     loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
#     optimizer.step() # Updates weights and biases with the optimizer (SGD)

# y_proba = nn_clf(X)
# y_pred = torch.where(y_proba >.5, 1, 0).squeeze()
# print(torch.sum(y_pred == y) / len(y))

# for param in nn_clf.named_parameters():
#     print(param)
    

# # %%
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Z = nn_clf(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)).squeeze() > 0.5
# Z = Z.reshape(xx.shape)

# plt.contourf(xx, yy, Z, alpha=0.4)
# plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

# plt.show()

# %%
def eta(x, lgr_param, intercept, kappa=10):
    return torch.pow(1 / (1 + torch.exp(-(x.double() @ lgr_param.T + intercept))), kappa)

kappa = 10

propensity = eta(
        X, 
        torch.tensor(clf.coef_, dtype=torch.double), 
        torch.tensor(clf.intercept_, dtype=torch.double),
        kappa = kappa,
    ).reshape(-1).double()
propensity[torch.where(y == 0)] = 0
propensity

# %%
def eta(x, power=10):
    propensity_dim = x[:, 1]
    propensity = (propensity_dim - propensity_dim.min()) / (propensity_dim.max() - propensity_dim.min())
    return torch.pow(propensity, power)

propensity = eta(
        X, 
        power = 4
    ).reshape(-1).double()
propensity[torch.where(y == 0)] = 0
propensity


# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Z = (eta(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), 
#         torch.tensor(clf.coef_, dtype=torch.double), 
#         torch.tensor(clf.intercept_, dtype=torch.double),
#         kappa = kappa))
Z = eta(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), power=4)
Z = Z.reshape(xx.shape).detach().numpy()

plt.figure(figsize=(8, 6))
contour = plt.contourf(xx, yy, Z, levels=np.linspace(0, 1, 11))
plt.colorbar(contour)
plt.savefig(f"true_eta_kappa_{kappa}.pdf")

plt.show()


# %%
# weights = propensity / propensity.sum()

# c = 0.4 * 0.5
# selected = np.random.choice(range(n), replace = False, size = int(c * n), p = weights)
# # selected
# s = torch.zeros_like(y)
# s[selected] = 1
# s

### V2

# s = torch.bernoulli(propensity)
# s = torch.where((s == 1) & (y == 1), 1, 0)
# s

### V3

c = 0.4
num_labeled = int(c * n_pos)
idx = propensity.multinomial(num_samples=num_labeled, replacement=True)
# idx = propensity.multinomial(num_samples=num_labeled, replacement=False)
s = torch.zeros_like(y)
s[idx] = 1

# %%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=s, s=20, edgecolor="k")
legend1 = plt.legend(*scatter.legend_elements(), title="Observed")

# # %%
# import torch.optim as optim

# from model import LBE
# from model import LBE_alternative
# lbe = LBE(2, kind="LF")
# # lbe = LBE(2, kind="MLP")
# # lbe = LBE_alternative(2)

# X = X.float()
# s = s.float()
# lbe.pre_train(X, s)

# s_pred = lbe.h(X)
# print(torch.sum((s_pred.squeeze() > 0.5) == s) / len(s))

# for param in lbe.h.named_parameters():
#     print(param)

# %%
from sarpu.pu_learning import pu_learn_sar_em

result = pu_learn_sar_em(X.numpy(), s.numpy(), range(X.shape[1]))
result

# %%
from sklearn import metrics

classifier = result[0]
propensity_model = result[0]
y_proba = propensity_model.predict_proba(X.numpy())
y_pred = np.where(y_proba > 0.5, 1, 0)

auc = metrics.roc_auc_score(y, y_proba)
acc = metrics.accuracy_score(y, y_pred)

f'ACC: {100 * acc:.2f}%, AUC: {100 * auc:.2f}%'

# %%
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

Z = (propensity_model.predict_proba(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")

plt.show()

# %%
from sarpu.experiments import evaluate_all

evaluate_all(y.numpy(), s.numpy(), propensity.numpy(), classifier.predict_proba(X.numpy()), propensity_model.predict_proba(X.numpy()))

# %%
