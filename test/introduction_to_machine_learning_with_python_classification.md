---
sort : 5
---
## Classification

### Decision tree

결정트리는 결정에 이르기 위해서 예/아니오 질문을 연속적으로 이어 나가면서 학습합니다. 결정트리를 학습한다는 것은 정답에 가장 빨리 도달하는 예/아니오 질문 목록을 학습한다는 뜻입니다.


```
!pip install mglearn
import mglearn
mglearn.plots.plot_animal_tree()
```

    Requirement already satisfied: mglearn in /usr/local/lib/python3.6/dist-packages (0.1.9)
    Requirement already satisfied: cycler in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.10.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from mglearn) (3.2.2)
    Requirement already satisfied: imageio in /usr/local/lib/python3.6/dist-packages (from mglearn) (2.4.1)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from mglearn) (7.0.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from mglearn) (1.18.5)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.16.0)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from mglearn) (1.0.5)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.22.2.post1)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from cycler->mglearn) (1.15.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (2.8.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (1.2.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (2.4.7)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->mglearn) (2018.9)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->mglearn) (1.4.1)



![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_3_1.png)


결정트리에서도 일반적인 머신러닝에서와 마찬가지로 과대적합문제를 어떻게 해결 할 것인지의 문제가 있고, 2가지 해결 전략이 있습니다. 트리 생성을 일찍 중단하는 전략으로 사전-가지치기(pre-puning)과 트리를 만든 후 데이터 포인트가 적은 노드를 삭제하는 사후가지치기(post-pruning or pruning) 전략입니다.


```
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
```


```
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(tree.score(X_test, y_test)))
```

    훈련 세트 점수: 1.000
    테스트 세트 점수: 0.937



```
tree = DecisionTreeClassifier(max_depth = 4, random_state=0)
tree.fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(tree.score(X_test, y_test)))
```

    훈련 세트 점수: 0.988
    테스트 세트 점수: 0.951


결정 트리 분석


```
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file="tree.dot", class_names=["악성", "양성"], 
                feature_names=cancer.feature_names, 
                impurity=False, filled=True)
```


```
import graphviz

with open("tree.dot") as f:
  dot_graph = f.read()
display(graphviz.Source(dot_graph))
```


![svg](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_10_0.svg)


그래프트리를 시각화 하면, 알고리즘을 이해할 수 있으며, 비전문가에 알고리즘을 설명하기에 좋습니다. 여기서 보듯 깊이가 4만 되어도 장활해집니다. 트리가 더 깊어지면, 보통 10정도까지는 사용하는데, 한 눈에 보기가 어려워 집니다.  따라서 데이터의 흐름을 잘 보는 것이 중요합니다. 

전체 트리를 살펴보는 것이 어려울 수 있으니, 요약하는 속성을 사용하여 보겠습니다. 각 트리를 결정하는데 각 특성이 얼마나 쓰였는지 평가하는 특성 중요도(feature importance)를 이용합니다. 이 값은 0과 1 사이이의 숫자로 0은 전혀 사용하지 않음을, 1은 완벽하게 타깃을 예측했다는 뜻입니다.


```
print("feature importance: \n{}".format(tree.feature_importances_))
```

    feature importance: 
    [0.         0.         0.         0.         0.         0.
     0.         0.         0.         0.         0.01019737 0.04839825
     0.         0.         0.0024156  0.         0.         0.
     0.         0.         0.72682851 0.0458159  0.         0.
     0.0141577  0.         0.018188   0.1221132  0.01188548 0.        ]



```
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

```


```
def plot_feature_importance_cancer(model):
  n_features = cancer.data.shape[1]
  plt.barh(range(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), cancer.feature_names)
  plt.xlabel("feature_importance")
  plt.ylabel("Feature")
  plt.ylim(-1, n_features)
plot_feature_importance_cancer(tree)
```


![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_15_0.png)


이렇게 첫 노드에서 사용되었던 Worst radius가 가장 중요한 특징으로 나타납니다. 왜 첫번째 노드에서 이 특성을 사용했는지 뒷받침을 해주고 있으며, 특성의 중요도가 낮다고 해서 유용하지 않은 것은 아닙니다. 


```
tree = mglearn.plots.plot_tree_not_monotone()
display(tree)
```

    Feature importances: [0. 1.]



![svg](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_17_1.svg)



![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_17_2.png)



```
import pandas as pd
import os
```


```
ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "ram_price.csv"))

plt.semilogy(ram_prices.date, ram_prices.price)
plt.xlabel('years')
plt.ylabel('price ($/Mbyte)')
```




    Text(0, 0.5, 'price ($/Mbyte)')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_19_1.png)


위 그래프에서 y축은 log scale입니다. 약간의 굴곡을 제외하고는 선형적으로 나타나기에 비교적 예측이 쉬워집니다. 

여기서는 DesicisionTreeRegressor와 LinearRegression을 비교해보겠습니다. 


```
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
```


```
data_train = ram_prices[ram_prices.date < 2000]
data_test = ram_prices[ram_prices.date >= 2000]

# 가격 예측을 위해 날짜만 사용
X_train = data_train.date[:, np.newaxis]
y_train = np.log(data_train.price)

tree = DecisionTreeRegressor().fit(X_train, y_train)
linear_reg = LinearRegression().fit(X_train, y_train)
```


```
X_all = ram_prices.date[:, np.newaxis]

pred_tree = tree.predict(X_all)
pred_lr = linear_reg.predict(X_all)

price_tree = np.exp(pred_tree)
price_lr = np.exp(pred_lr)
```


```
plt.semilogy(data_train.date, data_train.price, label = "Training")
plt.semilogy(data_test.date, data_test.price, label = "Test")
plt.semilogy(ram_prices.date, price_tree, label = 'tree prediction')
plt.semilogy(ram_prices.date, price_lr, label = 'lr prediction')
plt.legend()
```




    <matplotlib.legend.Legend at 0x7f6a2b57be80>




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_24_1.png)


두 모델의 차이는 확연하죠. Linear모델은 우리는 아는대로 직선으로 근사하며, 미세한 굴곡은 매끈하게 만들어버리죠. 2000년 이후에도 예측력이 좋습니다. 그러나 Tree모델의 경우는 훈련 데이터는 완벽하게 예측을 하지만, 범위 밖으로 나가면 마지막 포인트를 이용해 예측하는 것이 전부입니다. 트리 모델은 훈련 데이터 범위 밖의 데이터를 예측할 수 있는 능력이 없습니다. 

장단점과 매개변수

앞에서 설명한 것처럼 결정 트리에서 모델 복잡도를 조절하는 매개변수는 트리가 완전히 만들어지기 전에 멈추게 하는 사전 가지치기 변수입니다. 보통 max_depth, max_leaf_nodes/min_sample_leaf 중 하나만 지정해도 overfitting을 막을 수 있습니다. 

결정 트리가 다른 알고리즘에 비해 나은 점은 2가지 입니다. 
1) 만들어진 모델을 쉽게 시각화 할 수 있어서 이해하기 쉽다.
2) 데이터의 스케일에 구애받지 않는다. 
정규화나 표준화 처리를 하지 않아도 된다는 장점이 있죠. 특히 특성의 스케일이 서로 다르거나 이진 특성과 연속적인 특성이 혼합되어 있을 때에도 잘 동작합니다. 

그럼에도 단점은 가지치기를 사용해도 과대적합문제가 있습니다. 일반화에서 성능이 좋지 않습니다. 그래서 다음에 설명할 앙상블 방법을 단일 결정 트리의 대안으로 흔히 사용합니다. 

### Decision Tree ensemble

앙상블(Ensemble)은 머신러닝/딥러닝 모델을 서로 연결하여 더 강력한 모델을 만드는 기법입니다. 다양한 데이터셋에 효과적으로 알려진 Random Forest와 Gradient Boosting 결정트리가 있고, 이들은 모델을 구성하는 기본 요소로 결정트리를 사용합니다. 

#### Random Forest


```
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons
```


```
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                           criterion='gini', max_depth=None, max_features='auto',
                           max_leaf_nodes=None, max_samples=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=5,
                           n_jobs=None, oob_score=False, random_state=2, verbose=0,
                           warm_start=False)




```
fig, axes = plt.subplots(2, 3, figsize=(20,10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
  ax.set_title("Tree {}".format(i))
  mglearn.plots.plot_tree_partition(X, y, tree, ax=ax)

mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mglearn.discrete_scatter(X[:,0], X[:,1], y)
```




    [<matplotlib.lines.Line2D at 0x7f6a28d2dc88>,
     <matplotlib.lines.Line2D at 0x7f6a28c9b080>]




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_33_1.png)


유방암 데이터 예


```
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)
```


```
forest = RandomForestClassifier(n_estimators=500, random_state=0)
forest.fit(X_train, y_train)

print("훈련 세트 점수: {:.3f}".format(forest.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(forest.score(X_test, y_test)))
```

    훈련 세트 점수: 1.000
    테스트 세트 점수: 0.944


RF는 매개변수 튜닝 없이도 선형 모델이나 단일 결정 트리에 비해서 높은 테스트 세트 점수를 얻었습니다. 단일 결정트리에서 한 것처럼 max_features 매개변수를 조정하거나 사전 가지치지를 할 수도 있지만, 기본 설정으로도 좋은 결과를 만들어 줄 때가 많이 있습니다. 


```
plot_feature_importance_cancer(forest)
```


![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_38_0.png)


다른 예시에서 만들어 보았던 feature importance를 살펴보면, 단일 트리에 비해서 많은 특성이 0 이상의 중요도를 나타냅니다. RF의 무작위성이 더 넓은 시각으로 데이터를 바라보고 있다는 것을 할 수 있습니다. 

장단점과 매개변수

회귀와 분류에 있어서 랜덤 포레스트(RF)는 현재 가장 널리 사용되는 머신러닝 알고리즘입니다. 성능이 좋고, 튜닝을 많이 하지 않아도 잘 동작하며 데이터의 스케일을 맞출 필요도 없기 때문이죠. 

기본적으로 RF는 단일트리의 단점을 보완하고, 장점을 그대로 가지고 있습니다. 비전문가에게 쉽게 보여주기 위해서는 단일트리를 이용하여 보여주는 것이 효과적일 것입니다. RF model을 만드는데 다소 시간이 걸릴 수 있지만, n_jobs 매개변수를 이용해서 multi-core를 이용하면 보다 따르게 학습할 수 있습니다.  n_jobs = -1 를 사용하면 모든 코어를 사용하게 됩니다. 

유념할 점은 이름에서부터 Random이 들어있기 때문에 텍스트 데이터 같이 매우 차원이 높은 데이터에서는 잘 동작하지 않습니다. 

중요 매개변수는 n_estimators, max_features이고, max_depth같은 사전 가지칭기 옵션이 있습니다. n_estimators는 클수록 좋습니다. 더 많은 트리를 이용하여 평균하면 과대적합을 줄일 수 있어 안정적인 모델을 만들게 됩니다. 

#### GradientBoosting


```
from sklearn.ensemble import GradientBoostingClassifier
```


```
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(gbrt.score(X_test, y_test)))

```

    훈련 세트 점수: 1.000
    테스트 세트 점수: 0.958


훈련이 100% 이면 과대적합일 확율이 높죠. 가지치기를 해보죠


```
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(gbrt.score(X_test, y_test)))
```

    훈련 세트 점수: 0.995
    테스트 세트 점수: 0.965



```
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.05)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(gbrt.score(X_test, y_test)))
```

    훈련 세트 점수: 1.000
    테스트 세트 점수: 0.958



```
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1, learning_rate=0.1)
gbrt.fit(X_train, y_train)

print("훈련 세트 점수: {:.3f}".format(gbrt.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(gbrt.score(X_test, y_test)))
```

    훈련 세트 점수: 0.995
    테스트 세트 점수: 0.965



```
plot_feature_importance_cancer(gbrt)
```


![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_48_0.png)


RF와 달리 GB은 일부 특성은 완전히 무시하고 있습니다. 
비슷한 종류의 데이터에서 둘 다 작동을 잘 하지만, 보통 조금 더 안정적인 RF모델을 사용합니다. 

대규모 머신러닝 문제에서는 GB을 적용하려면, xgboost 패키지와 python 인터페이스를 검토하는 것이 좋습니다. 2017년 기준으로 여러 데이터셋에서 scikit-learn보다 더 빨랐습니다. 그리고 튜닝하기가 조금 더 쉽습니다.

장단점과 매개변수

GB 결정 트리는 지도학습(supervised learning)에서 가장 강력하고 널리 사용되는 모델 중 하나입니다. 가장 큰 단점은 매개변수를 잘 조정해야 한다는 것과 훈련시간이 길다는 것이죠.  다른 트리 모델처럼 특성의 스케일 조정은 필요하지 않고, 이진 특성이 연속적인 특성에서도 잘 동작합니다. 트리의 특성 상 고차원 데이터에는 잘 작동하지 않습니다. 

GB의 매개변수는 n_esimators와 learning_rate이 있습니다. 
RF와 다르게 n_esimators를 너무 크게하면 overfitting 문제가 있을 수 있습니다. + (max_depth or max_leaf_nodes)를 조절할 수 있습니다. 통상적으로 5보다 깊어지지 않도록 합니다. 


```

```

### Kernel SVM

SVM에 대해서는 많이 들어보셨을 것이고, 또 많이 사용해보셨죠?
이번에 다룰 것은 kernel SVM입니다. 사실 SVM을 써보셨으면, kernel SVM을 써보셨을거에요... param. 중에 linear를 사용하지 않으셨다면요!

선형 모델과 비선형 모델의 특성


```
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8)
y = y %2
```


```
mglearn.discrete_scatter(X[:,0], X[:,1], y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
```




    Text(0, 0.5, 'Feature 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_56_1.png)


위 그림에서와 같이 선형적인 특징으로 구분할 수 없는 경우에 사용하게 됩니다. 


```
from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
```




    Text(0, 0.5, 'Feature 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_58_1.png)


이 두 특성을 이용해서 제 3의 특성을 만들어보겠습니다. 


```
X_new = np.hstack([X, X[:, 1:]**2])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()

ax = Axes3D(figure, elev=-152, azim=-26)
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 1 ** 2")
```




    Text(0.5, 0, 'Feature 1 ** 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_60_1.png)


이렇게 생긴 데이터셋에서는 선형 모델과 3차원 공간의 평면을 사용해 두 클래스를 구분할 수 있습니다. 


```
linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:,0].min() - 2, X_new[:,0].max() +2, 50)
yy = np.linspace(X_new[:,1].min() - 2, X_new[:,1].max() +2, 50)

XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
           cmap=mglearn.cm2, s=60, edgecolor='k')
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_zlabel("Feature 1 ** 2")
```




    Text(0.5, 0, 'Feature 1 ** 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_62_1.png)


원래 특성으로 투영해보면 이 선형 SVM은 더 이상 선형이 아니죠. 직선보다 타월에 가까움 모습을 확인할 수 있습니다. 


```
ZZ = YY **2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], 
             cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
```




    Text(0, 0.5, 'Feature 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_64_1.png)


Kernel Trick

앞서 데이터 셋의 변화를 주면서, 선형 모델을 강력하게 만들었습니다. 하지만 많은 경우 어떻게 확장을 해야할 지 모르고, 연산비용 역시 커집니다. 다행히 수학적 기교인 kernel trick을 이용하여 실제 데이터를 확장하지 않고 확장된 특성에 대한 데이터 포인트들의 거리를 계산합니다. 

서포트 벡터 머신에서 데이터를 고차원 공간에 매핑하는데 많이 사용하는 방법은 2가지 입니다. RBF(Gaussian) 커널과 다항식(polynomial) 커널이 있습니다. 

SVM 이해하기 \n
학습이 진행되는 동안 SVM은 각 훈련 데이터 포인트가 두 클래스 사이의 결정 경계를 구분하는데 얼마나 중요한지를 배우게 됩니다. 바로 두 클래스 사이의 경계에 위치한 데이터 포인트들을 서포트 벡터라고 칭하며, 이들에 의해서 학습이 되는데요, 여기서 SVM의 이름이 유래됩니다. 

새로운 데이터 포인트에 대해 예측하려면 각 서포트 벡터와의 거리를 측정합니다. 분류 결정은 서포트 벡터까지의 거리에 기반하며, 각 벡터의 중요도는 SVC.dual_coef_ 속성에 저장됩니다. 

K_rbf = exp(-gamma|x1 - x2|^2)


```
from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel = 'rbf', C=10, gamma=0.1).fit(X,y)

mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:,0], X[:,1], y)

sv = svm.support_vectors_
sv_labels = svm.dual_coef_.ravel() >0
mglearn.discrete_scatter(sv[:,0], sv[:,1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
```




    Text(0, 0.5, 'Feature 2')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_67_1.png)


매개변수 튜닝

gamma 매개변수는 카우시안 커널 폭의 역수에 해당합니다. 훈련샘플이 미치는 범위를 가우시안으로 생각하고 결정한다고 생각하면 되고, 작은 값이 넓은 영역을, 큰 값이면 영향을 미치는 영역이 좁아집니다. 

C 매개변수는 선형 모델에서 사용한 것과 비슷한 규제 매개변수입니다. 각 포인트의 중요도(dual_coef_) 값을 제한합니다. 


```
fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for ax, C in zip(axes, range(-1, 2)):
  for a, gamma in zip(ax, range(-1, 2)):
    mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)

axes[0, 0].legend(["클래스 0", "클래스 1", "클래스 0 서포트벡터", "클래스 1 서포트벡터"], 
                   ncol=4, loc = (.9, 1.2))
```




    <matplotlib.legend.Legend at 0x7f6a25cb5e10>




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_69_1.png)


왼쪽에서 오른쪽으로 가면서 gamma가 커집니다. 가우시안의 범위가 좁아지는 것을 의미하죠. 반경이 점점 줄어드는 것을 볼 수 있습니다. 
결정 경계가 한 포인트에 더욱 민감해집니다. 왼쪽으로 갈 수록 경계가 부드러워지죠.

한편 C는 아래로갈수록 커집니다. 제약이 매우커지게 되는 것이므로, 위쪽은 잘못 분류된 데이터 포인트가 경계에 거의 영향을 주지 않습니다. 반면 아래쪽은 결정 경계를 매우 강하게 휘게 합니다. 

유방암 데이터셋에 적용


```
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=0)

svc = SVC()
svc.fit(X_train, y_train)
print("훈련 세트 점수: {:.3f}".format(svc.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(svc.score(X_test, y_test)))

```

    훈련 세트 점수: 0.923
    테스트 세트 점수: 0.916



```
plt.boxplot(X_train, manage_ticks=False)
plt.yscale("symlog")
plt.xlabel("Feature index")
plt.ylabel("Feature size")
```




    Text(0, 0.5, 'Feature size')




![png](introduction_to_machine_learning_with_python_classification_files/introduction_to_machine_learning_with_python_classification_73_1.png)


그래프를 보니 유방암 데이터셋의 특성은 자릿수 자체가 완전히 다릅니다. 이것이 일부 모델(선형 모델 등)에서도 어느정도 문제가 될 수 있지만, 커널 SVM에서는 영향이 아주 큽니다. 이 문제를 해결하는 방법을 알아봅시다. 

SVM을 위한 데이터 전처리

데이터의 특징을 [0, 1]로 줄이는 방법입니다. 
minmaxScaler를 보통 많이 사용합니다. 


```
X_train[0:2][0:2]
```




    array([[1.231e+01, 1.652e+01, 7.919e+01, 4.709e+02, 9.172e-02, 6.829e-02,
            3.372e-02, 2.272e-02, 1.720e-01, 5.914e-02, 2.505e-01, 1.025e+00,
            1.740e+00, 1.968e+01, 4.854e-03, 1.819e-02, 1.826e-02, 7.965e-03,
            1.386e-02, 2.304e-03, 1.411e+01, 2.321e+01, 8.971e+01, 6.111e+02,
            1.176e-01, 1.843e-01, 1.703e-01, 8.660e-02, 2.618e-01, 7.609e-02],
           [1.754e+01, 1.932e+01, 1.151e+02, 9.516e+02, 8.968e-02, 1.198e-01,
            1.036e-01, 7.488e-02, 1.506e-01, 5.491e-02, 3.971e-01, 8.282e-01,
            3.088e+00, 4.073e+01, 6.090e-03, 2.569e-02, 2.713e-02, 1.345e-02,
            1.594e-02, 2.658e-03, 2.042e+01, 2.584e+01, 1.395e+02, 1.239e+03,
            1.381e-01, 3.420e-01, 3.508e-01, 1.939e-01, 2.928e-01, 7.867e-02]])




```
min_on_training = X_train.min(axis=0)
range_on_training = X_train.max(axis=0) - min_on_training

X_train_scaled = (X_train - min_on_training) / range_on_training

print("Feature min value \n{}".format(X_train_scaled.min(axis=0)))
print("Feature max value \n{}".format(X_train_scaled.max(axis=0)))
```

    Feature min value 
    [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0. 0. 0. 0. 0. 0.]
    Feature max value 
    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1.]



```
svc_sc = SVC()
svc_sc.fit(X_train_scaled, y_train)
X_test_scaled = (X_test - min_on_training) / range_on_training

print("훈련 세트 점수: {:.3f}".format(svc_sc.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(svc_sc.score(X_test, y_test)))
```

    훈련 세트 점수: 0.373
    테스트 세트 점수: 0.371


이건 왜 학습이 제대로 안 이뤄질까...


```
svc = SVC(C=1000, gamma=0.01)
svc.fit(X_train_scaled, y_train)
X_test_scaled = (X_test - min_on_training) / range_on_training

print("훈련 세트 점수: {:.3f}".format(svc.score(X_train, y_train)))
print("테스트 세트 점수: {:.3f}".format(svc.score(X_test, y_test)))
```

    훈련 세트 점수: 0.627
    테스트 세트 점수: 0.629


장단점과 매개변수

커널 SVM은 강력한 모델이며 다양한 데이터셋에서 잘 작동합니다.  SVM은 데이터의 특성이 몇 개 안되더라도 복잡한 결정 경계를 만들 수 있습니다. 차원의 수에는 상관이 없지만, 데이터 샘플 수가 많을 때에는 잘 맞지 않습니다. 

1만개의 샘플까지는 잘 되더라도, 10만개의 샘플이 되면 메모리 관점에서 도전적인 과제가 됩니다. 

한편, svm의 단점은 데이터 전처리와, 매개변수 설정에 신경을 많이 써야 한다는 점입니다. 분석도 어렵습니다. 하지만 모든 특성값들이 비슷한 단위이고, 스케일이 비슷하다면 시도해볼 가치는 있습니다. 

규제 매개변수 C가 중요하고, 어떤 커널을 사용할 지와 각 커널에 따른 매개변수가 있습니다. scikit-learn에서는 RBF의 경우 gamma 하나만 갖습니다. 두 변수가 복잡도를 조정하며, 둘 다 큰 값이 더 복잡한 모델을 만들게 됩니다. 


