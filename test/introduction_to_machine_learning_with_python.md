---
sort: 4
---

## Regression Study
### Linear regression


```
from sklearn.linear_model import LinearRegression
```


```
!pip install mglearn
```

    Collecting mglearn
    [?25l  Downloading https://files.pythonhosted.org/packages/65/38/8aced26fce0b2ae82c3c87cd3b6105f38ca6d9d51704ecc44aa54473e6b9/mglearn-0.1.9.tar.gz (540kB)
    [K     |████████████████████████████████| 542kB 7.1MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from mglearn) (1.18.5)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.6/dist-packages (from mglearn) (3.2.2)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.22.2.post1)
    Requirement already satisfied: pandas in /usr/local/lib/python3.6/dist-packages (from mglearn) (1.0.5)
    Requirement already satisfied: pillow in /usr/local/lib/python3.6/dist-packages (from mglearn) (7.0.0)
    Requirement already satisfied: cycler in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.10.0)
    Requirement already satisfied: imageio in /usr/local/lib/python3.6/dist-packages (from mglearn) (2.4.1)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from mglearn) (0.16.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (2.4.7)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (1.2.0)
    Requirement already satisfied: python-dateutil>=2.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib->mglearn) (2.8.1)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->mglearn) (1.4.1)
    Requirement already satisfied: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas->mglearn) (2018.9)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from cycler->mglearn) (1.15.0)
    Building wheels for collected packages: mglearn
      Building wheel for mglearn (setup.py) ... [?25l[?25hdone
      Created wheel for mglearn: filename=mglearn-0.1.9-py2.py3-none-any.whl size=582639 sha256=8ca0814a4524c34a09786c1861ba4a99ad8a366bd0a054be5b0436e4d82235c6
      Stored in directory: /root/.cache/pip/wheels/eb/a6/ea/a6a3716233fa62fc561259b5cb1e28f79e9ff3592c0adac5f0
    Successfully built mglearn
    Installing collected packages: mglearn
    Successfully installed mglearn-0.1.9



```
import mglearn
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_wave(n_samples = 60)
```


```
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
```


```
print("lr.coef_: %.4f" %lr.coef_)
print("lr.intercept_: {}".format(lr.intercept_))
```

    lr.coef_: 0.3939
    lr.intercept_: -0.031804343026759746



```

```


```
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
```

    훈련 세트 점수: 0.67
    테스트 세트 점수: 0.66


위의 샘플 예제의 경우는 모델이 매우 단순하여 overfitting(과대적합)문제를 고민할 필요는 없습니다. 

Linear Regression 모델이 보스턴 주택가격 세이터셋 같은 복잡한 데이터셋에서 어떻게 동작하는 지 한번 살펴보겠습니다.


```
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lr = LinearRegression().fit(X_train, y_train)
```


```
print("훈련 세트 점수: {:.2f}".format(lr.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lr.score(X_test, y_test)))
```

    훈련 세트 점수: 0.95
    테스트 세트 점수: 0.61


이렇게 샘플 506개, 특성값이 104개를 쓰게되면, 과대적합문제가 발생하죠. 훈련과 테스트 세트의 점수를 보면 이를 유추할 수 있습니다.

### Ridge regression


릿지(Ridge) 회귀도 선형모델을 사용하므로 최소적합법(least square)에서 사용한 것과 같은 예측함수를 사용합니다. 하지만, 릿지(Ridge) 회귀에는 제약조건을 둘 수가 있습니다. W의 크기를 제약으로 두는 regularization 을 하는 것입니다.  릿지(Ridge) 회귀에서는 L2 regularization을 사용합니다.


```
from sklearn.linear_model import Ridge
```


```
ridge = Ridge().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge.score(X_test, y_test)))
```

    훈련 세트 점수: 0.89
    테스트 세트 점수: 0.75


결과가 Linear Regression(선형회귀) 모델에 비해서 테스트 세트 점수가 개선된 것을 볼 수 있고, 한편 훈련세트의 경우 성능이 낮아졌습니다. 선형회귀의 경우는 과대적합이 되었지만, 릿지회귀는 덜 자유로운 모델이기에 과대적합이 상대적으로 적어집니다. 우리의 초점은 테스트 세트이기 때문에 릿지 회귀를 사용하는 것이 더 바람직합니다.

**Q. 규제의 강도는 어떻게 조절할까요?** 

alpha 값을 이용하여 조정합니다. 위 예제에서는 alpha = 1.0이 사용되었으니, alpha값을 높이면 계수를 0에 더 가깝게 만들어서 훈련성능은 낮아지고, 일반화에는 도움을 줍니다.




```
ridge10 = Ridge(alpha=10).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge10.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge10.score(X_test, y_test)))
```

    훈련 세트 점수: 0.79
    테스트 세트 점수: 0.64



```
ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(ridge01.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(ridge01.score(X_test, y_test)))
```

    훈련 세트 점수: 0.93
    테스트 세트 점수: 0.77


여기에서는 alpha = 0.1이 좋은 성능을 내었습니다. 이렇게 parameter(매개변수)를 어떻게 설정하는 것이 좋은지에 대해서도 연구가 많이 있습니다. 이 내용은 다음에 다시 다뤄보겠습니다.


```
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.plot(ridge10.coef_, '^', label="Ridge a10")
plt.plot(ridge.coef_, '^', label="Ridge a1")
plt.plot(ridge01.coef_, '^', label="Ridge a0.1")

plt.plot(lr.coef_, 'o', label="LinearRegression")
plt.xlabel("계수 목록")
plt.ylabel("계수 크기")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()
plt.show()
```


![png](introduction_to_machine_learning_with_python_files/introduction_to_machine_learning_with_python_20_0.png)



```
mglearn.plots.plot_ridge_n_samples()
```


![png](introduction_to_machine_learning_with_python_files/introduction_to_machine_learning_with_python_21_0.png)


### Lasso

선형회귀에 규제를 적용하는데 Ridge의 대한으로 Lasso가 있습니다. 규제의 방식에서 차이가 있는데, L1 regularization(규제)를 사용합니다. 이 규제의 특징은 규제의 결과로 특정 계수는 0으로 만들게 됩니다. 이 말은 해당 특성은 완전히 제외하고 보겠다는 의미이죠. 특성의 선택까지 자동으로 이뤄진다고 보면 되겠습니다. 


```
from sklearn.linear_model import Lasso
import numpy as np
```


```
lasso = Lasso().fit(X_train, y_train)
print("훈련 세트 점수: {:.2f}".format(lasso.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso.coef_ !=0)))
```

    훈련 세트 점수: 0.29
    테스트 세트 점수: 0.21
    사용한 특성의 수: 4


코드에서 본 것처럼, 104개의 특성 중 단 4개만 사용한 것을 볼 수 있습니다. Lasso역시 계수를 얼마나 강하게 0으로 보낼 지 조절하는 alpha값이 있으며, underfitting(과소적합)을 줄이기 위해서 alpha값을 줄여보는 것이 필요합니다.


```
lasso001 = lasso = Lasso(alpha = 0.01, max_iter=100000).fit(X_train, y_train)
```


```
print("훈련 세트 점수: {:.2f}".format(lasso001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso001.coef_ !=0)))
```

    훈련 세트 점수: 0.90
    테스트 세트 점수: 0.77
    사용한 특성의 수: 33



```
lasso00001 = lasso = Lasso(alpha = 0.0001, max_iter=100000).fit(X_train, y_train)

print("훈련 세트 점수: {:.2f}".format(lasso00001.score(X_train, y_train)))
print("테스트 세트 점수: {:.2f}".format(lasso00001.score(X_test, y_test)))
print("사용한 특성의 수: {}".format(np.sum(lasso00001.coef_ !=0)))
```

    훈련 세트 점수: 0.95
    테스트 세트 점수: 0.64
    사용한 특성의 수: 96



```
plt.plot(lasso.coef_, 's', label="Ridge a1")
plt.plot(lasso001.coef_, '^', label="Ridge a0.01")
plt.plot(lasso00001.coef_, 'v', label="Ridge a0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge a0.01")
plt.legend(ncol=2, loc=(0, 1.05))
```




    <matplotlib.legend.Legend at 0x7fba0b65bb38>




![png](introduction_to_machine_learning_with_python_files/introduction_to_machine_learning_with_python_30_1.png)


위 그림에서 alpha의 크기에 따른 변화를 볼 수 있습니다. alpha = 1일 때, 계수 대부분이 0일 뿐만 아니라 나머지 계수들도 크기가 작다는 것을 볼 수 있습니다. alpha가 작아지면, 규제를 더 적게 받는 것이고, 0.00001의 alpha값을 주면 거의 규제를 받지 않는 것을 확인할 수 있었습니다. alpha=0.01인 Ridge와 lasso는 성능이 비슷하지만, Ridge의 경우는 어떤 계수도 0이 되지 않는 다는 점!

두 규제방식의 패널티를 결합한 형태의 ElasticNet도 있으니, 이 것을 이용하면 최상의 성능을 낼 수 있을 것이고, 한편 2가지의 parameter를 조절해주어야 합니다. L1, L2 각각의 alpha값을요!


```

```


```

```


```

```


```

```


```

```
