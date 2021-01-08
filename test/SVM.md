---
sort: 7
---

# SVM

사전과제. 부록E. 라그랑주 승수법에 대해서 공부해본다. 

## 7.1 최대 마진 분류기

2 class 분류문제를 푸는 것에서 시작한다.

<div align="center">
$y(x) = w^T \phi(x) + b$
</div>
여기서 $\phi(x)$는 고정된 특징 공간 변환(fixed feature-space transformation)을 지칭한다. $b$는 바이어스 파라미터이다. 커널함수를 바탕으로 듀얼 표현(dual representation)법을 도입할 것인데, 이를 이용하여 특징공간을 직접 사용하지 않는다. $y(x)의$ target은 {-1. 1}이다.

가정 1. 데이터 집합이 특징 공간상에서 선형적으로 분류가 가능하다. <br>
$y(x_n) > 0$ 인 모든 포인트에서 $t_n = 1$이고,  <br>
$y(x_n) < 0$ 인 모든 포인트에서 $t_n = -1$이다. <br>
각 상황을 만족하는 매개변수 $w, b$가 최소한 하나는 존재한다. <br>
이 경우, 모둔 훈련 데이터 포인트에서 $t_n y(x_x) > 0$ 이 만족된다.

만약 훈련 집합을 정확하게 분류하는 해가 여러가지가 존재한다면, 가장 작은 일반화 오류 값을 주는 하나의 해를 찾아내는 것이 바람직할 것이다. SVM에서는 margin의 개념을 바탕으로 해당 문제를 접근한다. 여기서 마진은 결정 경계와 표본 사이의 최소 거리를 의미한다. 

<그림7.1>

---


이 마진이 최대화 되는 경계를 선택하며, 최대 마진을 구하기 위해서 계산적 학습 이론(=통계적 학습 이론, Statistical learning Theory)를 바탕으로 접근해야 한다고 생각할 수 있지만, $Tong \space and \space Koller (2000)$에서 제안한 hybrid of generative and discriminative approaches 를 사용한 방식으로 계산을 해볼 것이다. 

각각의 클래스에 대한 입력 벡터 x의 분포를 파젠 밀도 추정기(Parzen density estimator with Gaussian kernels with $\sigma^2$)로 모델을 했다. 결정 경계를 사용하기보다는 초평면(the best hyperplane) 형태로 해결을 할 것이다. $lim \space \sigma \rightarrow 0$ 의 경우 이 초평면이 최대 마진을 가지는 평면이 된다. <br>
** std가 감소됨에 따라서 멀리 있는 데이터가 아닌 가까이 있는 데이터들에 의해서만 초평면이 결정되고, 서포터 벡터 이외의 데이터 포인트들로부터 독립적이 된다. 

초평면으로부터 포인트 x까지 수직 거리는 $|y(x)| \over ||w||$ 로 주어지게 된다.  <br>
<img src="https://lh3.googleusercontent.com/proxy/ls4sObvKVqnqdA5mT_X-AFkk-PRoiasCVS2eSfhXPN9lGvLOUYcJkDYugCzIo2PfS-MPGKlZ-K0BV7--SrFOb_sfuctlbhE2biMUkiYDgMihnpgumr3uzahLEr96fDGn7pvm6Yj4j6J_77AO8Jwplg">

한편, $t_n y(x_x) > 0$을 만족하기 때문에, <br>

<div align="center">
$\Large{t_n|y(x)| \over ||w||}$ $\Large=$ $ \Large { t_n (w^T \phi(x) + b) \over ||w||}$
</div><br>

형태로 표현이 가능하며, 이 거리를 최대화 하는 방식으로 목적함수를 설정하여 해결할 수 있다. <br>

<div align="center">
$\underset{w, b}{\mathrm{argmax}}$ $\{ {1 \over ||w||} {\underset{n}{\mathrm{min}} [t_n (w^T \phi(x) + b)]} \}$ 
</div>

여기서 $1 \over ||w||$는 n과 관련이 없으므로 최적화에서 빼내었다. 이 최적화 문제의 해를 직접 구하기가 어렵기 때문에 치환하여 문제를 풀어보겠다. $w \rightarrow \kappa w, b \rightarrow \kappa b $를 시행하더라도 모든 데이터 포인트들로부터 결정 경계까지의 거리는 변하지 않는 다. 따라서 표면에 가장 가까운 포인트에 대해서 다음과 같이 임의로 정할 수 있다. $\kappa$를 이용한 rescale 전략을 취해주면,  <br><br>

<div align="center">$t_n (w^T \phi(x_n) + b) = 1$ </div><br>

의 식으로 변형할 수 있다. 이 경우 모든 데이터 포인트들은 다음의 제약 조건을 만족하게 된다. <br><br>
<div align="center">
$t_n (w^T \phi(x_n) + b) \geqslant 1$,    $n = 1, ..., N$
</div><br> <div align="right"><font color='red'> ... 식 7.5</font></div>

따라서 최적화 문제는 단순히 $1 \over ||w||$를 푸는 문제로 변경된다. 이는 $||w||$의 최소화 문제와 동일하기 때문에 다음과 같이 표현을 해주어도 무방하다. 

<div align="center">
$\underset{w, b}{\mathrm{argmax}}$ $1 \over 2 $ $||w||^2$ 
</div><br>
<div align="right"><font color='red'> ... 식 7.6</font></div>























---

제약조건이 있는 최적화의 문제를 풀기 위해서 라그랑주 승수 $a_n \geq 0$ 을 도입한다.
<font color='red'> 식 7.5</font> 에서 하나의 승수를 도입하면, 

\begin{equation}
     L(w, b, a) = {1 \over 2} ||w||^2 - \overset{N}{\sum_{n=1}} a_n \{ t_n (w^T \phi(x_n) + b) - 1\}
\end{equation}

여기서 $a_n = \{a_1, ..., a_N \}^T$ 이다. $w, b$ 에 대해서는 최소화하고, a에 대해서는 최대화 하므로 라그랑주 승수 항 앞에 음의 부호가 추가되었다. 위 수식을 $w, b$의 편미분은 0으로 놓으면 다음의 두 조건을 얻게 된다. 

\begin{equation}
     w = \overset{N}{\sum_{n=1}} a_n t_n \phi(x_n)
\end{equation}
<div align="right"><font color='red'> ... 식 7.8</font></div>

\begin{equation}
     0 = \overset{N}{\sum_{n=1}} a_n t_n
\end{equation}
<div align="right"><font color='red'> ... 식 7.9</font></div>
이 조건들을 이용해서 $L(w, b, a)$으로부터 w, b에 대입하면, 최대 마진 문제의 듀얼표현을 얻게된다. 

\begin{equation}
     \tilde L(a) = \overset{N}{\sum_{n=1}} a_n - {1 \over 2} \overset{N}{\sum_{n=1}} \overset{M}{\sum_{m=1}} a_n a_m t_n t_m k(x_n, x_m)
\end{equation} <div align="right"><font color='red'> ... 식 7.10</font></div>



이때 최대화는 a에 대해서 일어나야 하며, 다음 제약 조건들을 만족시켜야 한다. 

\begin{equation}
  a_n \geq 0, {\space}{\space}{\space} n = 1, ..., N
\end{equation}
\begin{equation}
  \overset{N}{\sum_{n=1}} a_n t_n = 0
\end{equation}

여기서 커널함수는 $k(x, x \prime ) = \phi(x)^T \phi(x \prime)$ 으로 정의된다. 

부등식 제약 조건을 바탕으로 a에 대한 이차함수를 최적화하는 문제를 풀고 있어서, M개의 변수가 있는 이차 계획법의 문제의 해를 구하려면 $O(M^3)$의 문제를 풀어야 한다. 원래의 문제는 M개의 변수에 대해 식 7.6을 최소화하는 것이고, 변환된 후의 문제는 N개의 변수에 대해 식 7.10을 최소화하는 것이다. 고정된 기저함수(basis function)의 집합을 사용할 때 이 함수의 집합의 수 M이 데이터 포인트의 숫자 N보다 작은 경우는 듀얼 문제로 바꾸는 것이 오히려 불이익으로 보일 수도 있다.  하지만 듀얼 표현을 사용할 경우 커널을 사용하는 방식으로 문제가 바뀌게 된다. 따라서 데이터 포인트의 숫자를 능가하는 차원 수를 가지는 특징 공간에 대해서 최대마진 분류기를 효율적으로 적용할 수 있다. 








훈련된 모델을 바탕으로 새로운 데이터를 분류할 때에는 처음 정의한 식 7.1에서 정의된 $y(x)$를 계산하여 그 부호를 알아내야 한다. 따라서 식 7.8을 통해 w를 대입해주면 원래 식이 매개변수 $a_n$와 커널함수로 구성된 식을 얻을 수 있다. 

\begin{equation}
     y(x) = \overset{N}{\sum_{n=1}} a_n t_n k(x, x_n) + b
\end{equation}
<div align="right"><font color='red'> ... 식 7.13</font></div>

부록 E에서 이러한 형태의 제약 최적화 분제가 KKT조건을 만족시킨다는 것을 배웠기 때문에 이 경우 다음의 세 성질을 만족해야 한다. 

\begin{equation}
\begin{split}
     a_n \geq 0 \\ t_n y(x_n) \geq 0 \\ a_n\{t_ny(x_n) - 1\} = 0
\end{split}
\end{equation}


따라서 모든 데이터 포인트에 대해서 $a_n = 0$이거나 $t_ny(x_n) = 1$이어야 한다. $a_n = 0$인 데이터 포인트는 식 7.13의 합에 나타나지 않을 것이며, 따라서 새로운 데이터 포인트에 대해 예측하는 데 있어서 아무 역할을 하지 않는다. 이외의 나머지 데이터 포인트들은 서포트 벡터라 하며, 예측하는 경계를 제작하는데 사용된다. 이 서포트 벡터들은 $t_ny(x_n) = 1$인 조건을 만족하며, 그렇기 때문에 특징 공간의 최대 마진 초평면상에 있는 포인트에 해당되는 것이다. 한번 모델이 훈련되고 나면, 데이터 포인트들 중 예측에 사용되지 않는 많은 포인트들은 버리게 되고, 서포트 벡터만 남겨 두면 되는 것이다. 

이차 계획법의 문제를 풀어서 a의 값을 찾았다면 이제 b도 계산을 해야 한다. $t_ny(x_n) = 1$에서 7.13식을 대입하면, 
\begin{equation}
     t_n ({\sum_{m \in S}} a_m t_n k(x_n, x_m) + b) = 1
\end{equation}
<div align="right"><font color='red'> ... 식 7.17</font></div>

여기서 $S$는 서포트 벡터의 인덱스의 집합을 지칭한다. 임의로 선택한 서포트 벡터 $x_n$을 사용해서 이 방정식을 풀고 b를 구할 수도 있다. 하지만 양변에 $t_n$을 곱한 후 모든 서포트 벡터들에 대해 이 식의 평균을 내서 b에 대해 푸는 방식을 사용하면 수치적으로 더 안정적인 해를 구할 수 있다. ${t_n}^2 = 1$을 적용하게 된다. 
\begin{equation}
     b = {1 \over N_S} ({\sum_{n \in S}} ( t_n - {\sum_{m \in S}} a_m t_m k(x_n, x_m))
\end{equation}
<div align="right"><font color='red'> ... 식 7.18</font></div>

여기서 $N_S$는 서포트 벡터의 전체 개수에 해당한다. 

다른 대안적인 모델과의 나중 비교를 위해서 최대 마진 분류기를 오류함수를 최소화 하는 형태로 적을 수 있다. 

\begin{equation}
     \overset{N}{\sum_{n=1}} E_{\infty}(y(x_n)t_n -1) + \lambda ||w||^2
\end{equation}
<div align="right"><font color='red'> ... 식 7.19</font></div>

$E_{\infty}(z)$는 $z \geq 0$ 이면, 0이고, 아니면 $\infty$인 함수이다. 이는 제약 조건식 7.5가 만족되도록 해준다. 정규화 매개변수 $\lambda > 0$ 을 만족하기만 하면 정확한 값은 그리 중요하지 않다는 점을 깊고 넘어가도록 한다. 

단순한 합성 데이터 집합에 가우시안 커널을 사용한 서프트 벡터 머신을 적용해서 분류를 시행한 결과의 예시를 그림 7.2에서 볼 수 있다. 이 데이터 집한은 이차원 데이터 공간 x에서 선형적으로 분리되지 않는다. 하지만 비선형 커널 함수를 통해 간접적으로 정의된 비선형 특징 공간에서는 선형적으로 분리가 가능하다. 따라서 훈련 데이터 포인트들은 원래의 데이터 공간에서도 완벽히 분리된다. 


<그림 7.2>












```

```


```

```


```

```
