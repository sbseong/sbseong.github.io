---
sort: 3
---

## XLnet for NLP example

## Section 1. 개발 테스트 환경 구축
  구글 드라이브의 데이터와 모델을 불러올 수 있도록 연동하는 과정


```
#!pip install pytorch-transformers # For Colab 
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```
!ls drive/'My Drive'/'Colab Notebooks'
```

     amazon_cells_labelled.txt	    Untitled0.ipynb
     bert_sentiment.ipynb		    XLM-Roberta_tensor.ipynb
     Cooljamm_stat.ipynb		    XLnet_sentiment.ipynb
    'Cooljamm_stat.ipynb의 사본'	    xtest_93_bs3_ep7.npy
     data				    xtest_93.npy
     ids2.npy			    xtest_95_bs3_ep7.npy
     ids_93.npy			    xtrain_93_bs3_ep7.npy
     Keras_API.ipynb		    xtrain_93.npy
     model				    xtrain_95_bs3_ep7.npy
     musicEmotion.ipynb		    ytest_93_bs3_ep7.npy
     requirement.txt		    ytest_93.npy
     sst2_albert.ipynb		    ytest_95_bs3_ep7.npy
    'sst2_albert.ipynb의 사본'	    ytrain_93_bs3_ep7.npy
     test.ipynb			    ytrain_93.npy
     transformerXLnet_sentiment.ipynb   ytrain_95_bs3_ep7.npy



```
PATH = "drive/My Drive/Colab Notebooks/amazon_cells_labelled.txt"
```

## Section 2. 알고리즘 구동을 위한 라이브러리 설치


```
!pip install transformers
```

    Collecting transformers
    [?25l  Downloading https://files.pythonhosted.org/packages/d8/f4/9f93f06dd2c57c7cd7aa515ffbf9fcfd8a084b92285732289f4a5696dd91/transformers-3.2.0-py3-none-any.whl (1.0MB)
    [K     |████████████████████████████████| 1.0MB 4.6MB/s 
    [?25hCollecting sacremoses
    [?25l  Downloading https://files.pythonhosted.org/packages/7d/34/09d19aff26edcc8eb2a01bed8e98f13a1537005d31e95233fd48216eed10/sacremoses-0.0.43.tar.gz (883kB)
    [K     |████████████████████████████████| 890kB 23.6MB/s 
    [?25hRequirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.7)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers) (20.4)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.41.1)
    Collecting sentencepiece!=0.1.92
    [?25l  Downloading https://files.pythonhosted.org/packages/d4/a4/d0a884c4300004a78cca907a6ff9a5e9fe4f090f5d95ab341c53d28cbc58/sentencepiece-0.1.91-cp36-cp36m-manylinux1_x86_64.whl (1.1MB)
    [K     |████████████████████████████████| 1.1MB 32.5MB/s 
    [?25hCollecting tokenizers==0.8.1.rc2
    [?25l  Downloading https://files.pythonhosted.org/packages/80/83/8b9fccb9e48eeb575ee19179e2bdde0ee9a1904f97de5f02d19016b8804f/tokenizers-0.8.1rc2-cp36-cp36m-manylinux1_x86_64.whl (3.0MB)
    [K     |████████████████████████████████| 3.0MB 35.2MB/s 
    [?25hRequirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.18.5)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.15.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.2)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (0.16.0)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.6.20)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers) (2.4.7)
    Building wheels for collected packages: sacremoses
      Building wheel for sacremoses (setup.py) ... [?25l[?25hdone
      Created wheel for sacremoses: filename=sacremoses-0.0.43-cp36-none-any.whl size=893257 sha256=79007c1c1f53169cbfb3b49aa3d9a07bdbd6418a278bbd84f34db9208b8236a8
      Stored in directory: /root/.cache/pip/wheels/29/3c/fd/7ce5c3f0666dab31a50123635e6fb5e19ceb42ce38d4e58f45
    Successfully built sacremoses
    Installing collected packages: sacremoses, sentencepiece, tokenizers, transformers
    Successfully installed sacremoses-0.0.43 sentencepiece-0.1.91 tokenizers-0.8.1rc2 transformers-3.2.0



```
from transformers import *
#from google.colab import files
#uploaded = files.upload()
import torch
# from pytorch_transformers import XLNetTokenizer,XLNetForSequenceClassification
# from pytorch_transformers import XLNetTokenizer
from transformers import XLNetForSequenceClassification
from sklearn.model_selection import train_test_split
#from pytorch_transformers import AdamW
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,DataLoader,RandomSampler,SequentialSampler
```

Dataset 불러오기 및 샘플 데이터 확인


```
fd = pd.read_csv(PATH,sep='\t')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```
fd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>So there is no way for me to plug it in here in the US unless I go by a converter.</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good case, Excellent value.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The mic is great.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I have to jiggle the plug to get it to line up...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



데이터셋 전처리


```
fd.columns = ['sentence','value']
fd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sentence</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good case, Excellent value.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Great for the jawbone.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Tied to charger for conversations lasting more...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The mic is great.</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>I have to jiggle the plug to get it to line up...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```
sentences  = []
for sentence in fd['sentence']:
  sentence = sentence+"[SEP] [CLS]"
  sentences.append(sentence)
```


```
sentences[0:3] 
```




    ['Good case, Excellent value.[SEP] [CLS]',
     'Great for the jawbone.[SEP] [CLS]',
     'Tied to charger for conversations lasting more than 45 minutes.MAJOR PROBLEMS!![SEP] [CLS]']



## Section 3. 학습 및 분류(prediction) 정확도 출력
  1. XLNet 사전학습 모델을 기본으로한 tokenizer를 이용하여 텍스트 정보를 token으로 전환한다. 이때 XLNet’s vocabulary 셋트를 이용하였다.
  2. 정수로 조합된 token의 정보들이 인덱스 정보로 저장되어 모델 학습 및 테스트용으로 사용된다.

Load pre-trained model tokenizer


```
tokenizer  = XLNetTokenizer.from_pretrained('xlnet-base-cased',do_lower_case=True) #xlnet-base-cased
tokenized_text = [tokenizer.tokenize(sent) for sent in sentences]
```


    HBox(children=(FloatProgress(value=0.0, description='Downloading', max=798011.0, style=ProgressStyle(descripti…


    


sample text


```
tokenized_text[100]
```




    ['▁buyer',
     '▁be',
     'ware',
     ',',
     '▁you',
     '▁could',
     '▁flush',
     '▁money',
     '▁right',
     '▁down',
     '▁the',
     '▁toilet',
     '.',
     '[',
     's',
     'ep',
     ']',
     '▁[',
     'cl',
     's',
     ']']




```
ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]
```


```
print(ids[100])
labels = fd['value'].values
print(labels[100])
```

    [8689, 39, 3676, 19, 44, 121, 13179, 356, 203, 151, 18, 8976, 9, 10849, 23, 3882, 3158, 4145, 11974, 23, 3158]
    0


Token 길이 통일을 위한 Padding - 가장 긴 길이의 token을 기준으로 나머지 token의 길이를 맞춰줌




```
max1 = len(ids[0])
for i in ids:
  if(len(i)>max1):
    max1=len(i)
print(max1)
MAX_LEN = max1
```

    54


패딩(padding) 수행


```
#input_ids2 = pad_sequences(ids,maxlen=MAX_LEN,dtype="long",truncating="post",padding="post")
import numpy as np
input_ids2 = np.load('drive/My Drive/Colab Notebooks/ids_93.npy')
```

학습과 테스트를 위한 데이터 분류작업을 진행한다.


```
#xtrain,xtest,ytrain,ytest = train_test_split(input_ids2,labels,test_size=0.15)
xtrain = np.load('drive/My Drive/Colab Notebooks/xtrain_95_bs3_ep7.npy')
ytrain = np.load('drive/My Drive/Colab Notebooks/ytrain_95_bs3_ep7.npy')
xtest = np.load('drive/My Drive/Colab Notebooks/xtest_95_bs3_ep7.npy')
ytest = np.load('drive/My Drive/Colab Notebooks/ytest_95_bs3_ep7.npy')
```


```
ytest[0:10]
```




    array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1])




```
# np.save('drive/My Drive/Colab Notebooks/ids_93.npy', input_ids2) # x_save.npy
# 동일한 index를 이용하여 다시 테스트하기 위한 index저장 코드
```


```
PATH2 = "drive/My Drive/Colab Notebooks/model/XLnet-based-Custom-96"
```


```
Xtrain = torch.tensor(xtrain)
Ytrain = torch.tensor(ytrain)
Xtest = torch.tensor(xtest)
Ytest = torch.tensor(ytest)
```


```
# np.save('drive/My Drive/Colab Notebooks/xtrain_93_bs3_ep7.npy', xtrain) # x_save.npy
# np.save('drive/My Drive/Colab Notebooks/ytrain_93_bs3_ep7.npy', ytrain) # x_save.npy
# np.save('drive/My Drive/Colab Notebooks/ytest_93_bs3_ep7.npy', ytest) # x_save.npy
# np.save('drive/My Drive/Colab Notebooks/xtest_93_bs3_ep7.npy', xtest) # x_save.npy
```

### 사전 학습 모델 load 및 학습 설정 및 학습진행

한번의 iteration에 들어갈 데이터의 양(bs)과, 학습할 레이어의 개수와 얼마나 학습할 지(epoch)를 선택한다. batch_size :(default) 3


```
batch_size = 5 #3
no_train = 0
epochs = 5 # 3
```


```
train_data = TensorDataset(Xtrain,Ytrain)
test_data = TensorDataset(Xtest,Ytest)
loader = DataLoader(train_data,batch_size=batch_size)
test_loader = DataLoader(test_data,batch_size=batch_size)
```

사전 학습된 모델로 성능 테스트 진행


```
def flat_accuracy(preds,labels):  # A function to predict Accuracy
  correct=0
  for i in range(0,len(labels)):
    if(preds[i]==labels[i]):
      correct+=1
  return (correct/len(labels))*100
```


```
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",num_labels=2) # xlnet-base(large)-cased
torch.cuda.empty_cache()
model.cuda()
optimizer = AdamW(model.parameters(),lr=2e-5) # initial learning rate 0.00002

checkpoint = torch.load(PATH2)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
acc = []
lab = []
t = 0
for inp,lab1 in test_loader:
  inp.to(device)
  lab1.to(device)
  t+=lab1.size(0)
  outp1 = model(inp.to(device))
  [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
  [lab.append(z1.item()) for z1 in lab1]
print("Total Examples : {} Accuracy {}".format(t,flat_accuracy(acc,lab)))
```

    /usr/local/lib/python3.6/dist-packages/transformers/configuration_xlnet.py:211: FutureWarning: This config doesn't use attention memories, a core feature of XLNet. Consider setting `men_len` to a non-zero value, for example `xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=1024)`, for accurate training performance as well as an order of magnitude faster inference. Starting from version 3.5.0, the default parameter will be 1024, following the implementation in https://arxiv.org/abs/1906.08237
      FutureWarning,
    Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
    - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['sequence_summary.summary.weight', 'sequence_summary.summary.bias', 'logits_proj.weight', 'logits_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


    Total Examples : 150 Accuracy 96.66666666666667


## Section 4. 학습과정 증명 자료
학습 모델을 다운로드 하고, 여기에 parameter를 설정하여 학습과정을 증명한다.

모델 초기화


```
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased",num_labels=2) # xlnet-base(large)-cased
torch.cuda.empty_cache()
model.cuda()
```

    /usr/local/lib/python3.6/dist-packages/transformers/configuration_xlnet.py:211: FutureWarning: This config doesn't use attention memories, a core feature of XLNet. Consider setting `men_len` to a non-zero value, for example `xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=1024)`, for accurate training performance as well as an order of magnitude faster inference. Starting from version 3.5.0, the default parameter will be 1024, following the implementation in https://arxiv.org/abs/1906.08237
      FutureWarning,
    Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.weight', 'lm_loss.bias']
    - This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).
    - This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['sequence_summary.summary.weight', 'sequence_summary.summary.bias', 'logits_proj.weight', 'logits_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.





    XLNetForSequenceClassification(
      (transformer): XLNetModel(
        (word_embedding): Embedding(32000, 768)
        (layer): ModuleList(
          (0): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (1): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (2): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (3): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (4): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (5): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (6): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (7): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (8): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (9): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (10): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (11): XLNetLayer(
            (rel_attn): XLNetRelativeAttention(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (ff): XLNetFeedForward(
              (layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (layer_1): Linear(in_features=768, out_features=3072, bias=True)
              (layer_2): Linear(in_features=3072, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (sequence_summary): SequenceSummary(
        (summary): Linear(in_features=768, out_features=768, bias=True)
        (first_dropout): Identity()
        (last_dropout): Dropout(p=0.1, inplace=False)
      )
      (logits_proj): Linear(in_features=768, out_features=2, bias=True)
    )




```
optimizer = AdamW(model.parameters(),lr=2e-5) # initial learning rate 0.00002
```


```
import torch.nn as nn
criterion = nn.CrossEntropyLoss()
```


```
# def flat_accuracy(preds,labels):  # A function to predict Accuracy
#   correct=0
#   for i in range(0,len(labels)):
#     if(preds[i]==labels[i]):
#       correct+=1
#   return (correct/len(labels))*100
```

Transfer learning을 시작한다.


```
for epoch in range(epochs):
  model.train()
  loss1 = []
  steps = 0
  train_loss = []
  l = []
  for inputs,labels1 in loader :
    inputs.to(device)
    labels1.to(device)
    optimizer.zero_grad()
    outputs = model(inputs.to(device))
    loss = criterion(outputs[0],labels1.to(device)).to(device)
    logits = outputs[1]
    #ll=outp(loss)
    [train_loss.append(p.item()) for p in torch.argmax(outputs[0],axis=1).flatten() ]#our predicted 
    [l.append(z.item()) for z in labels1]# real labels
    loss.backward()
    optimizer.step()
    loss1.append(loss.item())
    no_train += inputs.size(0)
    steps += 1
  print("Current Loss is : {} Step is : {} number of Example : {} Accuracy : {}".format(loss.item(),epoch,no_train,flat_accuracy(train_loss,l)))
```

    Current Loss is : 0.6339799165725708 Step is : 0 number of Example : 849 Accuracy : 51.9434628975265
    Current Loss is : 0.8297128677368164 Step is : 1 number of Example : 1698 Accuracy : 86.45465253239105
    Current Loss is : 0.020124394446611404 Step is : 2 number of Example : 2547 Accuracy : 91.51943462897526
    Current Loss is : 0.007085510529577732 Step is : 3 number of Example : 3396 Accuracy : 97.1731448763251
    Current Loss is : 0.005526750348508358 Step is : 4 number of Example : 4245 Accuracy : 98.70435806831567


테스트 결과


```
model.eval()#Testing our Model
acc = []
lab = []
t = 0
for inp,lab1 in test_loader:
  inp.to(device)
  lab1.to(device)
  t+=lab1.size(0)
  outp1 = model(inp.to(device))
  [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
  [lab.append(z1.item()) for z1 in lab1]
print("Total Examples : {} Accuracy {}".format(t,flat_accuracy(acc,lab)))

```

    Total Examples : 150 Accuracy 96.0



```
#!pip freeze > 'drive/My Drive/Colab Notebooks/requirement.txt'
```

테스트 결과 모델 저장


```
PATH2 = "drive/My Drive/Colab Notebooks/model/XLnet-based-Custom-96"
# torch.save(model.state_dict(), PATH2)
```


```
# torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'loss': loss,
#             }, PATH2)
```

학습된 모델 재구동 테스트


```
# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)

checkpoint = torch.load(PATH2)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

model.eval()
acc = []
lab = []
t = 0
for inp,lab1 in test_loader:
  inp.to(device)
  lab1.to(device)
  t+=lab1.size(0)
  outp1 = model(inp.to(device))
  [acc.append(p1.item()) for p1 in torch.argmax(outp1[0],axis=1).flatten() ]
  [lab.append(z1.item()) for z1 in lab1]
print("Total Examples : {} Accuracy {}".format(t,flat_accuracy(acc,lab)))
```

    Total Examples : 150 Accuracy 96.66666666666667

