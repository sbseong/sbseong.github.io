---
sort: 1
---

# Si-Baek Seong

안녕하세요, 
네이버 블로그에서 벗어나서, github블로그를 만들어보고 있습니다. 
아무래도 코드 수준의 정보를 쉽게 다룰 수 있는 곳이 좀 더 적합하다고 생각했고, 
최근 블로그 제작에 도움을 줄 수 있는 사이트들을 몇개 알게 되면서 도전을 시작합니다. 

연습
Text Bold : **Bold sample**
Text _italic_italic sample_
[Links](https://blog.naver.com/tlqordl89)



# 자기소개

학부에서는 전자공학을 전공하였고, 뇌공학 연구실에 1년정도 몸 답으며, 패턴인식을 공부하였습니다. 한편 EEG기반의 데이터를 어떻게 처리하여 Brain computer interface를 할 수 있는 지 맛보기를 할 수 있던 기간입니다. 여기에서 연구를 하는 방법, 논문 정리 방법과 PPT로 연구를 정리하는 방법, 논문을 찾고, Ranks를 확인하고, Key Paper를 찾아 내 연구를 수행할 수 있는 기본 소양을 익히게 되었습니다. 

대학원 생활은 이론뇌공학연구실에서 하게 됩니다. 세브란스 병원 핵의학과에 소속되어있는 연구실에서, 의료데이터(MRI, EEG, PET-CT, Survey) 데이터를 다루는 방법에 대해서 배우게 되었습니다. 한편, 패턴인식으로 
접하게 된 기계학습과 딥러닝 이론을 MRI에 적용할 수 있지 않을까 하는 아이디어를 구현하며, CNN 도메인 지식을 얻게되고, 비숍의 패턴인식 책을 기반으로 공부하며, Geometric Convolutional Neural Network논문을 쓰게 되었습니다.  

**gCNN**논문을 가볍게 소개하자면, 뇌는 Gray matter와 White matter로 구분되어 있습니다. Gray matter가 세포들이 구성된 곳으로, 이 두께가 neuro-function과 영향이 있을 것이라는 생각에서 출발하여, Gray matter의 thickness를 기반으로 사람의 functional difference를 볼 수 있을 것이라는 생각에서 기인합니다. Freesurfer라는 Tool을 이용하면, 각 반구(hemisphere)를 구 형태로 전환을 할 수 있게 됩니다. cpu기반으로 처리하여 매우 오래걸리는 작업이 필요하지만, MRI를 이용한 brain을 segmentation할 수 있는 좋은 툴입니다. 이렇게 변환된 sphere형태의 brain을 이용하여 3D이 아닌 2D의 형태의 CNN을 할 수 있도록 제작한 것이 이 논문의 contribution입니다. 

구동테스트를 위해서는 가장 쉬운 데이터로 확인을 해보아야 합니다. HCP라고, Human Connetome Project에서 제공한 MRI 데이터를 이용하면, MRI와 많은 데이터를 획득할 수 있는데, 여기에서 가장 확실한 정보인 성별데이터를 이용하여 이 방법론을 테스트합니다. 남성과 여성을 뇌 두께로 구분해보겠다고? 주변의 신경과 의사선생님들께 물어보니 사실상 불가능하다고 말씀을 해주셨지만, 관련 논문을 어느정도 찾아볼 순 있었습니다. 아무튼 제가 생성한 데이터와 기존 논문을 비교하면서 해당 방법론이 잘 동작하고, 이유있게 구분할 수 있음을 밝혀낸 논문이 [Links]https://www.frontiersin.org/articles/10.3389/fninf.2018.00042/full 해당 논문이 되겠습니다. 

한편 그 이후로는 이 연구를 치매 중 알츠하이머병(영어: Alzheimer's disease, AD)을 진단하는데 사용할 수 있지 않을까 하는 생각으로 ADNI, Alzheimer's Disease Neuroimaging Initiative에서 제공하는 freesurfer 후처리된 데이터를 이용하여 검사를 해보게 되었습니다. 정상노인군과 치매노인군의 데이터를 이용하여 테스트하면, 90% 이상의 확률로 진단을 명확하게 해내면서, 이 논문은 다양한 방법론들을 추가로 적용하여 KHBM, OHBM에 중간중간 발전된 포스터 논문의 형태로 개제하였습니다. 이렇게 남녀 성별문제에서 치매에 이르기까지, 다양한 데이터의 전처리가 있었지만, 결과만을 공유해보았습니다. 

이렇게 익숙해진 CNN에 더불어, 한국에서 아동/청소년을 대상으로한 연구를 진행하게 됩니다. 기존의 ADHD(Attention Deficit Hyperactivity Disorder)의 진단을 수행할 때에는 본인들이 직접 할 수 없기 때문에 주변의 선생님, 부모님, 의사들의 설문조사를 기반으로 학생들을 평가하고 있습니다. 여기서 설문지를 1D vector로 바꾸고, 뇌의 MRI에서의 특징 데이터 또는 MRI의 공간데이터를 이용하여 ADHD를 진단하는 연구를 통해서 80%이상의 진단 결과를 얻어내면서 연구를 종료하게 되었습니다. 

한편, 연구실의 다른 과제, 칼슘이미지 데이터에도 CNN을 기반으로 한 RCNN계열을 통하여 object(Neuron)을 검출하게 되는 연구도 진행하는데, 다음 시간이 되면 정리해보겠습니다. 


###### Test

| What    | Follows  |
| ------- | -------- |
| A table | A header |
| A table | A header |
| A table | A header |

---

And an unordered task list:

- [x] Create a sample markdown document
- [x] Add task lists to it
- [ ] Take a vacation
- [ ] move naver.blog to here

Definition lists can be used with HTML syntax. Definition terms are bold and italic.

<dl>
    <dt>Name</dt>
    <dd>Godzilla</dd>
    <dt>Born</dt>
    <dd>1952</dd>
    <dt>Birthplace</dt>
    <dd>Japan</dd>
    <dt>Color</dt>
    <dd>Green</dd>
</dl>

---

Tables should have bold headings and alternating shaded rows.

| Artist          | Album          | Year |
| --------------- | -------------- | ---- |
| Michael Jackson | Thriller       | 1982 |
| Prince          | Purple Rain    | 1984 |
| Beastie Boys    | License to Ill | 1986 |

If a table is too wide, it should condense down and/or scroll horizontally.

---

Small images should be shown at their actual size.

![Octocat](https://github.githubassets.com/images/icons/emoji/octocat.png)

Large images should always scale down and fit in the content container.

![Branching](https://guides.github.com/activities/hello-world/branching.png)

```
This is the final element on the page and there should be no margin below this.
```
