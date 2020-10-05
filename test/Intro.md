---
sort: 1
---

# Si-Baek Seong

안녕하세요, 
성시백, Joseph Seong입니다.
네이버 블로그에서 벗어나서, github 블로그를 통해 커리어를 정리해보고 있습니다. 
**머신러닝, 통계, 딥러닝 등의 커리어**에는 코드 수준의 정보를 쉽게 다룰 수 있는 곳이 좀 더 적합해보입니다.
최근 블로그 제작에 도움을 줄 수 있는 사이트들을 몇개 알게 되면서 도전을 시작합니다. 

한편 그간 공부했던 내용들도 조금씩 정리해볼 예정입니다.

[기존 네이버 블로그 Links](https://blog.naver.com/tlqordl89)



# 자기소개

기억나지 않는 학부에서는 **전자공학**을 전공하였습니다. 
석사의 시작은 고려대학교 **뇌공학 연구실에서 패턴인식/기계학습을 공부**하였습니다. EEG기반의 데이터를 어떻게 처리하여 Brain computer interface를 할 수 있는 지 맛보기를 할 수 있던 기간입니다. 
과정 중에 '연구'라는 장르에 대해서, 논문을 찾고, IF, Ranks, Contribution 등을 정리 방법에 대해서 익히고, 연구과제를 PPT로 연구를 정리하는 방법, 연구 Key Paper를 찾아 내 연구논문을 작성할 수 있는 기본 소양을 익히게 되었습니다. 

박사과정의 생활은 연세대학교 Monet 연구실에서 하게 됩니다. 
세브란스 병원 핵의학과에 소속되어있는 연구실에서, **의료데이터(MRI, EEG, PET-CT, Survey) 데이터를 다루는 방법**에 대해서 배우게 되었습니다. 
한편, 패턴인식으로 접하게 된 기계학습과 딥러닝 이론을 SPECT, MRI에 적용할 수 있지 않을까 하는 아이디어를 구현하며, CNN 도메인 지식을 얻게되고, 
비숍의 패턴인식 책을 기반으로 기본기를 다지며, 2018년에 Geometric Convolutional Neural Network논문을 쓰게 되었습니다.  

**gCNN**논문을 가볍게 소개하자면, 뇌(Brain)는 Gray matter(GM)와 White matter(WM)로 구분되어 있습니다. GM가 세포들의 핵으로 구성된 곳이고, WM은 세포들끼리 연결된 전선 같은 곳으로 보시면 되겠습니다. GM의 두께가 neuro-function과 영향이 있을 것(Van Essen and Drury, 1997;Van Essen et al., 1998;Dale et al., 1999;Fischl et al., 1999a;MacDonald et al., 2000) 이라는 생각에서 출발하여, Gray matter의 thickness를 기반으로 각 개인의 functional difference를 볼 수 있을 것이라는 생각을 하게 되었습니다. 이를 통해서 개인별 Classification과 전체적인 function의 저하인 치매 등을 진단할 수 있을 것이라는 가설을 잡았습니다. 방법으로는 Freesurfer라는 Tool을 이용하여, 각 반구(hemisphere)를 Templete space의 구(sphere) 형태로 전환을 할 수 있습니다. Freesurfer는 cpu기반으로 작업하여 매우 오래걸리는 작업(인당 9시간)이 필요하지만, MRI를 이용한 brain을 segmentation할 수 있는 좋은 도구입니다. 이렇게 얻어진 구(sphere) 형태의 brainMRI 정보 중 thickness 데이터를 이용하여 뇌 전체를 보는 three-dimension(3D)이 아닌 2D의 형태의 CNN based 모델을 제작할 수 있도록 여러기법들을 적용한 것이 이 논문의 contribution입니다. 

gCNN의 구동테스트를 위해서는 가장 쉽고 정확한 데이터로 확인을 해보아야 합니다. HCP, Human Connetome Project에서 제공한 청년들의 MRI 데이터와 많은 데모그래픽 데이터를 획득할 수 있습니다. 여기에서 가장 확실한 정보인 성별(Sex)데이터를 이용하여 이 방법론을 테스트하여 논문을 제출하였습니다. 남성과 여성을 GM 두께로 구분해보겠다고? 주변의 신경과 의사선생님들께 물어보니 사실상 불가능하다고 말씀을 해주셨지만, 그렇다면 더 임펙트가 있겠다라는 생각으로 관련 논문을 검색해보았습니다. 설득력을 높여서 제가 생성한 데이터+방법론과 기존 논문을 비교하면서 해당 방법론이 잘 동작하고, 이유있게 구분할 수 있음을 밝혀낸 논문이 [gCNN Links](https://www.frontiersin.org/articles/10.3389/fninf.2018.00042/full) 해당 논문이 되겠습니다. 

이후로 이 연구를 **치매 중 알츠하이머병(영어: Alzheimer's disease, AD)을 진단**하는데 사용할 수 있지 않을까 하는 생각으로 ADNI, Alzheimer's Disease Neuroimaging Initiative에서 제공하는 freesurfer 후처리된 데이터를 이용하여 검사를 해보게 되었습니다. 정상노인군과 치매노인군의 데이터를 이용하여 테스트하면, **90% 이상의 확률로 진단**을 명확하게 해내면서, 이 논문은 다양한 방법론들을 추가로 적용하여 KHBM, OHBM에 중간중간 발전된 포스터 논문의 형태로 개제하였습니다. 이렇게 남녀 성별문제에서 치매에 이르기까지, 다양한 데이터의 전처리가 있었지만, 결과만을 공유해보았습니다. 

이렇게 익숙해진 CNN에 더불어, 한국에서 아동/청소년을 대상으로한 연구를 진행하게 됩니다. 기존의 ADHD(Attention Deficit Hyperactivity Disorder)의 진단을 수행할 때에는 본인들이 직접 할 수 없기 때문에 주변의 선생님, 부모님, 의사들의 설문조사를 기반으로 학생들을 평가하고 있습니다. 여기서 설문지를 1D vector로 바꾸고, 뇌의 MRI에서의 특징 데이터 또는 MRI의 공간데이터를 이용하여 ADHD를 진단하는 연구를 통해서 80%이상의 진단 결과를 얻어내면서 연구를 종료하게 되었습니다. 

연구실의 다른 과제, 칼슘이미지 데이터에도 CNN을 기반으로 한 RCNN계열을 통하여 object(Neuron)을 검출하게 되는 연구도 진행하는데, 다음 시간이 되면 정리해보겠습니다. 

매주 목요일 저녁 8시에는 스터디 그룹을 통해서 날로 새롭게 발전하는 딥러닝 방법에 대해서 수학을 하고 있습니다. 
[스터디 논문](https://trello.com/b/vCD6pP9t/paper-study)의 리스트를 참고하실 수 있습니다.


###### Education

| Course  | Subjects |
| ------- | -------- |
| Master  | BCI, ML  |
| doctorate | DNN, MRI |


###### Jobs
**editmate** 서비스 발전 - [에딧살롱 기획](https://www.facebook.com/editmate.kr/videos/3193190497400795)

결혼과 동시에 금전적인 문제로 회사생활을 하게 되었습니다. 
회사에서는 연구소장이라는 직책을 맡으며 국가 연구과제를 담당해서 처리하고 있고, 
서비스 피벗을 통해서 editmate라는 서비스를 발전시키고, 사업화를 진행하였습니다. 
여기서 데이터를 이용한 마케팅에 대해서 공부하게 되었고, 이메일 마케팅을 이용하여 3인이 월 매출을 5000만원까지 올려보는 데 성공하였습니다. 
새로운 PO를 뽑고, 저는 다시 원래의 Job으로 돌아오고자 노력하고 있습니다. 

---

Task list:

- [x] Create a sample markdown document
- [x] Add task lists to it
- [ ] Looking for a new job
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

```
This is the final element on the page and there should be no margin below this.
```
