
# Modu_Image
Modu Deep Lab Image Team 
 
2016.8.24. 표정인식 
<오늘 한 내용> 
1. 장난감 그룹 진행 사항 소개 
 - Vision + CNN > Q-Learning > 유전자 알고리즘 3가지를 알아 봤고 가장 쉬운 유전자 알고리즘 부터 시작 했다. 
 - 장난감 키트는 3개 제작 하였다. 
 - 앞으로의 진행 방향에 대한 설명 
2. Q-Learning 개념 설명 (소장님) 
3. RaFD 데이터 Set은 학교에서만 신청이 가능해서 국민대 교수님에게 부탁하는게 좋을거 같다는 의견 도출 
4. 강은숙 연구원, 서기호 연구원에게 우리 그룹의 진행 방향에 대해서 간략한 설명 
5. 다음주 진행 할 내용 이야기
  - Kadenze 2장 
  - Network 같이 설계 (Happy  인지 아닌지 만 분류 해보자) 
  - 데이터 수집(각자 수집 요청) 및 Zero Centers & Normalization


2016.8.17. 표정인식


<오늘 한 내용>

1.Kadenze - Creative Applications of Deep Learning w/ Tensorflow  Session1 code review
 

<다음시간 할 내용>
1.Kadenze source를 이용하여 data를 전처리 해봄.

2.전처리한 데이터(얼굴 cropping + 100x100 size)를 구글드라이브에 올려 오픈링크로 공유하기 - Git의 readme.md

   한승엽 - Crawling Images / 신성진 - CK(+) / 추광재 - JAFFE / 전창욱 - RaFD

신성진 CK & CK+ Data

link: https://drive.google.com/file/d/0Bx984wTo1QhfUnVldXJRUTM5ODQ/view?usp=sharing

한승엽 - Crawling Images 

  추출방법 : t = pickle.load(open("emotion_crawling_dataset.pickle", "r"))
  
  y = t['train_labels']
  
  x = t['train_dataset']

link: https://drive.google.com/open?id=0B91EMY769GPfMjRvUl9Gb0ktRFU




3.하나의 감정을 모델링까지 (가능하다면) - Happy 인지 아닌지

<프로젝트 관련>

1.화면 실시간 캡쳐 및 인식 - 실시간 tracking보다 간헐적으로 capture 하는 방식으로 한다.

2.누구인지 인식 - MS face API detect를 이용?

3.최종적으로는 얼굴특징인식(화살표 방식의 특징 인식 기법)을 이용하여 인식율을 올려보자.

<Fer2013 Emotion Rule>

0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
