
# Modu_Image
Modu Deep Lab Image Team 

2016.9.7. 표정인식팀 진행내용 정리

1. 전창욱님
- MS API를 사용하여 DataSet Labeling 1차 구현 방법 공유

2. 신성진님
- MS API 관련하여 openCV를 사용한 방법 공유
- CNN FootBall Sound Data 오픈 소스 공유, 소리의 파장 데이터인 이미지를 CNN에 학습시킨 방법
- For 2013, csv데이터를 사용하여 데이터 전처리 방법 공유 : fer2013_read_Data.ipynb
- RaFD DataSet을 Kadenze 강의에서 사용했던 모델을 사용하여 모델 구현 방법 공유

3. 서기호님
- 딥마인드에서 만든 Neural Tunning Machine 논문 공유 
	: RNN, LSTM에 과 달리 Memory를 사용하여 보다 좋은 효율을 낼 수 있는 기법

4. 기타 토의 
- DataSet을 Model에 어떻게 넣어서 학습을 시켜야 하나??
 : MNIST 처럼 구성하면 될것 같은데 어떻게 구성해야하는지 검토 필요!
- 9/21 발표준비

2016.8.31. 표정인식팀 진행내용 정리

1.TFlearn으로 된 소스코드 리뷰 ==> 모델링팀에서 진행

2.Kadenze Lecture #2
- Lecture2 의 Training Parameters부터 다시 리뷰하기로 함.

3.참고자료
1)Deep Face 논문 (3D face recognition by Facebook AI team)
https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf

2)Face Detection with Dlib in Python : 간단하게 구현 가능 

http://dlib.net/face_detector.py.html

4.프로젝트 방향 : 정지영상(이미지)에서 얼굴을 감지한 후 감정별로 이미지를 분류하는 서비스 
1)데이터 전처리 ==> 전처리팀
2)데이터 트레이닝은 기존 소스를 이용해서 전처리한 데이터로 수행 ==> 모델링팀
3)무작위 이미지 디렉토리를 입력으로 제공하면 자동으로 감정 디렉토리별로 분류함. ==> 함께 진행. 

5.두팀으로 나누어서 진행
1)전처리팀: 전창욱, 강은숙, 옥정훈, HJM 
; 데이터전처리(googling DB, CK+, JAFfE) + 디렉토리별(감정별) 분류 + Face Detection  

2)모델링팀: 한승엽, 신성진, 추광재, 서기호
; 기존 소스를 현재 데이터로 돌아가게 만들어서 학습시키기 + Face Detection(전처리팀) + 이미지를 감정디렉토리별로 분류

6.우리조 발표: 9/14

7. JAFEE DB 설명 (추광재 님)
http://www.kasrl.org/jaffe.html

jaffe db 다운로드 링크입니다.
http://www.kasrl.org/jaffe_info.html
 
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

< Fer2013 Emotion Rule >

0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
