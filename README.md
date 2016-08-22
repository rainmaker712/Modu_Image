# Modu_Image
Modu Deep Lab Image Team



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

link: https://drive.google.com/open?id=0B91EMY769GPfMjRvUl9Gb0ktRFU

 추출방법 : t = pickle.load(open("emotion_crawling_dataset.pickle", "r"))

  y = t['train_labels']

  x = t['train_dataset']

3.하나의 감정을 모델링까지 (가능하다면) - Happy 인지 아닌지

<프로젝트 관련>

1.화면 실시간 캡쳐 및 인식 - 실시간 tracking보다 간헐적으로 capture 하는 방식으로 한다.

2.누구인지 인식 - MS face API detect를 이용?

3.최종적으로는 얼굴특징인식(화살표 방식의 특징 인식 기법)을 이용하여 인식율을 올려보자.
