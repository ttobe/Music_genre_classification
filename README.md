# Music_genre_classification with NUMPY
Music_genre_classification

팀 프로젝트

구현 방향: 최대한 라이브러리를 사용하지 않고 수업시간에 배운대로 numpy로만 구현하기
코드는 밑바닥부터 시작하는 딥러닝을 참고하였습니다.

Best model:
Batch normalization 적용
Node수 = 100
Learning rate = 1e-3 
Dropout ratio = 0.5
Batch size = 256
Hidden layer 수 = 5 
Optimizer = Adam

결과:
Epoch 50/50
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>]
train acc: 82.11098330094006 %
test acc:  77.24028548770818 %

![acc result](https://user-images.githubusercontent.com/101859033/207553124-2fd8f671-6b1e-4759-928a-329b58f7e0f2.png)
![loss result](https://user-images.githubusercontent.com/101859033/207553130-4b51a279-4b1a-4ca5-a357-1e639a075e63.png)



실행방법
1. download music_genre_classification.zip
2. unzip music_genre_classification.zip
3. python main.py
(parameter로 batch_size, num_node, num_hidden_layer, epoch, learingrate를 조절할 수 있습니다.
하는 방법은 python main.py --help를 확인)

출처
데이터 출처
https://www.kaggle.com/datasets/vicsuperman/prediction-of-music-genre

코드 출처 밑바닥부터 시작하는 딥러닝 깃허브 참고
https://github.com/youbeebee/deeplearning_from_scratch/tree/4bbb640b42bddc136abd88b8caa66e874b8a1935
