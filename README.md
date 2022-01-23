# Crypto_TF_Trading
가상화폐 패턴 학습을 이용한 자동거래 프로그램

1) graph/neutral_v1.py
  .csv 파일로부터 data/train/ 에 이미지 생성

2) graph/move_images.py
  이미지 파일을 무작위로 data/test, data/validation 폴더로 이동
  
3) train/EffNetV2XL_v2.py
  EfficientNet V2L 모델을 전이학습하여 model/ 폴더에 model, weight 저장
  
4) pridict/ccxt_v6.py
  ccxt, binance를 이용하여 선물거래. test_data/ccxt_binance_v6.jpg 를 1분마다 갱신하여 이미지 패턴 인식.
  
  4-1) pridict_3.py
    test셋에 대해 검증 (long, short, neutral)
    
  4-2) telegram_bot_Xl.py
    텔레그램으로 메시지 발송
