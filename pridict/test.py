import ccxt
import time
import matplotlib.pyplot as plt
import mpl_finance
from matplotlib.gridspec import GridSpec
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from datetime import datetime

binance_futures = ccxt.binance({ 'options': { 'defaultType': 'future' } })
ticker = binance_futures.fetch_ticker('BTC/USDT')


img_width, img_height = 300, 300
model_path = '../model/weights-improvement/weights-improvement-104-1.00.h5'#/model.h5'
weights_path = '../model/weights'
model = load_model(model_path)
model.load_weights(weights_path)

#ohlcv
#[[time, open, high, low, close, volume], [], []]
#   0      1     2    3     4       5
def first_5candle():
	candles = binance_futures.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=7)
	return candles

def live_data():
	#while (1):
		data = binance_futures.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=1)
		# print('\r',ohlcv[0][5])
		return data

def draw_graph(ohlcv):
	open = []
	high = []
	low = []
	close = []
	volume = []
	for x in range(7):
		open.append(ohlcv[x][1])
		high.append(ohlcv[x][2])
		low.append(ohlcv[x][3])
		close.append(ohlcv[x][4])
		volume.append(ohlcv[x][5])

	fig = plt.figure(num=1, figsize=(8, 4), dpi=90, facecolor='w', edgecolor='k')  # figsize: ppi dpi: 해상도
	gs = GridSpec(nrows=2, ncols=2, width_ratios=[2, 1], height_ratios=[2, 1])  ######비율
	dx = fig.add_subplot(gs[0, 0])  # 111은 subplot 그리드 인자를 정수 하나에 다 모아서 표현한 것.(1x1그리드에 첫 번째 subplot)
	ax = fig.add_subplot(gs[1, 0])  # 볼륨차트 추가
	mpl_finance.volume_overlay(ax, open, close, volume, width=0.4, colorup='r', colordown='b', alpha=1)
	mpl_finance.candlestick2_ochl(dx, open, close, high, low, width=0.9, colorup='r', colordown='b',
							  alpha=1)  # width=1.5 alpha=0.5
	plt.autoscale()  # 자동 스케일링
	# dx.plot(smb, color="gold", linewidth=3, alpha=1) #20일선 파란선으로 그리기
	plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
	dx.axis('off')
	#plt.show()
	plt.savefig('../test_data/tester1.jpg', bbox_inches='tight')
	open.clear()
	close.clear()
	volume.clear()
	high.clear()
	low.clear()
	plt.cla()  # 좌표축을 지운다.
	plt.clf()  # 현재 Figure를 지운다.


def predict(file):
  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]
  if result[0] > result[1]:
    if result[0] > 0.8:
      #print("Predicted answer: Long")
      answer = 'long'
      #print(result)
    else:
      #print("Predicted answer: Not confident")
      answer = 'n/a'
      #print(result)
  else:
    if result[1] > 0.8:
      #print("Predicted answer: Short")
      answer = 'short'
      #print(result)
    else:
      #print("Predicted answer: Not confident")
      answer = 'n/a'
      #print(result)

  return answer


def buy(USDT, BTC, price):
	PNL = (USDT - price)*20
	remain_USDT = USDT + PNL
	remain_BTC = BTC + 1
	return remain_USDT, remain_BTC

def sell(USDT, BTC, price):
	PNL = (USDT - price)*20
	remain_USDT = USDT + PNL
	remain_BTC = BTC - 1
	return remain_USDT, remain_BTC


if __name__ == "__main__":
	USDT = 100000
	BTC = 0
	my_pos = None
	my_pos_price = None
	ticker = binance_futures.fetch_ticker('BTC/USDT')
	first_budget = USDT + (BTC*ticker['close'])
	print("Start Budget:",first_budget)
	#was_pos = 'short'
	long_select = 0
	short_select = 0
	win = 0
	lose = 0
	#################################################
	ohlcv = first_5candle()
	draw_graph(ohlcv)
	answer = predict('../test_data/tester1.jpg')
	ticker = binance_futures.fetch_ticker('BTC/USDT')
	if (answer == 'long'):
		print("buy long pos:", ticker['close'])
		my_pos = 'long'
		my_pos_price = ticker['close']
		long_select += 1
		USDT -= ticker['close']
	elif (answer == 'short'):
		print("buy short pos:", ticker['close'])
		my_pos = 'short'
		my_pos_price = ticker['close']
		short_select += 1
		USDT -= ticker['close']
	budget_1 = USDT
	#################################################
	while(1):
		print("")
		ohlcv = first_5candle()
		draw_graph(ohlcv)
		answer = predict('../test_data/tester1.jpg')
		ticker = binance_futures.fetch_ticker('BTC/USDT')
		if(answer == 'long'):
			if(my_pos == 'short'):
				print("buy long pos:",ticker['close'])
				my_pos = 'long'
				if(ticker['close'] < my_pos_price):
					PNL = abs(ticker['close'] - my_pos_price)#*20
					USDT = USDT + PNL + ticker['close']
					print("win:", PNL)
					win += 1
				else:
					PNL = abs(my_pos_price - ticker['close'])#*20
					USDT = USDT - PNL + ticker['close']
					print("lose:", PNL)
					lose += 1
				my_pos_price = ticker['close']
				USDT -= ticker['close']
				#USDT, BTC = buy(USDT, BTC, ticker['close'])
				long_select += 1
			else:
				print("Long, but Do nothing")
		elif(answer == 'short'): #elif
			if (my_pos == 'long'):
				print("buy short pos:", ticker['close'])
				my_pos = 'short'
				if (ticker['close'] > my_pos_price):
					PNL = abs(ticker['close'] - my_pos_price) #* 20
					USDT = USDT - PNL + ticker['close']
					print("win:",PNL)
					win += 1
				else:
					PNL = abs(my_pos_price - ticker['close']) #* 20
					USDT = USDT + PNL + ticker['close']
					print("lose:", PNL)
					lose += 1
				my_pos_price = ticker['close']
				USDT -= ticker['close']
				# USDT, BTC = buy(USDT, BTC, ticker['close'])
				short_select += 1
			else:
				print("Short, but Do nothing")

		if(my_pos == 'short'):
			ROE = ((my_pos_price - ticker['close'])/ticker['close'])*20
		else:
			ROE = ((ticker['close'] - my_pos_price)/ticker['close'])*20

		print(datetime.now())
		print("USDT:",USDT,"POS:",my_pos,my_pos_price)
		budget = USDT
		print("budget:", budget,"NOW:",ticker['close'])
		print("Pos ROE:",ROE)
		print("Total PNL:",(budget - budget_1)/100)
		print("Long:",long_select,"Short:",short_select)
		print("Win:",win,"Lose",lose)
		time.sleep(10)

