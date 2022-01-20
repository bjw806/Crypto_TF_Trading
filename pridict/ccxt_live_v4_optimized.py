import matplotlib.pyplot as plt
import mpl_finance
import numpy
from matplotlib.gridspec import GridSpec
import talib
import ccxt
import time as TIME
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential, load_model
import pandas
import datetime
import pprint
import telegram_bot

img_width, img_height = 300, 300
model_path = '../models/EffNet_v3/fine_model.h5' ##/model.h5
predict_file = '../test_data/ccxt_binance_test.jpg'
model = load_model(model_path)

binance_futures = ccxt.binance(config={
        'apiKey': '3vldXTBVRHRM9C3cXdWBvfa4wkyOaVFjbL91dpYfyYonXsoMraO1MXcrXaxZ8vSW',
        'secret': 'kPL3Gl06MRTZ6tfkwWTJN9ZXT5sEQhpyV9hE5kOL0cw4OgVTPO9WadYLOagQmWx3',
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}})

binance = ccxt.binance({'options': {'defaultType': 'future'}})
ticker = binance.fetch_ticker('BTC/USDT')
markets = binance_futures.load_markets()
market = binance_futures.market("BTC/USDT")
leverage = 10 #x배
resp = binance_futures.set_leverage("BTC/USDT",leverage)

position_counter = ['n/a','n/a','n/a'] # like a stack 
my_position = 'n/a'
assets = 0
total_profit = 0
margin = 0
have_position = False
trade_amount=0.001
win_count = 0
lose_count = 0
winrate = 0

# ohlcv
# [[time, open, high, low, close, volume], [], []]
#   0      1     2    3     4       5
def ccxt_graph():
    ohlcv_1m = binance.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=30)
    ohlcv_1m_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=130)
    ohlcv_5m = binance.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=20)
    ohlcv_5m_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=120)
    ohlcv_15m = binance.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=15)
    ohlcv_15m_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=115)
    ohlcv_1h = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=10)
    ohlcv_1h_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=110)

    open_1m, high_1m, low_1m, close_1m, volume_1m, date_1m, close_1m_pd = [], [], [], [], [], [], []
    open_5m, high_5m, low_5m, close_5m, volume_5m, date_5m, close_5m_pd = [], [], [], [], [], [], []
    open_15m, high_15m, low_15m, close_15m, volume_15m, date_15m, close_15m_pd = [], [], [], [], [], [], []
    open_1h, high_1h, low_1h, close_1h, volume_1h, date_1h, close_1h_pd = [], [], [], [], [], [], []
    for x in range(30):
        open_1m.append(float(ohlcv_1m[x][1]))
        high_1m.append(float(ohlcv_1m[x][2]))
        low_1m.append(float(ohlcv_1m[x][3]))
        close_1m.append(float(ohlcv_1m[x][4]))
        volume_1m.append(float(ohlcv_1m[x][5]))
        date_1m.append(ohlcv_1m[x][0])
    for x in range(130):
        close_1m_pd.append(float(ohlcv_1m_for_ma[x][4]))
    for x in range(20):
        open_5m.append(float(ohlcv_5m[x][1]))
        high_5m.append(float(ohlcv_5m[x][2]))
        low_5m.append(float(ohlcv_5m[x][3]))
        close_5m.append(float(ohlcv_5m[x][4]))
        volume_5m.append(float(ohlcv_5m[x][5]))
        date_5m.append(ohlcv_5m[x][0])
    for x in range(120):
        close_5m_pd.append(float(ohlcv_5m_for_ma[x][4]))
    for x in range(15):
        open_15m.append(float(ohlcv_15m[x][1]))
        high_15m.append(float(ohlcv_15m[x][2]))
        low_15m.append(float(ohlcv_15m[x][3]))
        close_15m.append(float(ohlcv_15m[x][4]))
        volume_15m.append(float(ohlcv_15m[x][5]))
        date_15m.append(ohlcv_15m[x][0])
    for x in range(115):
        close_15m_pd.append(float(ohlcv_15m_for_ma[x][4]))
    for x in range(10):
        open_1h.append(float(ohlcv_1h[x][1]))
        high_1h.append(float(ohlcv_1h[x][2]))
        low_1h.append(float(ohlcv_1h[x][3]))
        close_1h.append(float(ohlcv_1h[x][4]))
        volume_1h.append(float(ohlcv_1h[x][5]))
        date_1h.append(ohlcv_1h[x][0])
    for x in range(110):
        close_1h_pd.append(float(ohlcv_1h_for_ma[x][4]))

    MA_7_1m = talib.MA(numpy.array(close_1m_pd), timeperiod=7)
    MA_25_1m = talib.MA(numpy.array(close_1m_pd), timeperiod=25)
    MA_99_1m = talib.MA(numpy.array(close_1m_pd), timeperiod=99)
    MA_7_5m = talib.MA(numpy.array(close_5m_pd), timeperiod=7)
    MA_25_5m = talib.MA(numpy.array(close_5m_pd), timeperiod=25)
    MA_99_5m = talib.MA(numpy.array(close_5m_pd), timeperiod=99)
    MA_7_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=7)
    MA_25_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=25)
    MA_99_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=99)
    MA_7_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=7)
    MA_25_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=25)
    MA_99_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=99)

    fig = plt.figure(num=1, figsize=(15.25, 15.35), dpi=50, facecolor='w', edgecolor='k') # figsize: ppi dpi: 해상도
    gs = GridSpec(nrows=20, ncols=1)######비율
    cx_1h = fig.add_subplot(gs[0:4, 0])
    vx_1h = fig.add_subplot(gs[4, 0])
    cx_15m = fig.add_subplot(gs[5:9, 0])
    vx_15m = fig.add_subplot(gs[9, 0])
    cx_5m = fig.add_subplot(gs[10:14, 0]) #111은 subplot 그리드 인자를 정수 하나에 다 모아서 표현한 것.(1x1그리드에 첫 번째 subplot)
    vx_5m = fig.add_subplot(gs[14, 0]) #볼륨차트 추가
    cx_1m = fig.add_subplot(gs[15:19, 0])
    vx_1m = fig.add_subplot(gs[19, 0])
    mpl_finance.volume_overlay(vx_1h, open_1h, close_1h, volume_1h, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_1h, open_1h,close_1h,high_1h,low_1h, width=0.965, colorup='r', colordown='b', alpha=1)
    mpl_finance.volume_overlay(vx_15m, open_15m, close_15m, volume_15m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_15m, open_15m,close_15m,high_15m,low_15m, width=0.965, colorup='r', colordown='b', alpha=1)
    mpl_finance.volume_overlay(vx_5m, open_5m, close_5m, volume_5m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_5m, open_5m,close_5m,high_5m,low_5m, width=0.965, colorup='r', colordown='b', alpha=1)
    mpl_finance.volume_overlay(vx_1m, open_1m, close_1m, volume_1m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_1m, open_1m, close_1m, high_1m, low_1m, width=0.965, colorup='r', colordown='b', alpha=1)
    plt.autoscale()  # 자동 스케일링

    line_width = 4
    cx_1h.plot(MA_7_1h[99:], color='gold', linewidth=line_width, alpha=1)
    cx_1h.plot(MA_25_1h[99:], color='violet', linewidth=line_width, alpha=1)
    cx_1h.plot(MA_99_1h[99:], color='green', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_7_15m[99:], color='gold', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_25_15m[99:], color='violet', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_99_15m[99:], color='green', linewidth=line_width, alpha=1)
    cx_5m.plot(MA_7_5m[99:], color='gold', linewidth=line_width, alpha=1)#99+~이니까 99부터 시작
    cx_5m.plot(MA_25_5m[99:], color='violet', linewidth=line_width, alpha=1)
    cx_5m.plot(MA_99_5m[99:], color='green', linewidth=line_width, alpha=1)
    cx_1m.plot(MA_7_1m[99:], color='gold', linewidth=line_width, alpha=1)
    cx_1m.plot(MA_25_1m[99:], color='violet', linewidth=line_width, alpha=1)
    cx_1m.plot(MA_99_1m[99:], color='green', linewidth=line_width, alpha=1)
    plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
    vx_1h.axis('off')
    cx_1h.axis('off')
    vx_15m.axis('off')
    cx_15m.axis('off')
    vx_5m.axis('off')
    cx_5m.axis('off')
    cx_1m.axis('off')
    vx_1m.axis('off')

    plt.savefig('../test_data/ccxt_binance_test.jpg', bbox_inches='tight')  # uuid.uuid4()
    # plt.show()
    plt.cla()  # 좌표축을 지운다.
    plt.clf()  # 현재 Figure를 지운다.

def predict(file):
    x = load_img(file, target_size=(img_width,img_height))
    x = img_to_array(x)
    x = numpy.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    precision = 0
    if result[0] > result[1]:
        if result[0] > 0.9:
            print("Predicted: Long")
            answer = 'long'
            precision = result[0]
        else:
            print("Predicted: Not confident")
            answer = 'n/a'
            precision = result[0]
            print(result)
    else:
        if result[1] > 0.9:
            print("Predicted: Short")
            answer = 'short'
            precision = result[1]
        else:
            print("Predicted: Not confident")
            answer = 'n/a'
            precision = result[1]
            print(result)
    precision = f'{precision:0.4f}'
    return answer, precision

def select_position(input_pos):
    #print(position_counter)
    global position_counter
    if(input_pos == 'n/a'):
        position_counter = ['n/a','n/a','n/a']
        print(position_counter)
    else:
        if(len(position_counter) == 0):
            position_counter = [input_pos]
            print(position_counter)
        elif(len(position_counter) == 1):
            if(position_counter[0] == input_pos):
                position_counter = [input_pos, input_pos]
                print(position_counter)
            else:
                position_counter = [input_pos]
                print(position_counter)
        elif(len(position_counter) == 2):
            if(position_counter[1] == input_pos):
                position_counter = [input_pos, input_pos, input_pos]
                print(position_counter)
                return input_pos
            else:
                position_counter = [input_pos]
                print(position_counter)
        elif(len(position_counter) == 3): # 연속으로 같은 pos가 4번 이상 나왔을 경우 무시
            if(position_counter[2] == input_pos):
                print(position_counter)
            else:
                position_counter = [input_pos]
                print(position_counter) # [l, x, x]
    return None

def trade(selected_position):
    global my_position
    if(selected_position == 'long' and my_position != 'long'):#
        #print("Long 포지션 진입")
        if(my_position == 'short'): # short to long
            close_short_position()
            open_long_position()
            my_position = 'long'
        elif(my_position == 'n/a'): # n/a to long
            open_long_position()
            my_position = 'long'
    elif(selected_position == 'short' and my_position != 'short'):#
        #print("Short 포지션 진입")
        if(my_position == 'long'): # long to short
            close_long_position()
            open_short_position()
            my_position = 'short'
        elif(my_position == 'n/a'): # n/a to short
            open_short_position()
            my_position = 'short'
    else:
        print("trade: None")

def open_long_position():
    print('Long Open')
    order = binance_futures.create_market_buy_order(symbol="BTC/USDT", amount=trade_amount)
    price = order["price"]
    cost = order["cost"]
    telegram_bot.send_trading_message(price, cost, "Long", "Open", trade_amount, 0, total_profit)
    #pprint.pprint(order)

def open_short_position():
    print('Short Open')
    order = binance_futures.create_market_sell_order(symbol="BTC/USDT", amount=trade_amount)
    price = order["price"]
    cost = order["cost"]
    telegram_bot.send_trading_message(price, cost, "Short", "Open", trade_amount, 0, total_profit)
    #pprint.pprint(order)

def close_long_position():
    global total_profit
    global win_count
    global lose_count
    global winrate
    print('Long Close')
    #re_info()
    total_profit += margin
    if(margin > 0):
        win_count += 1
    else:
        lose_count += 1
    if((win_count+lose_count) != 0):
        winrate = win_count / (win_count + lose_count)
    order = binance_futures.create_market_sell_order(symbol="BTC/USDT", amount=trade_amount)
    price = order["price"]
    cost = order["cost"]
    telegram_bot.send_trading_message(price, cost, "Long", "Close", trade_amount, margin, total_profit)
    #pprint.pprint(order)

def close_short_position():
    global total_profit
    global win_count
    global lose_count
    global winrate
    print('Short Close')
    #re_info()
    total_profit += margin
    if(margin > 0):
        win_count += 1
    else:
        lose_count += 1
    if((win_count+lose_count) != 0):
        winrate = win_count / (win_count + lose_count)
    order = binance_futures.create_market_buy_order(symbol="BTC/USDT", amount=trade_amount)
    price = order["price"]
    cost = order["cost"]
    telegram_bot.send_trading_message(price, cost, "Short", "Close", trade_amount, margin, total_profit)
    #pprint.pprint(order)

def re_info():
    global margin
    global have_position
    global assets
    global my_position

    balance = binance_futures.fetch_balance()
    assets = (binance_futures.fetch_balance(params={"type": "future"}))['USDT']
    positions = balance['info']['positions']
    for position in positions:
        if position["symbol"] == "BTCUSDT":
            margin = position["unrealizedProfit"]
            if(str(position["positionAmt"]) == '0.000'):
                have_position = False
                my_position = 'n/a'
            else:
                have_position = True

def print_my_info():
    print("==========| My Info |==========")
    print("Assets:",assets)
    print("My Position:", my_position)
    print("Total_profit:", total_profit," USDT")
    print("Present Margin:", margin," USDT")
    print("Wins:", win_count)
    print("Loses:",lose_count)
    print("Winrate:",winrate," %")


if __name__ == "__main__":
    re_info()
    print(assets)
    while (1):
        date = datetime.datetime.now()
        time = date.strftime('%H:%M:%S')
        sec = (str(time).split(':'))[2]
        min = (str(time).split(':'))[1]
        if(sec == '25' or sec == '55'): # 연산 지연시간때문. 
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            t = "\r\n==========|"+date+"|=========="
            print(t)
            ccxt_graph()
            answer, precision = predict(predict_file)
            re_info()
            select = select_position(answer)
            if(answer != 'n/a' and answer != None): #long or short 일 경우
                precision = " " + str(float(precision)*100) + "%"
                print("          Precision:",precision)
                #print(date, precision)
                if(select != None):
                    trade(select)
            print("unrealizedProfit:", margin)         
            print("\r\n \r\n")

        if(min == '10' or min == '20' or min == '30' or min == '40' or min == '50' or min == '00'):
            print_my_info()