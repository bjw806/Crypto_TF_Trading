import matplotlib.pyplot as plt
import mpl_finance
import numpy
from matplotlib.gridspec import GridSpec
import talib
import ccxt
import time as TIME
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import datetime
import pprint
import sys
import telegram_bot_Xl

numpy.set_printoptions(precision=5, suppress=True)

img_width, img_height = 480, 480
model_path = '../model/weights-improvement/weights-improvement-11-0.99.h5'
predict_file = '../test_data/ccxt_binance_v6.jpg'
model = load_model(model_path)
weights_path = '../model/weights.h5'
model.load_weights(weights_path)

binance_futures = ccxt.binance(config={
    'apiKey': 'Your-api-Key',
    'secret': 'Your-secret',
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}})

binance = ccxt.binance({'options': {'defaultType': 'future'}})
ticker = binance.fetch_ticker('BTC/USDT')
markets = binance_futures.load_markets()
market = binance_futures.market("BTC/USDT")
# leverage = 10 #x배
# resp = binance_futures.set_leverage("BTC/USDT",int(leverage))

my_position = 'n/a'
assets = 0
total_profit = 0
margin = 0
have_position = False
trade_amount = 0.02
win_count = 0
lose_count = 0
winrate = 0


# ohlcv
# [[time, open, high, low, close, volume], [], []]
#   0      1     2    3     4       5
def ccxt_graph():
    ohlcv_15m = binance.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=48)
    ohlcv_15m_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='15m', limit=148)
    ohlcv_1h = binance.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=48)
    ohlcv_1h_for_ma = binance.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=148)

    open_1h, high_1h, low_1h, close_1h, volume_1h, date_1h, close_1h_pd = [], [], [], [], [], [], []
    open_15m, high_15m, low_15m, close_15m, volume_15m, date_15m, close_15m_pd = [], [], [], [], [], [], []

    for x in range(48):
        open_1h.append(float(ohlcv_1h[x][1]))
        high_1h.append(float(ohlcv_1h[x][2]))
        low_1h.append(float(ohlcv_1h[x][3]))
        close_1h.append(float(ohlcv_1h[x][4]))
        volume_1h.append(float(ohlcv_1h[x][5]))
        date_1h.append(ohlcv_1h[x][0])
    for x in range(148):
        close_1h_pd.append(float(ohlcv_1h_for_ma[x][4]))
    for x in range(48):
        open_15m.append(float(ohlcv_15m[x][1]))
        high_15m.append(float(ohlcv_15m[x][2]))
        low_15m.append(float(ohlcv_15m[x][3]))
        close_15m.append(float(ohlcv_15m[x][4]))
        volume_15m.append(float(ohlcv_15m[x][5]))
        date_15m.append(ohlcv_15m[x][0])
    for x in range(148):
        close_15m_pd.append(float(ohlcv_15m_for_ma[x][4]))

    MA_7_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=7)
    MA_25_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=25)
    MA_99_1h = talib.MA(numpy.array(close_1h_pd), timeperiod=99)
    MA_7_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=7)
    MA_25_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=25)
    MA_99_15m = talib.MA(numpy.array(close_15m_pd), timeperiod=99)

    fig = plt.figure(num=1, figsize=(5.94, 5.98), dpi=100, facecolor='w', edgecolor='k')  # figsize: ppi dpi: 해상도
    gs = GridSpec(nrows=10, ncols=1)  ######비율
    cx_15m = fig.add_subplot(gs[0:4, 0])  # 111은 subplot 그리드 인자를 정수 하나에 다 모아서 표현한 것.(1x1그리드에 첫 번째 subplot)
    vx_15m = fig.add_subplot(gs[4, 0])  # 볼륨차트 추가
    cx_1h = fig.add_subplot(gs[5:9, 0])
    vx_1h = fig.add_subplot(gs[9, 0])
    mpl_finance.volume_overlay(vx_1h, open_1h, close_1h, volume_1h, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_1h, open_1h, close_1h, high_1h, low_1h, width=0.965, colorup='r', colordown='b',
                                  alpha=1)
    mpl_finance.volume_overlay(vx_15m, open_15m, close_15m, volume_15m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_15m, open_15m, close_15m, high_15m, low_15m, width=0.965, colorup='r',
                                  colordown='b', alpha=1)
    plt.autoscale()  # 자동 스케일링

    line_width = 3
    cx_1h.plot(MA_7_1h[99:], color='gold', linewidth=line_width, alpha=1)  # 99+~이니까 99부터 시작
    cx_1h.plot(MA_25_1h[99:], color='violet', linewidth=line_width, alpha=1)
    cx_1h.plot(MA_99_1h[99:], color='green', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_7_15m[99:], color='gold', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_25_15m[99:], color='violet', linewidth=line_width, alpha=1)
    cx_15m.plot(MA_99_15m[99:], color='green', linewidth=line_width, alpha=1)
    plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
    vx_1h.axis('off')
    cx_1h.axis('off')
    cx_15m.axis('off')
    vx_15m.axis('off')

    plt.savefig('../test_data/ccxt_binance_v6.jpg', bbox_inches='tight')  # uuid.uuid4()
    plt.cla()  # 좌표축을 지운다.
    plt.clf()  # 현재 Figure를 지운다.


def get_max(input_array):
    max_tmp = 0
    for x in range(3):
        if(max_tmp < input_array[x]):
            max_tmp = input_array[x]
            index = x
        else:
            pass
    return index


def predict(file):
    x = load_img(file, target_size=(img_width, img_height))
    x = img_to_array(x)
    x = numpy.expand_dims(x, axis=0)
    array = model.predict(x)
    result = array[0]
    index = get_max(result)


    if(index == 0):
        if(result[0] > 0.9):
            answer = 'long'
        else:
            answer = 'n/a'
        precision = result[0]
    elif(index == 1):
        if (result[1] > 0.9):
            answer = 'neutral'
        else:
            answer = 'n/a'
        precision = result[1]
    elif (index == 2):
        if (result[2] > 0.9):
            answer = 'short'
        else:
            answer = 'n/a'
        precision = result[2]
    else:
        print("e")
        answer = 'n/a'
        precision = -1

    if(answer == 'n/a'):
        print(result)

    #precision = f'{precision:0.4f}'
    precision = f'{precision:0.4f}'
    return answer, precision


def trade(input_pos, precision):
    # print(position_counter)
    if (input_pos == 'n/a long' or input_pos == 'n/a short'):
        if (precision > 0.8):
            print("[Hold Position]")
        elif (precision < 0.6):
            print("[Close Position]")
            if (input_pos == 'n/a short'):
                if (my_position == 'short'):
                    close_short_position()
            elif (input_pos == 'n/a long'):
                if (my_position == 'long'):
                    close_long_position()
    else:
        if (input_pos == 'long'):
            if (my_position == 'short'):
                print("[Long Position]")
                close_short_position()
                open_long_position()
            elif (my_position == 'n/a'):
                print("[Long Position]")
                open_long_position()
            else:
                print("Already have same position")
        elif (input_pos == 'short'):
            if (my_position == 'long'):
                print("[Short Position]")
                close_long_position()
                open_short_position()
            elif (my_position == 'n/a'):
                print("[Short Position]")
                open_short_position()
            else:
                print("Already have same position")


def open_long_position():
    global my_position
    my_position = 'long'
    print('Long Open')
    # order = binance_futures.create_market_buy_order(symbol="BTC/USDT", amount=trade_amount)
    # price = order["price"]
    # cost = order["cost"]
    # telegram_bot.send_trading_message(price, cost, "Long", "Open", trade_amount, 0, total_profit)
    # pprint.pprint(order)


def open_short_position():
    global my_position
    my_position = 'short'
    print('Short Open')
    # order = binance_futures.create_market_sell_order(symbol="BTC/USDT", amount=trade_amount)
    # price = order["price"]
    # cost = order["cost"]
    # telegram_bot.send_trading_message(price, cost, "Short", "Open", trade_amount, 0, total_profit)
    # pprint.pprint(order)


def close_long_position():
    global total_profit
    global win_count
    global lose_count
    global winrate
    global my_position
    my_position = 'n/a'
    print('Long Close')
    # re_info()

    # total_profit += margin
    # if(margin > 0):
    #    win_count += 1
    # else:
    #    lose_count += 1
    # if((win_count+lose_count) != 0):
    #    winrate = win_count / (win_count + lose_count)
    # order = binance_futures.create_market_sell_order(symbol="BTC/USDT", amount=trade_amount)
    # price = order["price"]
    # cost = order["cost"]
    # telegram_bot.send_trading_message(price, cost, "Long", "Close", trade_amount, margin, total_profit)

    # pprint.pprint(order)


def close_short_position():
    global total_profit
    global win_count
    global lose_count
    global winrate
    global my_position
    my_position = 'n/a'
    print('Short Close')
    # re_info()
    # total_profit += margin
    # if(margin > 0):
    #    win_count += 1
    # else:
    #    lose_count += 1
    # if((win_count+lose_count) != 0):
    #    winrate = win_count / (win_count + lose_count)
    # order = binance_futures.create_market_buy_order(symbol="BTC/USDT", amount=trade_amount)
    # price = order["price"]
    # cost = order["cost"]
    # telegram_bot.send_trading_message(price, cost, "Short", "Close", trade_amount, margin, total_profit)
    # pprint.pprint(order)


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
            margin = float(position["unrealizedProfit"])
            if (float(position["positionAmt"]) == trade_amount):
                have_position = True
                my_position = 'long'
            elif (float(position["positionAmt"]) == -trade_amount):
                have_position = True
                my_position = 'short'
            else:
                have_position = False
                my_position = 'n/a'

            """if(str(position["positionAmt"]) == '0.000'):
                have_position = False
                my_position = 'n/a'
            else:
                have_position = True"""


def print_my_info():
    print("\r\n==========| My Info |==========")
    print("Assets:", assets)
    print("My Position:", my_position)
    print("Total_profit:", total_profit, "USDT")
    print("Present Margin:", margin, "USDT")
    print("Wins:", win_count)
    print("Loses:", lose_count)
    print("Winrate:", winrate, "%")


if __name__ == "__main__":
    re_info()
    print(assets)
    print_my_info()
    close_in_min = 5
    open_time_min = 0
    ccxt_graph()
    answer, precision = predict(predict_file)
    print(answer, precision)
    telegram_bot_Xl.send_message("Loaded")
    while (1):
        date = datetime.datetime.now()
        time = date.strftime('%H:%M:%S')
        sys.stdout.write("\r{0}".format(time))
        sys.stdout.flush()
        sec = (str(time).split(':'))[2]
        min = (str(time).split(':'))[1]
        if (sec == '55'):  # 연산 지연시간때문.min == '29' or
            date = date.strftime('%Y-%m-%d %H:%M:%S')
            t = "\r\n==========|" + date + "|=========="
            print(t)
            ccxt_graph()
            answer, precision = predict(predict_file)
            re_info()
            prec = " " + str(float(precision) * 100) + " %"
            print(answer, prec)
            #trade(answer, float(precision))
            if(answer == 'long'):
                telegram_bot_Xl.send_message("long")
            elif(answer == 'short'):
                telegram_bot_Xl.send_message("short")
            print("My Position:", my_position)
            print("unrealizedProfit:", margin)
            print("\r\n")
        if ((min == '10' or min == '20' or min == '30' or min == '40' or min == '50' or min == '00') and sec == '00'):
            print_my_info()
            # ccxt_graph()
            answer, precision = predict(predict_file)
            # telegram_bot.send_prediction(answer, precision)
            TIME.sleep(1)
        else:
            TIME.sleep(1)
