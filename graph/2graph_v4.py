from numpy import MAY_SHARE_BOUNDS, genfromtxt
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
from matplotlib.gridspec import GridSpec
from numpy.lib.function_base import corrcoef
import talib
import time as Time

data_1m = genfromtxt('../financial_data/binance_1m_01_04.csv', delimiter=',', dtype=str)

long_dir = '../data/train/long/'
short_dir = '../data/train/short/'
neutral_dir = '../data/train/neutral/'
long_count = 0
short_count = 0
correction = 0


def custom_resmapler(np, start, finish, timedelta):
    open, high, low, close, volume, date = [], [], [], [], [], []
    pd_close = []
    t = start - timedelta  # 한봉 뒤부터 시작
    k = 0
    index = 0
    for x in range(timedelta): # timedelta - 1 절대 하지마!!!!!!!!
        time = data_1m[t][0]  # 0 ~ 4
        time = time.split(':')
        time = int(time[1])
        if (time % timedelta == 0):
            index = t
            k = index  # k에 index를 저장해놓음
            break
        t += 1
    if (index == 0):
        return [], [], [], [], [], [], []
    for x in range((finish - start) // timedelta):  # start(이미 보정된) ~ finish까지 5분봉. 마지막 빼고
        date.append(data_1m[index][0])
        open.append(float(data_1m[index][1]))  # first
        high.append(high_get_max(index, index + timedelta - 1))
        low.append(low_get_min(index, index + timedelta - 1))
        close.append(float(data_1m[index + timedelta - 1][4]))  # last
        volume.append(vol_get_sum(index, index + timedelta - 1))  # 볼륨: 5개 더함
        index += timedelta
    date.append(data_1m[index + 1][0])
    open.append(float(data_1m[index + 1][1]))  # first
    high.append(high_get_max(index + 1, finish - 1))
    low.append(low_get_min(index + 1, finish - 1))
    close.append(float(data_1m[finish - 1][4]))  # last
    volume.append(vol_get_sum(index + 1, finish - 1))  # 볼륨: 5개 더함

    index = k - 100 * timedelta  # close를 MA99 구하기 위해 -100칸 전부터 따로 모은다
    for x in range((finish - start + 100 * timedelta) // timedelta):
        pd_close.append(float(data_1m[index + timedelta - 1][4]))
        index += timedelta
    return open, high, low, close, volume, date, pd_close


def high_get_max(start_index, finish_index):
    max_tmp = float(data_1m[start_index][2])
    for x in range(finish_index - start_index):
        max_tmp = max(max_tmp, float(data_1m[start_index + x][2]))
    return max_tmp


def low_get_min(start_index, finish_index):
    min_tmp = float(data_1m[start_index][3])
    for x in range(finish_index - start_index):
        min_tmp = min(min_tmp, float(data_1m[start_index + x][3]))
    return min_tmp


def vol_get_sum(start_index, finish_index):
    sum = 0
    for x in range(finish_index - start_index):
        sum += float(data_1m[start_index + x][5])
    return sum


def calculate_ma_close(data, start, finish, period):
    MA_data = talib.MA(np.array(data), timeperiod=period)
    return MA_data


def csvwerk(start, finish, candles):
    global long_count, short_count, iter, correction

    backward = 150  # 60min
    timedelta_1st = 5  # min
    timedelta_2nd = 15  # min
    candles_1st = 24  # candles
    candles_2nd = 12  # candles
    open, high, low, close, volume, date = [], [], [], [], [], []

    tmp = start
    for x in range(candles):  # finish - start
        open.append(float(data_1m[tmp][1]))
        high.append(float(data_1m[tmp][2]))
        low.append(float(data_1m[tmp][3]))
        close.append(float(data_1m[tmp][4]))
        volume.append(float(data_1m[tmp][5]))
        date.append(data_1m[tmp][0])
        tmp += 1

    start = finish - candles_1st
    percentage = 3  # 1% 상승/하락
    candle_start = 0
    candle_finish = candles - 1
    diff_start_finish = close[candle_start] - close[candle_finish]
    if ((close[candle_start] * percentage) / 100 < abs(diff_start_finish)
            and ((close[0] < close[candles - 1]) or (close[0] > close[candles - 1]))):
        if (diff_start_finish > 0):  # 양수면 숏
            draw_graph('short', start - backward, finish - backward, candles_1st, candles_2nd, timedelta_1st,
                       timedelta_2nd)
            print("Date:", date[0], " Short:", diff_start_finish)
            short_count += 1
        else:  # 음수면 롱
            draw_graph('long', start - backward, finish - backward, candles_1st, candles_2nd, timedelta_1st,
                       timedelta_2nd)
            print("Date:", date[0], " Long:", diff_start_finish)
            long_count += 1
        time = date[0].split(':')
        k = int(time[1]) % 5
        if (k != 0):
            if (k == 1):
                iter -= 1
                correction += 1
            elif (k == 2):
                iter -= 2
                correction += 1
            elif (k == 3):
                iter -= 3
                correction += 1
            elif (k == 4):
                iter -= 4
                correction += 1
        print(long_count, short_count, correction)


def draw_graph(position, start, finish, candles_1st, candles_2nd, timedelta_1st, timedelta_2nd):
    open_1st, high_1st, low_1st, close_1st, volume_1st, date_1st, pd_close_1st = custom_resmapler( \
        data_1m, start - abs((finish - start) - candles_1st * timedelta_1st) + 1, finish, timedelta_1st)

    open_2nd, high_2nd, low_2nd, close_2nd, volume_2nd, date_2nd, pd_close_2nd = custom_resmapler( \
        data_1m, start - abs((finish - start) - candles_2nd * timedelta_2nd) + 1, finish, timedelta_2nd)

    if (len(open_1st) == 0 or len(open_2nd) == 0):
        return

    start_1m = start
    finish_1m = finish
    start_1st = start // timedelta_1st
    finish_1st = finish // timedelta_1st
    start_2nd = start // timedelta_2nd
    finish_2nd = finish // timedelta_2nd

    tmp = start_1m
    date_1m = []
    for x in range(finish_1m - start_1m):
        date_1m.append(data_1m[tmp][0])
        tmp += 1
    tmp = start_1m - 100

    MA_7_1st = calculate_ma_close(pd_close_1st, start_1st, finish_1st, 7)
    MA_25_1st = calculate_ma_close(pd_close_1st, start_1st, finish_1st, 25)
    MA_99_1st = calculate_ma_close(pd_close_1st, start_1st, finish_1st, 99)

    MA_7_2nd = calculate_ma_close(pd_close_2nd, start_2nd, finish_2nd, 7)
    MA_25_2nd = calculate_ma_close(pd_close_2nd, start_2nd, finish_2nd, 25)
    MA_99_2nd = calculate_ma_close(pd_close_2nd, start_2nd, finish_2nd, 99)

    fig = plt.figure(num=1, figsize=(7.5, 7.55), dpi=50, facecolor='w', edgecolor='k')
    gs = GridSpec(nrows=10, ncols=1)
    cx_2nd = fig.add_subplot(gs[0:4, 0])
    vx_2nd = fig.add_subplot(gs[4, 0])
    cx_1st = fig.add_subplot(gs[5:9, 0])
    vx_1st = fig.add_subplot(gs[9, 0])

    mpl_finance.volume_overlay(vx_2nd, open_2nd, close_2nd, volume_2nd, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_2nd, open_2nd, close_2nd, high_2nd, low_2nd, width=0.965, colorup='r',
                                  colordown='b', alpha=1)
    mpl_finance.volume_overlay(vx_1st, open_1st, close_1st, volume_1st, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(cx_1st, open_1st, close_1st, high_1st, low_1st, width=0.965, colorup='r',
                                  colordown='b', alpha=1)
    plt.autoscale()
    line_width = 6
    cx_2nd.plot(MA_7_2nd[99:], color='gold', linewidth=line_width, alpha=1)
    cx_2nd.plot(MA_25_2nd[99:], color='violet', linewidth=line_width, alpha=1)
    cx_2nd.plot(MA_99_2nd[99:], color='green', linewidth=line_width, alpha=1)
    cx_1st.plot(MA_7_1st[99:], color='gold', linewidth=line_width, alpha=1)
    cx_1st.plot(MA_25_1st[99:], color='violet', linewidth=line_width, alpha=1)
    cx_1st.plot(MA_99_1st[99:], color='green', linewidth=line_width, alpha=1)
    plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
    vx_2nd.axis('off')
    cx_2nd.axis('off')
    vx_1st.axis('off')
    cx_1st.axis('off')
    fig_name = date_1m[finish - start - 1].split(':')
    #fig_name = date_1m[0].split(':')
    if (position == 'long'):
        fig_name = long_dir + str(fig_name[0]) + '_' + str(fig_name[1]) + '.jpg'
    elif (position == 'short'):
        fig_name = short_dir + str(fig_name[0]) + '_' + str(fig_name[1]) + '.jpg'
    elif(position == 'neutral'):
        fig_name = neutral_dir + str(fig_name[0]) + '_' + str(fig_name[1]) + '.jpg'
    plt.savefig(fig_name, bbox_inches='tight')  # uuid.uuid4()
    # plt.show()
    plt.cla()  # 좌표축을 지운다.
    plt.clf()  # 현재 Figure를 지운다.


if __name__ == "__main__":
    iter = 1000  # start point
    scan_candles = 180  # 3 hours
    skip = 5
    for x in range(len(data_1m) - scan_candles - 1):  #
        csvwerk(iter, iter + scan_candles, scan_candles)  # 5 -> 검사하는 봉 개수
        iter = iter + skip
