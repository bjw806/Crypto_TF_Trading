from numpy import MAY_SHARE_BOUNDS, genfromtxt
import matplotlib.pyplot as plt
import mpl_finance
import numpy as np
import pandas
import uuid
from matplotlib.gridspec import GridSpec
from pandas.core.base import DataError
from pandas.core.frame import DataFrame
import talib
import ccxt

data_1m = genfromtxt('../financial_data/Binance_BTCUSDT_minute.csv', delimiter=',' ,dtype=str)

long_dir = '../test_data/long/'
short_dir = '../test_data/short/'

def custom_resmapler(np, start, finish, timedelta):
    open,high,low,close,volume,date = [],[],[],[],[],[]
    pd_close = []
    t = start - timedelta #한봉 뒤부터 시작
    k = 0
    index = 0
    for x in range(4):
        time =data_1m[t][0]
        time = time.split(':')
        time = int(time[1])
        if(time % timedelta == 0):
            index = t
            k = index #k에 index를 저장해놓음
            break
        t += 1
    if(index==0):
        return [],[],[],[],[],[],[]
    for x in range((finish - start)//timedelta):#start(이미 보정된) ~ finish까지 5분봉. 마지막 빼고
        date.append(data_1m[index][0])
        open.append(float(data_1m[index][1])) #first
        high.append(high_get_max(index, index+timedelta-1))
        low.append(low_get_min(index, index+timedelta-1))
        close.append(float(data_1m[index+timedelta-1][4])) #last
        volume.append(vol_get_sum(index, index+timedelta-1))# 볼륨: 5개 더함
        index += timedelta
    date.append(data_1m[index+1][0])
    open.append(float(data_1m[index+1][1])) #first
    high.append(high_get_max(index+1, finish-1))
    low.append(low_get_min(index+1, finish-1))
    close.append(float(data_1m[finish-1][4])) #last
    volume.append(vol_get_sum(index+1, finish-1))# 볼륨: 5개 더함

    index = k - 100*timedelta #close를 MA99 구하기 위해 -100칸 전부터 따로 모은다
    for x in range((finish - start + 100*timedelta)//timedelta):
        pd_close.append(float(data_1m[index+timedelta-1][4]))
        index += timedelta
    return open, high, low, close, volume, date, pd_close
#마지막(진행중인 캔들)에는 남은 캔들이 % == 0 이 아니라도 추가하는 기능: 예정
def high_get_max(start_index, finish_index):
    max_tmp = float(data_1m[start_index][2])
    for x in range(finish_index - start_index):
        max_tmp = max(max_tmp, float(data_1m[start_index+x][2]))
    return max_tmp

def low_get_min(start_index, finish_index):
    min_tmp = float(data_1m[start_index][3])
    for x in range(finish_index - start_index):
        min_tmp = min(min_tmp, float(data_1m[start_index+x][3]))
    return min_tmp

def vol_get_sum(start_index, finish_index):
    sum = 0
    for x in range(finish_index - start_index):
        sum += float(data_1m[start_index+x][5])
    return sum

def calculate_ma(start, finish, period):
    close = []
    t = start - 100
    for x in range(finish - start + period):
        close.append(float(data_1m[t][4]))
        t = t + 1

    df = talib.MA(np.array(close), timeperiod=period)
    return df

def calculate_ma_close(data, start, finish, period):
    #close = []
    #t = start - 100
    #for x in range(finish - start + period):
    #    close.append(float(pd[t][4]))
    #    t = t + 1

    MA_data = talib.MA(np.array(data), timeperiod=period)
    return MA_data

def csvwerk(start, finish, candles):
    backward = 3 #3
    timedelta_x_min = 5
    candles_1m = 15 
    candles_5m = timedelta_x_min * 10 #time x candles
    open,high,low,close,volume,date = [],[],[],[],[],[]
    
    tmp = start
    for x in range(candles):#finish - start
        open.append(float(data_1m[tmp][1]))
        high.append(float(data_1m[tmp][2]))
        low.append(float(data_1m[tmp][3]))
        close.append(float(data_1m[tmp][4]))
        volume.append(float(data_1m[tmp][5]))
        date.append(data_1m[tmp][0])
        tmp += 1
    
    start = finish - candles_1m
    percentage = 1 #x percent
    diff_start = 0
    diff_finish = candles - 1
    difference = close[diff_start] - close[diff_finish]
    if ((close[diff_start]*percentage)/100 < abs(difference) 
        and ((close[0] < close[candles-1]) or (close[0] > close[candles-1]))):
        if (difference > 0):  # 양수면 숏
            draw_graph('short',start-backward, finish-backward, candles_5m, timedelta_x_min)
            #shortgraph(finish-(backward+graph1_candles), finish-backward, finish-(graph2_candles+backward), finish-backward)
            print("Date:", date[0], " Short:", difference)
        else:  # 음수면 롱
            draw_graph('long',start-backward, finish-backward, candles_5m, timedelta_x_min)
            print("Date:", date[0], " Long:", difference)

#그래프 그리는 함수를 하나로 합치기: ok
def draw_graph(position, start, finish, candles_5min, timedelta_x_min):
    custom_open,custom_high,custom_low,custom_close,custom_volume,custom_date,pd_close_5m = custom_resmapler(
        data_1m, start-abs((finish-start)-candles_5min)+1, finish, timedelta_x_min)
    if(len(custom_open)==0):
        return
    #1min 캔들 개수 정확히 표시: ok
    #5min '' : ok
    #backward: ok
    start_1m = start
    finish_1m = finish
    start_5m = (start//5)#-10 # 봉 보정 개수
    finish_5m = (finish//5)# +1

    tmp = start_1m
    open_1m,high_1m,low_1m,close_1m,volume_1m,date_1m,pd_close_1m = [],[],[],[],[],[],[]
    for x in range(finish_1m - start_1m):
        open_1m.append(float(data_1m[tmp][1]))
        high_1m.append(float(data_1m[tmp][2]))
        low_1m.append(float(data_1m[tmp][3]))
        close_1m.append(float(data_1m[tmp][4]))
        volume_1m.append(float(data_1m[tmp][5]))
        date_1m.append(data_1m[tmp][0])
        tmp += 1
    tmp = start_1m - 100
    for x in range(finish - start + 100):
        pd_close_1m.append(float(data_1m[tmp][4]))
        tmp += 1

    MA_7_1m = calculate_ma_close(pd_close_1m,start_1m, finish_1m, 7)
    MA_25_1m = calculate_ma_close(pd_close_1m,start_1m, finish_1m, 25)
    MA_99_1m = calculate_ma_close(pd_close_1m,start_1m, finish_1m, 99)
    MA_7_5m = calculate_ma_close(pd_close_5m, start_5m, finish_5m, 7)
    MA_25_5m = calculate_ma_close(pd_close_5m, start_5m, finish_5m, 25)
    MA_99_5m = calculate_ma_close(pd_close_5m, start_5m, finish_5m, 99)

    fig = plt.figure(num=1, figsize=(5, 5), dpi=100, facecolor='w', edgecolor='k') # figsize: ppi dpi: 해상도
    gs = GridSpec(nrows=12, ncols=1)######비율
    dx = fig.add_subplot(gs[0:5, 0]) #111은 subplot 그리드 인자를 정수 하나에 다 모아서 표현한 것.(1x1그리드에 첫 번째 subplot)
    ax = fig.add_subplot(gs[5, 0]) #볼륨차트 추가
    bx = fig.add_subplot(gs[6:11, 0])
    cx = fig.add_subplot(gs[11, 0])
    mpl_finance.volume_overlay(ax, custom_open, custom_close, custom_volume, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(dx, custom_open,custom_close,custom_high,custom_low, width=0.965, colorup='r', colordown='b', alpha=1)
    mpl_finance.volume_overlay(cx, open_1m, close_1m, volume_1m, width=0.4, colorup='r', colordown='b', alpha=1)
    mpl_finance.candlestick2_ochl(bx, open_1m, close_1m, high_1m, low_1m, width=0.965, colorup='r', colordown='b', alpha=1)
    plt.autoscale() #자동 스케일링
    dx.plot(MA_7_5m[99:], color='gold', linewidth=1, alpha=1)#99+~이니까 99부터 시작
    dx.plot(MA_25_5m[99:], color='violet', linewidth=1, alpha=1)
    dx.plot(MA_99_5m[99:], color='green', linewidth=1, alpha=1)
    bx.plot(MA_7_1m[99:], color='gold', linewidth=1.5, alpha=1)
    bx.plot(MA_25_1m[99:], color='violet', linewidth=1.5, alpha=1)
    bx.plot(MA_99_1m[99:], color='green', linewidth=1.5, alpha=1)
    plt.axis('off')  # 상하좌우 축과 라벨 모두 제거
    ax.axis('off')
    dx.axis('off')
    bx.axis('off')
    cx.axis('off')
    fig_name = date_1m[finish-start-1].split(':')
    if(position=='long'):
        fig_name = long_dir + str(fig_name[0])+'_'+str(fig_name[1]) + '.jpg'
    elif(position=='short'):
        fig_name = short_dir + str(fig_name[0])+'_'+str(fig_name[1]) + '.jpg'
    plt.savefig(fig_name, bbox_inches='tight')#uuid.uuid4()
    #plt.show()
    plt.cla() #좌표축을 지운다.
    plt.clf() #현재 Figure를 지운다.

if __name__ == "__main__":
    iter = 0
    scan_candles = 5
    skip = 1
    for x in range(len(data_1m)-scan_candles-1):#
        csvwerk(iter, iter+scan_candles, scan_candles) #5 -> 검사하는 봉 개수
        iter = iter + skip