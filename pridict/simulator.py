


s_assets = 1000 #USDT
s_profit = 0
s_total_profit = 0
s_leverage = 10
s_my_position = 'n/a'
s_position_price = 0
s_cost = 0
s_have_position = False


def open_long(price, amt):
    global s_assets
    global s_my_position
    global s_have_position
    global s_position_price
    global s_cost

    s_cost = (price*amt)
    s_my_position = 'long'
    s_have_position = True
    s_position_price = price
    s_assets -= s_cost

def open_short(price, amt):
    global s_assets
    global s_my_position
    global s_have_position
    global s_position_price
    global s_cost

    s_cost = (price*amt)
    s_my_position = 'short'
    s_have_position = True
    s_position_price = price
    s_assets -= s_cost

def close_long(price, amt):
    global s_assets
    global s_my_position
    global s_have_position
    global s_position_price
    global s_cost
    global s_total_profit

    s_cost = 0
    s_my_position = 'n/a'
    s_have_position = False
    s_position_price = 0
    profit = (price*amt*s_leverage) - s_cost
    s_total_profit += profit
    s_assets += profit

def close_short(price, amt):
    global s_assets
    global s_my_position
    global s_have_position
    global s_position_price
    global s_cost
    global s_total_profit

    s_cost = (price*amt)
    s_my_position = 'n/a'
    s_have_position = False
    s_position_price = 0
    profit = s_cost - (price*amt*s_leverage)
    s_total_profit += profit
    s_assets += profit