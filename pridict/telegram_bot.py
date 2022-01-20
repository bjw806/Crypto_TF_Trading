import telegram
import datetime
from telegram.ext import Updater, CommandHandler 
import ccxt_live_v4_optimized

telegram_bot_token = "Your-token"
telegram_chat_id = "Your-chat-id"
updater = Updater(token=telegram_bot_token)#, use_context=True) 
dispatcher = updater.dispatcher
bot = telegram.Bot(token = telegram_bot_token)

def send_message(input_text):
    bot.sendMessage(chat_id = telegram_chat_id, text = input_text)

#전송해야하는 요소: 시간, long/short open/close, 이익
def send_trading_message(price, cost, LS, OC, amount, profit, total_profit):
    time = datetime.datetime.now()
    time = time.strftime('%Y-%m-%d %H:%M:%S')
    text = "Time: "+time+"\r\nType: "+LS+" "+OC+"\r\nAmount: "+ str(amount) + " BTC\r\nPrice: "+str(price)+" USDT\r\nCost: "+str(cost)+" USDT"
    if(OC == "Close"):
        text = text + "\r\nProfit: "+str(profit)+" USDT\r\nTotal Profit: "+str(total_profit)+" USDT"
    send_message(text)

def send_info(update, context):
    str = ("==========| My Info |==========" +
    "Assets:" + ccxt_live_v4_optimized.assets +
    "My Position:" +  ccxt_live_v4_optimized.my_position +
    "Total_profit:" + ccxt_live_v4_optimized.total_profit + " USDT" +
    "Present Margin:" + ccxt_live_v4_optimized.margin + " USDT" +
    "Wins:" + ccxt_live_v4_optimized.win_count +
    "Loses:" + ccxt_live_v4_optimized.lose_count +
    "Winrate:" + ccxt_live_v4_optimized.winrate + " %")
    context.bot.send_message(chat_id=update.effective_chat.id, text=str)

start_handler = CommandHandler('info', send_info)
dispatcher.add_handler(start_handler)
updater.start_polling() 
updater.idle()



#if __name__ == "__main__":
#    send_trading_message("long","close", 0.005, 2.45)
