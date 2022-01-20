import discord, asyncio, os
from discord.ext import commands

game = discord.Game("돈 복사") #xxxx 하는중
bot = commands.Bot(command_prefix='!', status=discord.Status.online, activity=game)

bot.run('여기에 토큰을 입력')
applcation id
879193737689980939
client key
oUtSbLq_5FymCKZE4xfykpQ_hC8aUIFq
public key
b50ce0c4e9fb49cc75cb192f81aca6811da65e97dd594c6c5c332c38f9b3ea8a

url
https://discord.com/api/oauth2/authorize?client_id=879193737689980939&permissions=8&scope=bot