import discord
import json
import logging
import inspect
from discord.ext import commands
from wheatley import Wheatley

logging.basicConfig(level=logging.INFO)

with open('./config/auth.json') as data_file:
    auth = json.load(data_file)

bot = commands.Bot(command_prefix=commands.when_mentioned_or(
    '!'), description='Wheatley, yay!')
wheatley = Wheatley(bot)
bot.add_cog(wheatley)


@bot.event
async def on_ready():
    logging.info('Logged in as:{0} (ID: {0.id})'.format(bot.user))


@bot.event
async def on_message(message):
    if (message.author.id != bot.user.id and not message.author.bot):
        await wheatley.talk(message)

bot.run(auth['token'])
wheatley.shutdown()
