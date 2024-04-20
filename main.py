
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from ai import AI

load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

intents = discord.Intents.default()

bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    for guild in bot.guilds:
        print(f"Guild Name: {guild.name}")


@bot.command(name="llm")
async def daisy_text(ctx):
    ai = AI()
    prompt = ctx.message.content.replace('!llm', '')
    response = ai.infer_text(prompt)
    await ctx.send(response)


@bot.command(name="image")
async def daisy_image(ctx):
    message = await ctx.send("Generating Image...")
    ai = AI()
    prompt = ctx.message.content.replace('!image', '')
    image = await ai.infer_text_image(prompt)
    file = discord.File(fp=image, filename="generated.png")
    await ctx.send(content=f"Prompt: {prompt}", file=file)

bot.run(TOKEN)

