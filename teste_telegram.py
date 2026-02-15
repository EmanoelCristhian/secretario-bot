import asyncio
from aiogram import Bot

TOKEN = "8034880624:AAHJoKfUeYYcAZyV-LtKb2wlmOw4TB2SdMs"

async def main():
    bot = Bot(token=TOKEN)
    me = await bot.get_me()
    print(f"Bot conectado com sucesso!")
    print(f"Nome do Bot: {me.first_name}")
    print(f"Username: @{me.username}")
    await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())