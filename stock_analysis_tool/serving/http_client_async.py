import aiohttp
import asyncio


async def fetch(session, url):
    async with session.get(url) as response:
        print(response.status)
        return await response.json()


async def main():
    async with aiohttp.ClientSession() as session:
        html = await fetch(session, 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=MSFT&apikey=demo')
        print(html)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
