import asyncio


async def f1():
    print("f1 - 1")
    await asyncio.sleep(1.0)
    print("f1 - 2")


async def f2():
    print("f2 - 1")
    await asyncio.sleep(1.0)
    print("f2 - 2")


async def main():
    await asyncio.gather(f1(), f2())


asyncio.run(main())
