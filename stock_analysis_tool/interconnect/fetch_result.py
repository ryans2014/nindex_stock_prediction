import asyncio
from interconnect.mongodb_async_wrapper import AsyncResultDocument


async def get_result(symbol: str) -> str:
    """ Call database fetcher or TF server to get results """

    # fetch from mongodb
    jdoc = AsyncResultDocument(symbol)
    await jdoc.fetch()
    jdoc.update()
    if not jdoc.need_update_csv():
        return jdoc.get_csv()

    # update by calling the full pipline
    # call tf server
    csv_string = ""
    jdoc.set_csv(csv_string)
    return csv_string
