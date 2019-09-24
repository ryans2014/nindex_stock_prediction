import asyncio
from serving.mongodb_async_wrapper import AsyncResultDocument
from models import TensorflowProduction


tf_production = TensorflowProduction()


async def get_result(symbol: str) -> str:
    """ Call database fetcher or TF server to get results """

    # fetch from mongodb
    jdoc = AsyncResultDocument(symbol)
    await jdoc.fetch()
    jdoc.update()
    if not jdoc.need_update_csv():
        return jdoc.get_csv()

    # update csv by calling the full pipline
    csv_string = await tf_production.predict(symbol, 5)
    jdoc.set_csv(csv_string)
    await jdoc.push()
    return csv_string
