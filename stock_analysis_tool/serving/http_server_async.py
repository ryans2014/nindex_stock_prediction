from aiohttp import web, web_request
import asyncio
import logging
from serving.fetch_result import get_result
from serving.mongodb_async_wrapper import connect

HTTP_ERROR_STRING = "ERROR"


async def http_request_handler(request: web_request.Request) -> web.Response:
    symbol = request.match_info.get('symbol', "error")
    logging.info("In http_request_handler, handle task %s." % symbol)

    # if no symbol in url,
    if symbol == "error":
        return web.Response(text=HTTP_ERROR_STRING)

    # get or predict
    try:
        csv_string = await get_result(symbol)
    except Exception as e:
        logging.error(e)
        csv_string = HTTP_ERROR_STRING

    return web.Response(text=csv_string)


def run_http_server():
    # init: connect to mongodb
    connect()
    # run http server
    app = web.Application()
    app.add_routes([web.get('/', http_request_handler), web.get('/result/csv/{symbol}', http_request_handler)])
    web.run_app(app)


if __name__ == "__main__":
    run_http_server()
