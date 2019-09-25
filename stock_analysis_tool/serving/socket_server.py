import asyncio
from motor import motor_asyncio
import socket
import utility
import logging
from serving.util import prefixed_socket_async_recv, prefixed_socket_async_send
from serving.fetch_result import get_result
from serving.mongodb_async_wrapper import connect


num_tasks = 0
max_tasks = 500


async def socket_request_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    global num_tasks, max_tasks

    if num_tasks >= max_tasks:
        logging.info("In request_handler, discarded request due to max_tasks limit.")
        try:
            await prefixed_socket_async_send(writer, "max_tasks_error")
        except:
            pass
        writer.close()
        return

    num_tasks += 1
    try:
        symbol = await prefixed_socket_async_recv(reader)
        logging.info("In request_handler, handle task %s." % symbol)
        csv_string = await get_result(symbol)
    except Exception as e:
        logging.error(e)
        csv_string = "ERROR"
    await prefixed_socket_async_send(writer, csv_string)
    writer.close()
    num_tasks -= 1


async def socket_main():
    server = await asyncio.start_server(socket_request_handler, host='127.0.0.1', port=1001)
    await server.serve_forever()


def run_socket_server():
    # init: connect to mongodb
    connect()

    # start socket server
    loop = asyncio.get_event_loop()
    loop.run_until_complete(socket_main())


if __name__ == "__main__":
    run_socket_server()
