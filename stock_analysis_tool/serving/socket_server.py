import asyncio
import socket
from serving.util import prefixed_socket_async_recv, prefixed_socket_async_send
from serving.fetch_result import get_result


num_tasks = 0
max_tasks = 500


async def request_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    global num_tasks, max_tasks
    if num_tasks >= max_tasks:
        await prefixed_socket_async_send(writer, "max_tasks_error")
        writer.close()
        return

    symbol = await prefixed_socket_async_recv(reader)
    num_tasks += 1
    csv_string = await get_result(symbol)
    await prefixed_socket_async_send(writer, csv_string)
    writer.close()
    num_tasks -= 1


async def socket_main():
    server = await asyncio.start_server(request_handler, host='127.0.0.1', port=1001)
    await server.serve_forever()
