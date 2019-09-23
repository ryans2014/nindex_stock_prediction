from serving.socket_server import socket_main
import serving.mongodb_async_wrapper as db_wrapper
from multiprocessing import Queue, Process
from motor import motor_asyncio
import asyncio


def run_server():
    # init: connect to mongodb
    client = motor_asyncio.AsyncIOMotorClient(
        "mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
    db = client.histresult
    db_wrapper.db_collection = db.results

    # start socket server
    loop = asyncio.get_event_loop()
    loop.run_until_complete(socket_main())


if __name__ == "__main__":
    run_server()

