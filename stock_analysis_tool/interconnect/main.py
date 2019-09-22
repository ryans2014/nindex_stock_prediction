from interconnect.socket_server import socket_main
import interconnect.mongodb_async_wrapper as db_wrapper


def real_main():
    # init: connect to mongodb
    client = motor_asyncio.AsyncIOMotorClient(
        "mongodb+srv://ryansu2011:susu1021@hispredict-dbzhi.mongodb.net/test?retryWrites=true&w=majority")
    db = client.histresult
    db_wrapper.db_collection = db.results

    # init: connect to tensorflow

    # start socket server
    loop = asyncio.get_event_loop()
    loop.run_until_complete(socket_main())

    # deal with process termination
    # add time to live
    # add logging


if __name__ == "__main__":
    real_main()

