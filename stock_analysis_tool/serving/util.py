from datetime import datetime, timedelta
import socket
import asyncio


def previous_close_utc_time() -> datetime:
    """
    :return: previosu trading close time (assume 4:30pm close at EAT) (return tiem is in UTC)
    """
    def utc(est: datetime) -> datetime:
        return est + timedelta(hours=4)

    est_now = datetime.utcnow() - timedelta(hours=4)
    today_close = datetime(year=est_now.year,
                           month=est_now.month,
                           day=est_now.day,
                           hour=16,
                           minute=30)
    if est_now > today_close and est_now.weekday() <= 5:
        return utc(today_close)

    previous_close = today_close - timedelta(hours=24)
    while previous_close.weekday() > 5:
        previous_close = previous_close - timedelta(hours=24)
    return utc(previous_close)


def prefixed_socket_send(skt: socket.socket, s: str) -> None:

    def prefix_length_6char(symbol: str) -> str:
        slen = '000000' + str(len(symbol))
        return slen[-6:]

    prefix = prefix_length_6char(s)
    skt.send((prefix + s).encode())


def prefixed_socket_recv(skt: socket.socket) -> str:

    def get_data_fix_length(n: int) -> str:
        byte_list = []
        len_left = n
        while len_left > 0:
            seg = skt.recv(len_left)
            if not seg:
                raise ValueError("Message is shorter than 6.")
            byte_list.append(seg)
            len_left -= len(seg)
        return b''.join(byte_list).decode()

    # receive 6 characters for data length
    data_length = int(get_data_fix_length(6))

    # receive data
    data = get_data_fix_length(data_length)

    return data


async def prefixed_socket_async_send(writer: asyncio.StreamWriter, s: str) -> None:

    def prefix_length_6char(symbol: str) -> str:
        slen = '000000' + str(len(symbol))
        return slen[-6:]

    prefix = prefix_length_6char(s)
    writer.write((prefix + s).encode())
    await writer.drain()


async def prefixed_socket_async_recv(reader: asyncio.StreamReader) -> str:

    async def get_data_fix_length(n: int) -> str:
        byte_list = []
        len_left = n
        while len_left > 0:
            seg = await reader.read(len_left)
            if not seg:
                raise ValueError("Message is shorter than 6.")
            byte_list.append(seg)
            len_left -= len(seg)
        return b''.join(byte_list).decode()

    # receive 6 characters for data length
    data_length = int(await get_data_fix_length(6))

    # receive data
    data = await get_data_fix_length(data_length)

    return data
