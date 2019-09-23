import socket
from serving.util import prefixed_socket_recv, prefixed_socket_send
import threading

"""
    Build socket connection to socket_server
    For testing purpose
"""


def request(symbol: str):
    # send request to
    skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    skt.connect(('127.0.0.1', 1001))
    prefixed_socket_send(skt, symbol)
    data = prefixed_socket_recv(skt)
    print(data)
    skt.close()


if __name__ == "__main__":
    t1 = threading.Thread(target=request, args=("B1",))
    t2 = threading.Thread(target=request, args=("B2",))
    t3 = threading.Thread(target=request, args=("B3",))
    t1.start()
    t2.start()
    t3.start()
    t1.join()
    t2.join()
    t3.join()
