import time
import socket

host = "127.0.0.1"
port = 5002

import struct


if __name__ == '__main__':
    import urllib.request
    import socket

    # 获取文件
    file_url = "http://cdn.minguncle.top/img/202212121417982.pcm"
    response = urllib.request.urlopen(file_url)
    file_data = response.read()

    # 创建 TCP 连接
#     sock.close()
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s :
#         try :
#             s.connect((host, port))
#             s.sendall(b'Hello world')
#             #循环发送文件
#             chunk_size = 1280
#             chunk_time = 0.04  # 40ms
#             for i in range(0, len(file_data), chunk_size):
#                 chunk = file_data[i:i + chunk_size]
#                 s.send(chunk)
#                 time.sleep(chunk_time)
#             data = s.recv(1024) # 阻塞，直到对方 socket 关闭
#         except OSError as err :
#             print(err)
#         else :
#             print("Received : ", repr(data))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('127.0.0.1', 5002)
    sock.connect(server_address)

    # 循环发送文件
    chunk_size = 1024
    chunk_time = 0.04  # 40ms
    for i in range(0, len(file_data), chunk_size):
        chunk = file_data[i:i + chunk_size]
        sock.send(chunk)
        res = sock.recv(1024)
        data = struct.unpack('<s', res)[0]
        print(data)
        time.sleep(chunk_time)
    # 关闭连接
    sock.close()




