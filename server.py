#导入socket模块
import socket
import struct
from speech_predict import lstm_predict
# #创建socket，类似于买手机
# skt=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# #绑定端口和ip，必须为元组类型，类似于手机插卡
# #,如果元组为('',9000)表示本机所有的网卡，相当于0.0.0.0
# skt.bind(('127.0.0.1', 5002))
# #侦听访问端口，类似于手机待机
# #若括号中有值，则表示对TCP连接的优化
# skt.listen()
# #此处循环表示服务器持续提供服务
# while True:
#     #conn表示接受的数据流，addr表示客户端的地址
#     conn,addr=skt.accept()
#     buf = bytes()
#     while True:
#         #接受客户端发送消息并打印
#         msg=conn.recv(1280)
#         # print(msg.decode('utf-8'))
#         print(msg, type(msg))
#         print(len(msg))
#         lstm_predict(msg)
#         buf += msg
#         print(len(buf))
#         #为客户端返回消息，表示接受成功
#         # conn.send(msg.upper())
#         if not msg:
#             break
# #关闭本次通信
# conn.close()
# #关闭链接
# skt.close()

import socketserver
import redis
import json

def get_json(key):
    r = redis.StrictRedis(host='172.18.166.156', port=6379, password='Shaue_redis_prd@2020', db=0, decode_responses=True)
    # if key == 'companyList':
    #     cl = r.smembers(key)
    #     return cl
    g_json = r.get(key)
    js = json.loads(g_json)
    return js

class MyServer(socketserver.BaseRequestHandler):
    def handle(self):
        # print(self.request, self.client_address,self.server)
        ip, port = self.client_address
        key = ip + str(port)
        print(key)
        conn = self.request
        Flag = True
        qus_now = -1
        buf = bytes()
        i = 0
        while Flag:
            # js = get_json(key)
            # 构建假的redis数据
            js = {'model_id': 1, 'qus_id': i}
            model_id = js['model_id']
            qus_id = js['qus_id']
            qus_now = qus_id if qus_now == -1 else qus_now
            data = conn.recv(1024)
            if not data:
                Flag = False
            else:
                if qus_now == qus_id:
                    print(data, type(data))
                    buf += data
                else:
                    print(qus_now, len(buf))
                    res = lstm_predict(buf).argmax()
                    res = str(model_id) + '_' + str(qus_now) + '_' + str(res)
                    res = struct.pack('<s', res)
                    print(res, type(res))
                    conn.send(res)
                    qus_now = qus_id
            i += 1


if __name__ == '__main__':
    server = socketserver.ThreadingTCPServer(('127.0.0.1', 5002),MyServer)
    server.serve_forever()