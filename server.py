import socket
import json
import os
from LLM.llm import LLM


class LLMServer():
    def __init__(self):
        self.llm = None
        # self.load("THUDM/chatglm3-6b")
        pass
            
    def socket(self):
        HOST = '127.0.0.1'
        PORT = 8899
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        while True:
            print("Try Connection")
            try:
                conn, address = server_socket.accept()
                print(conn, '\n', address)
                # conn.settimeout(5)
                
                recv = ""
                while True:
                    data = conn.recv(1024)
                    print({"data": data})
                    if not data:
                        break
                    recv += data.decode()
                    if recv[-2:] == "\r\n":
                        recv = recv[:-2]
                        break
                conn.shutdown(socket.SHUT_RD)
                print("recv:\n", recv)
                
                send = self.parse(recv)
                print("send:", send)
                conn.send(json.dumps(send).encode())
                
                conn.close()
                print("Close Connection")
        
            except Exception as err:
                print(err)
                continue
    
    
    def upload(self, value):
        pass


    def delete(self, id):
        if os.path.exists(f"models/{id}"):
            os.system(f"rm -rf models/{id}")
            msg = "Succesfully deleted"
        else:
            msg = "Not existed"
        return {
            "type": "message",
            "value": msg,
        }


    def load(self, id):
        self.llm = LLM(id)
        return {
            "type": "message",
            "value": "Successfully load model.",
        }


    def unload(self, value: None):
        del self.llm
        self.llm = None
        return {
            "type": "message",
            "value": "Successful unload model.",
        }


    def chat(self, value: dict):
        # return {
        #     "type": "message",
        #     "value": "Let's chat!",
        # }
        query, history = value["query"], None #, value["history"]
        return {
            "type": "chat",
            "value": self.llm.chat(query, history),
        }


    def parse(self, recv: str):
        print("parsing")
        operate = {
            "upload": self.upload,
            "delete": self.delete,
            "load": self.load,
            "unload": self.unload,
            "chat": self.chat,
            
        }
        recv = json.loads(recv)
        return operate[recv["type"]](recv["value"])
    

def main():
    server = LLMServer()
    server.socket()


if __name__ == "__main__":
    main()
