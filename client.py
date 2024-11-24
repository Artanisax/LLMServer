import socket
import json

def main():
    HOST = '127.0.0.1'
    PORT = 8899
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    send = json.dumps({
        "type": "chat",
        "value": {
            "query": "Hello!",
            "history": None,
        }
    })
    print(send)
    client_socket.send(send.encode())
    recv = ""
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        recv += data.decode()
    print(recv)
    client_socket.close()
    print("Close Connection")


if __name__ == "__main__":
    main()