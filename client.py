import socket
import json

HOST = "127.0.0.1"
PORT = 8082
CHAT = {
    "type": "chat",
    "value": {
        "base": "Llama_3_2_Instruct",
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "lora": None,
        "query": "Hello!",
        "history": None,
    },
}
FINETUNE = {
    "type": "finetune",
    "value": {
        "base": "Gemma_2",
        "id": "google/gemma-2-2b",
        "dataset": "Abirate/english_quotes",
        "name": "test",
        "steps": 8,
    },
}

TEST_CASE = CHAT

def main():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))
    send = json.dumps(TEST_CASE)
    print("<client send>\n", send)
    client_socket.send(send.encode() + b"\r\n")
    recv = ""
    while True:
        data = client_socket.recv(1024)
        if not data:
            break
        recv += data.decode()
        if data[-2:] == b"\r\n":
            recv = recv[:-2]
            break
        recv += data.decode()
    print("<client recv>\n", recv)
    client_socket.close()
    print("Close Connection")


if __name__ == "__main__":
    main()
