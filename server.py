import os
import json
import socket
import threading

from util import create_model

HOST = "localhost"
PORT = 8082


def handle(conn, address):
    recv = ""
    while True:
        data = conn.recv(1024)
        if not data:
            break
        recv += data.decode()
        if data[-2:] == b"\r\n":
            recv = recv[:-2]
            break
    
    print("<server recv>\n", recv)
    send = parse(recv)
    print("<server send>\n", send)
    conn.send(json.dumps(send).encode() + b"\r\n")
    conn.close()


def upload(value: str):
    print("upload")
    model = json.loads(value)
    id = model["id"]
    zip_file = model["zip_file"]
    if os.path.exists(f"models/{id}"):
        msg = "Model already exists!"
    else:
        os.makedirs(f"models/{id}")
        with open(f"models/{id}.zip", "wb") as file:
            file.write(zip_file)
        os.system()
        msg = "Succesfully uploaded."
    return {
        "type": "message",
        "value": msg,
    }


def delete(id):
    print("delete")
    if os.path.exists(f"models/{id}"):
        os.system(f"rm -rf models/{id}")
        msg = "Succesfully deleted."
    else:
        msg = "Model does not exist!"
    return {
        "type": "message",
        "value": msg,
    }


def chat(value: dict):
    print("chat")
    def sub_chat(base: str, id: str, lora: str, query: str, history: list[dict]):
        model = create_model(base, id, lora)
        response = model.chat(query, history)
        del model
        return response

    base = value["base"]
    id = value["id"]
    lora = value["lora"]
    query = value["query"]
    history = value["history"]
    
    return {
        "type": "chat",
        "value": sub_chat(base, id, lora, query, history),
    }


def finetune(value: dict):
    print("finetune")
    def sub_finetune(base: str, id: str, dataset: str, name: str, steps: int):
        os.system(
            f"conda run -n llm python finetune.py"
            + f" --base {base}"
            + f" --id {id}"
            + f" --dataset {dataset}"
            + f" --name {name}"
            + f" --steps {steps}"
        )

    base = value["base"]
    id = value["id"]
    dataset = value["dataset"]
    name = value["name"]
    steps = value["steps"]
    sub_finetune(base, id, dataset, name, steps)

    return {"type": "message", "value": f"{name}/checkpoint-{steps}"}


def parse(recv: str):
    print("parse")
    operate = {
        "upload": upload,
        "delete": delete,
        "chat": chat,
        "finetune": finetune,
    }
    recv = json.loads(recv)
    return operate[recv["type"]](recv["value"])


def main():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(2)

    while True:
        print("Try Connection")
        try:
            conn, address = server_socket.accept()
            thread = threading.Thread(target=handle, args=(conn, address))
            thread.start()

        except Exception as err:
            print(err)
            continue


if __name__ == "__main__":
    main()
