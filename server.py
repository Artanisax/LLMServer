import os
import json
import socket
import threading
import subprocess
import multiprocessing

from .util import create_model, lora_finetune

pool = multiprocessing.Pool(processes=2)  # for gpu tasks
HOST = "localhost"
PORT = 8899
config_cnt = 0
    
def handle(conn, address):
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
    # print("recv:\n", recv)
    
    send = parse(recv)
    # print("send:", send)
    conn.send(json.dumps(send).encode())
    
    conn.close()
    # print("Close Connection")

def upload(value: str):
    model = json.loads(value)
    id = model["id"]
    zip_file = model["zip_file"]
    if os.path.exists(f"models/{id}"):
        msg = "Model already exists!"
    else:
        os.makedirs(f"models/{id}")
        with open(f"models/{id}.zip", "wb") as file:
            file.write(zip_file)
        msg = "Succesfully uploaded."
    return {
        "type": "message",
        "value": msg,
    }

def delete(id):
    if os.path.exists(f"models/{id}"):
        os.system(f"rm -rf models/{id}")
        msg = "Succesfully deleted."
    else:
        msg = "Model does not exist!"
    return {
        "type": "message",
        "value": msg,
    }

def subchat(config: dict):
    env = "chatglm" if "ChatGLM" in id else "gemma"
    with threading.Lock():
        config_cnt += 1
        config_path = f".cache/{config_cnt}.json"
        with open(config_path, "w") as file:
            json.dump(config, file)
    cmd = f"conda run -n {env} python chat.py --config_path {config_path}"
    ret = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return ret.stdout

def chat(value: dict):
    id = value["id"]
    with threading.Lock():
        result = pool.apply(subchat, args=(value,))
    return {
        "type": "chat",
        "value": result,
    }

def finetune(value: dict):
    return NotImplemented
    id = value["id"]
    base = value["base"]
    dataset = value["dataset"]
    return {
        "type": "message",
        "value": lora_finetune(id, base, dataset),
    }

def parse(recv: str):
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
    server_socket.listen(1)
    
    while True:
        print("Try Connection")
        try:
            conn, address = server_socket.accept()
            print(conn, '\n', address)
            # conn.settimeout(5)
            thread = threading.Thread(target=handle, args=(conn, address))
            thread.start()
        
        except Exception as err:
            print(err)
            continue


if __name__ == "__main__":
    main()
