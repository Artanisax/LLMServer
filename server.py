import os
import json
import yaml
import subprocess
import asyncio
from websockets import serve

from util import create_model

HOST = "localhost"
PORT = 8080


def upload(value: dict):
    print("upload")
    
    id = value["id"]
    version = value["version"]
    url = value["url"]
    
    if os.path.exists(f"models/{id}"):
        msg = "Model already exists!"
    else:
        os.makedirs(f"models/{id}")
        os.system(f"wget -P models/{id} {url}")
        os.system(f"unzip -d models/{id} models/{id}/{id}.zip")
        os.system(f"rm -rf unzip models/{id}/{id}.zip")
        msg = "Succesfully uploaded."
    return {
        "type": "message",
        "value": msg,
    }


def delete(value: dict):
    print("delete")
    
    id = value["id"]
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

    def sub_chat(id: str, query: str, history: list[dict]):
        model = create_model(id)
        response = model.chat(query, history)
        del model
        return response

    id = value["id"]
    query = value["query"]
    history = None  # value["history"]
    # config = value["config"]

    return {
        "type": "chat",
        "value": sub_chat(id, query, history),
    }


def finetune(value: dict):
    print("finetune")

    def sub_finetune(id: str, base: str, dataset: str, steps: int):
        cmd = f"conda run -n llm python finetune.py" \
            + f" --id {id}" \
            + f" --base {base}" \
            + f" --dataset {dataset}" \
            + f" --steps {steps}"
        os.system(cmd)
    
    id = value["id"]
    base = value["base"]
    dataset = value["dataset"]
    steps = value["steps"]
    sub_finetune(id, base, dataset, steps)
    # os.system("Succesfully finetuned.")
    
    configs = {
        "class": "Gemma_2" if "gemma" in base.lower() else "Llama_3_2",
        "base": base,
        "lora": f"{id}/checkpoint-{steps}",
    }
    os.makedirs(f"models/{id}", exist_ok=True)
    with open(f"models/{id}/configs.yaml", "w") as file:
        yaml.dump(configs, file)
    
    return {"type": "message", "value": f"Succesfully finetuned."}


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


async def handle(websocket):
    print("handling")
    recv = await websocket.recv()
    print("<server recv>\n", recv)
    send = parse(recv)
    print("<server send>\n", send)
    await websocket.send(json.dumps(send))


async def main():
    async with serve(handle, HOST, PORT):
        await asyncio.get_running_loop().create_future()  # run forever


if __name__ == "__main__":
    print(f"start serving on http://{HOST}:{PORT}")
    asyncio.run(main())
