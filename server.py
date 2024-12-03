import os
import json
import asyncio
from websockets import serve

from util import create_model

HOST = "localhost"
PORT = 8080


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
    # config = value["config"]

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


async def handle(websocket):
    recv = await websocket.recv()
    print("<server recv>\n", recv)
    send = parse(recv)
    print("<server send>\n", send)
    await websocket.send(json.dumps(send))


async def main():
    async with serve(handle, HOST, PORT):
        await asyncio.get_running_loop().create_future()  # run forever


if __name__ == "__main__":
    print(f"start serving on http:/{HOST}:{PORT}")
    asyncio.run(main())
