import argparse
import json
from .util import create_model

def main(config):
    id = config["id"]
    base = config["base"]
    lora = config["lora"] if "lora" in config.keys() else None
    query = config["query"]
    history = config["history"] if "history" in config.keys() else None
    model = create_model(id, base, lora)
    print(json.dump(model.chat(query, history)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str) 
    args = parser.parse_known_args()
    with open(args.config_path, "r") as file:
        config = json.load(file)
    main(config)