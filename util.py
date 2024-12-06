import LLM
import yaml

def create_model(id: str) -> LLM.LLM:
    with open(f"models/{id}/configs.yaml") as file:
        configs = yaml.load(file, yaml.FullLoader)
    base = configs["base"]
    lora = configs["lora"]
    cls = configs["class"]
    package = __import__("LLM")
    model_class = getattr(package, cls)
    return model_class(base, lora)