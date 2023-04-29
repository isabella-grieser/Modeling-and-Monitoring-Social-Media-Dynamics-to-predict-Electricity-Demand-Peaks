import json
import gen.modelgen as mb

CONFIG_PATH = "config/arguments.json"


if __name__ == "__main__":
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    print(config)

    # create social media network model
    network_model = mb.create_social_network_graph(
        config["network"]["nodes"],
        config["network"]["type"],
        config["network"]
    )
