import yaml

from toolkit.data.dataset_generator import DatasetGenerator



if __name__ == "main":
    with open('./toolkit/config.yml', 'r') as file:
        config = yaml.safe_load(file)

    data_config = config.values()

    dataset_generator = DatasetGenerator(**data_config)
    dataset_generator.get_dataset()
