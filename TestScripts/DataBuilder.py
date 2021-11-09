import glob , yaml, csv


class DataBuilder:
    def __init__(self):
        self.data = {}
        self.base_keys = []
        self.n = 0
        self.path = f"Vehicles/"

    def build_keys(self):
        with open(f"{self.path}base_key_builder.yaml") as f:
            key_data = yaml.safe_load(f)

        for key in key_data:
            self.base_keys.append(key)
            # self.data[key] = []

    def read_data(self):
        folders = glob.glob(f"{self.path}*/")
        for i, folder in enumerate(folders):
            print(f"Folder being opened: {folder}")
            
            try:
                config = glob.glob(folder + '/*_record.yaml')[0]
            except Exception as e:
                print(f"Exception: {e}")
                print(f"Filename issue: {folder}")
                continue
            
            with open(config, 'r') as f:
                config_data = yaml.safe_load(f)

            self.data[i] = {}
            for key in config_data.keys():
                if key in self.base_keys:
                    self.data[i][key] = config_data[key]


    def save_data_table(self, name="DataTable"):
        directory = "SafetyAnalysis/" + name + ".csv"
        with open(directory, 'w') as file:
            writer = csv.DictWriter(file, fieldnames=self.base_keys)
            writer.writeheader()
            for key in self.data.keys():
            # for i in range(len(self.data.keys())):
                writer.writerow(self.data[key])


        print(f"Data saved to {name} --> {len(self.data)} Entries")

if __name__ == "__main__":
    db = DataBuilder()
    db.build_keys()
    db.read_data()
    db.save_data_table()


