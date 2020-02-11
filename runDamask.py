import ExportToDamask
import numpy as np


def main():
    filename = "./models/test.npy"
    #data = np.load(filename)

    data = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])

    de = ExportToDamask.DamaskExporter(data=data, project_name="vdk_test", project_dir="./ex-dam")
    de.create_geom_file()
    de.create_material_config()
    de.run_damask()


if __name__ == "__main__":
    main()
