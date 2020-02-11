import numpy as np
import os


class DamaskExporter:

    def __init__(self, data, project_name, project_dir):
        self.project_dir = project_dir
        self.abs_path = os.path.abspath(project_dir)
        if not os.path.exists(self.abs_path):
            os.makedirs(self.abs_path)
        self.data = data
        self.project_name = project_name
        pass

    def create_geom_file(self):
        w, h = self.data.shape
        geom_file = ""
        geom_file += "5	header\n"
        geom_file += "grid	a {0}	b {1}	c 1\n".format(h, w)
        geom_file += "size	x 1.000000	y 1.000000	z 1.000000\n"
        geom_file += "origin	x 0.000000	y 0.000000	z 0.000000\n"
        geom_file += "microstructures   1\n"
        geom_file += "homogenization    1\n"

        arr_str = np.array2string(self.data, max_line_width=9999999, threshold=9999999).replace('[', '').replace(']', '').replace('\n ', '\n')
        geom_file += arr_str

        text_file = open(self.abs_path + os.sep + self.project_name + ".geom", "w")
        text_file.write(geom_file)
        text_file.close()

        pass

    def create_material_config(self):

        material_file = self.load_template

        min_num = np.min(self.data)
        max_num = np.max(self.data) + 1

        material_file += "<microstructure>\n"

        for i in range(min_num, max_num, 1):
            material_file += "[Grain{0}]\n".format(i)
            material_file += "crystallite 1\n"
            material_file += "(constituent) phase 1 texture {0} fraction 1.0\n".format(i)

        material_file += "<texture>\n"
        for i in range(min_num, max_num, 1):
            material_file += "[Grain{0}]\n".format(i)
            material_file += "(gauss) phi1 {0} Phi {1} phi2 {2} scatter 0.0  fraction 1.0\n".format(
                np.random.random_sample() * 360, np.random.random_sample() * 360, np.random.random_sample() * 360)

        text_file = open(self.abs_path + os.sep + "material.config", "w")
        text_file.write(material_file)
        text_file.close()

    def run_damask(self):
        run_str = "DAMASK_spectral --geom {0}.geom --load {1}.load".format(self.project_name, self.project_name)
        os.chdir(self.abs_path)
        os.system("bash -c 'source ~/damask-2.0.3/env/DAMASK.sh;{0}'".format(run_str))
        pass

    load_template = """
#-------------------#
<homogenization>
#-------------------#

[SX]
mech	none

#-------------------#
<crystallite>
#-------------------#
[almostAll]
(output) phase
(output) texture
(output) volume
(output) orientation    # quaternion
(output) grainrotation  # deviation from initial orientation as axis (1-3) and angle in degree (4)
(output) f              # deformation gradient tensor; synonyms: "defgrad"
(output) fe             # elastic deformation gradient tensor
(output) fp             # plastic deformation gradient tensor
(output) p              # first Piola-Kichhoff stress tensor; synonyms: "firstpiola", "1stpiola"
(output) lp             # plastic velocity gradient tensor

#-------------------#
<phase>
#-------------------#
[Aluminum_phenopowerlaw]
elasticity              hooke
plasticity              phenopowerlaw

(output)                resistance_slip
(output)                shearrate_slip
(output)                resolvedstress_slip
(output)                totalshear
(output)                resistance_twin
(output)                shearrate_twin
(output)                resolvedstress_twin
(output)                totalvolfrac

lattice_structure       fcc
Nslip                   12   # per family
Ntwin                    0   # per family

c11                     106.75e9
c12                     60.41e9
c44                     28.34e9

gdot0_slip              0.001
n_slip                  20
tau0_slip                 31e6 # per family
tausat_slip               63e6 # per family
a_slip                  2.25
h0_slipslip             75e6
interaction_slipslip    1 1 1.4 1.4 1.4 1.4
atol_resistance         1

"""
