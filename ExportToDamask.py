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

        self.avg_data = None
        self.ls_data = None
        self.ls_nodal_data = None
        self.ls_data_file = ""
        self.n_ls = 0
        pass

    def tension_none(self):
        strain = np.full((3, 3), None)
        stress = np.full((3, 3), None)
        self.add_loadstep(strain=strain, stress=stress)
        return strain, stress

    def tension_x(self, val=1e-5):
        strain = np.zeros((3, 3))
        stress = np.full((3, 3), None)
        strain[0, 0] = val
        strain[2, 2] = None
        stress[2, 2] = 0
        self.add_loadstep(strain=strain, stress=stress)
        strain[0, 0] = -val
        self.add_loadstep(strain=strain, stress=stress)
        return strain, stress

    def tension_y(self, val=1e-5):
        strain = np.zeros((3, 3))
        stress = np.full((3, 3), None)
        strain[1, 1] = val
        strain[2, 2] = None
        stress[2, 2] = 0
        self.add_loadstep(strain=strain, stress=stress)
        strain[1, 1] = -val
        self.add_loadstep(strain=strain, stress=stress)
        return strain, stress

    def tension_x_and_y(self, xval=1e-5, yval=1e-5, restore=True):
        strain = np.zeros((3, 3))
        stress = np.full((3, 3), None)
        strain[0, 0] = xval
        strain[1, 1] = yval
        strain[2, 2] = None
        stress[2, 2] = 0
        self.add_loadstep(strain=strain, stress=stress)
        if restore:
            strain[0, 0] = -xval
            strain[1, 1] = -yval
            self.add_loadstep(strain=strain, stress=stress)
        return strain, stress

    def shear_xy(self, val=1e-5):
        strain = np.zeros((3, 3))
        stress = np.full((3, 3), np.nan)
        strain[1, 0] = val
        strain[2, 2] = None
        stress[2, 2] = 0
        self.add_loadstep(strain=strain, stress=stress)
        strain[1, 0] = -val
        self.add_loadstep(strain=strain, stress=stress)
        return strain, stress

    def create_geom_file(self):
        w, h = self.data.shape
        geom_file = ""
        geom_file += "5	header\n"
        geom_file += "grid	a {0}	b {1}	c 1\n".format(h, w)
        geom_file += "size	x 1.000000	y 1.000000	z 1.000000\n"
        geom_file += "origin	x 0.000000	y 0.0000000	z 0.000000\n"
        geom_file += "microstructures   1\n"
        geom_file += "homogenization    1\n"

        arr_str = np.array2string(self.data, max_line_width=9999999, threshold=9999999).replace('[', '').replace(']',
                                                                                                                 '').replace(
            '\n ', '\n')
        geom_file += arr_str

        text_file = open(self.abs_path + os.sep + self.project_name + ".geom", "w")
        text_file.write(geom_file)
        text_file.close()

        pass

    def create_material_config(self, rand_orient=True, phase1=None, phase2=None):

        material_file = self.load_template

        min_num = np.min(self.data)
        max_num = np.max(self.data) + 1

        material_file += "<microstructure>\n"

        for i in range(min_num, max_num, 1):
            material_file += "[Grain{0}]\n".format(i)
            material_file += "crystallite 1\n"
            if phase1 is not None and phase2 is not None:
                if i in phase1:
                    material_file += "(constituent) phase 1 texture {0} fraction 1.0\n".format(i)
                elif i in phase2:
                    material_file += "(constituent) phase 2 texture {0} fraction 1.0\n".format(i)
                else:
                    material_file += "(constituent) phase 1 texture {0} fraction 1.0\n".format(i)
            else:
                material_file += "(constituent) phase 1 texture {0} fraction 1.0\n".format(i)

        material_file += "<texture>\n"
        for i in range(min_num, max_num, 1):
            material_file += "[Grain{0}]\n".format(i)
            orient = np.random.rand((3)) * 360.0
            if not rand_orient:
                orient = 0.0 * orient
            material_file += "(gauss) phi1 {0} Phi {1} phi2 {2} scatter 0.0  fraction 1.0\n".format(*orient)

        text_file = open(self.abs_path + os.sep + "material.config", "w")
        text_file.write(material_file)
        text_file.close()

    def run_damask(self):
        self.write_loading()
        run_str = "DAMASK_spectral --geom {0}.geom --load {1}.load".format(self.project_name, self.project_name)
        os.chdir(self.abs_path)
        ret = os.system("bash -c 'source ~/damask-2.0.3/env/DAMASK.sh;{0}'".format(run_str))
        print('ret code == ', ret)
        if ret:
            exit(ret)
        pass

    def post_proc(self, avg_only=True):

        s = self.post_proc_script
        if avg_only:
            s = self.post_proc_script_short

        run_str = s.format(self.project_name)
        os.chdir(self.abs_path)
        print(run_str)
        os.system("bash -c 'source ~/damask-2.0.3/env/DAMASK.sh;{0}'".format(run_str))

        self.avg_data = self.post_txt("ttl" + self.project_name + "_" + self.project_name + ".txt")

        pass

    def load_by_ls_num(self, n):
        frmt = ":0{0}d".format(int(np.log(self.n_ls)) - 1)
        frmt = "{" + frmt + "}"
        self.ls_data = self.post_txt(self.project_name + "_" + self.project_name + "_inc{}.txt".format(frmt).format(n))
        #self.ls_nodal_data = self.post_txt(
        #    self.project_name + "_" + self.project_name + "_inc{}_nodal.txt".format(frmt).format(n))
        pass

    def post_txt(self, filename):
        text_file = open(self.abs_path + os.sep + "postProc" + os.sep + filename, "r")
        val = text_file.readline().split('\t')

        n_head = int(val[0]) + 1
        text_file.close()
        full_name = self.abs_path + os.sep + "postProc" + os.sep + filename

        data = np.loadtxt(full_name, skiprows=n_head, dtype=np.float64)

        return data

        pass

    def add_loadstep(self, strain, stress, time=1, nsubst=1, write_n_subs=1):

        dropguessing = ""
        if self.ls_data_file:
            dropguessing = "dropguessing"

        strain_str = "fdot {0} {1} {2}  {3} {4} {5}   {6} {7} {8} ".format(*strain.reshape((9)))
        stress_str = "stress  {0} {1} {2}  {3} {4} {5}   {6} {7} {8} ".format(*stress.reshape((9)))
        ls_str = "time {0}  incs {1} freq {2} {3}\n".format(time, nsubst, write_n_subs, dropguessing)

        strain_str = strain_str.replace('None', '*')
        stress_str = stress_str.replace('None', '*')

        strain_str = strain_str.replace('nan', '*')
        stress_str = stress_str.replace('nan', '*')

        self.ls_data_file += strain_str + stress_str + ls_str
        self.n_ls += 1
        print(self.ls_data_file)
        pass

    def write_loading(self):
        text_file = open(self.abs_path + os.sep + self.project_name + ".load", "w")
        text_file.write(self.ls_data_file)
        text_file.close()
        pass

    def stress_vonMizes(self, s):
        von_mizes = np.sqrt(((s[0, 0] - s[1, 1]) ** 2 + (s[1, 1] - s[2, 2]) ** 2 + (s[2, 2] - s[0, 0]) ** 2 + 6 * (
                s[0, 1] ** 2 + s[1, 2] ** 2 + s[0, 2] ** 2)) / 2)
        return von_mizes

    def strain_vonMizes(self, e):
        von_mizes = np.sqrt(
            3.0 / 2.0 * ((e[0, 0] - e[1, 1]) ** 2 + (e[1, 1] - e[2, 2]) ** 2 + (e[2, 2] - e[0, 0]) ** 2)
            + 3.0 / 4.0 * (e[0, 1] ** 2 + e[1, 2] ** 2 + e[0, 2] ** 2)
        ) * 2.0 / 3.0
        return von_mizes

    def get_avg_strain_tensor(self, n=-1):
        return self.avg_data[n, 36:36 + 9].reshape((3, 3))

    def get_avg_strain_vonMizes(self, n=-1):
        e = self.get_strain_tensor()
        return self.strain_vonMizes(e)

    def get_avg_stress_tensor(self, n=-1):
        return self.avg_data[n, 27:27 + 9].reshape((3, 3))

    def get_avg_stress_vonMizes(self, n=-1):
        s = self.get_avg_stress_tensor(n)
        return self.stress_vonMizes(s)

    def get_strain_tensor(self):
        w, _ = self.ls_data.shape
        return self.ls_data[:, 36:36 + 9].reshape((w, 3, 3))

    def get_stress_tensor(self):
        w, _ = self.ls_data.shape
        return self.ls_data[:, 27:27 + 9].reshape((w, 3, 3))

    def get_stress_vonMizes(self):
        s = self.get_stress_tensor()
        return self.stress_vonMizes(np.moveaxis(s, 0, -1))

    def get_node_coord(self):
        w, _ = self.ls_data.shape
        return self.ls_data[:, 5:5 + 3].reshape((w, 3))

    def get_displ(self):
        w, _ = self.ls_nodal_data.shape
        return self.ls_nodal_data[:, 3:3 + 3].reshape((w, 3))
        pass

    def get_displ_sum(self):
        d = self.get_displ()
        return np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
        pass

    def get_displ_nodal_coord(self):
        w, _ = self.ls_nodal_data.shape
        return self.ls_nodal_data[:, 0:0 + 3].reshape((w, 3))
        pass

    def get_max_stress_for_each_phase(self):
        phase = self.ls_data[:, 26]
        list_phase = np.unique(phase)
        max_stress = np.zeros_like(list_phase)
        von_mizes = self.get_stress_vonMizes()
        for i in list_phase:
            mask = phase == i
            max_stress[int(i - 1)] = np.max(von_mizes[mask])

        return max_stress

    def get_max095_stress_for_each_phase(self):
        phase = self.ls_data[:, 26]
        list_phase = np.unique(phase)
        max_stress = np.zeros_like(list_phase)
        von_mizes = self.get_stress_vonMizes()
        for i in list_phase:
            mask = phase == i
            sorted_stress = np.sort(von_mizes[mask])
            max_stress[int(i - 1)] = sorted_stress[int(sorted_stress.size * 0.95)]

        return max_stress

    def get_min_stress_for_each_phase(self):
        phase = self.ls_data[:, 26]
        list_phase = np.unique(phase)
        min_stress = np.zeros_like(list_phase)
        von_mizes = self.get_stress_vonMizes()
        for i in list_phase:
            mask = phase == i
            min_stress[int(i - 1)] = np.min(von_mizes[mask])

        return min_stress

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



<phase>

# Tasan et.al. 2015 Acta Materalia
# Tasan et.al. 2015 International Journal of Plasticity
# Diehl et.al. 2015 Meccanica
[BCC-Ferrite]

elasticity              hooke
plasticity              phenopowerlaw

lattice_structure       bcc
Nslip                   12  12                  # per family
Ntwin                    0                      # per family
c11                     233.3e9
c12                     135.5e9
c44                     118.0e9
gdot0_slip              0.001
n_slip                  20
tau0_slip                95.e6  97.e6           # per family, optimization long simplex 109
tausat_slip             222.e6 412.7e6          # per family, optimization long simplex 109
h0_slipslip             1000.0e6
interaction_slipslip    1 1 1.4 1.4 1.4 1.4
w0_slip                 2.0
a_slip                  2.0

# Tasan et.al. 2015 Acta Materalia
# Tasan et.al. 2015 International Journal of Plasticity
# Diehl et.al. 2015 Meccanica

###---------------------------------------------------------------------------------------------------------------

[Pearlite]

elasticity              hooke
plasticity              phenopowerlaw

lattice_structure       isotropic
c11                     297.0e9
c12                     107.2e9
m               3
tau0            0.7e9
gdot0           0.001
n               20
h0              0.728e9
tausat          1.6e9
a               2.25
atol_resistance 1



"""

    post_proc_script = """
rm -r ./postProc

filename="{0}_{0}"

echo "$filename.spectralOut"

postResults --nodal --cr f,p,phase --split --separation x,y,z "$filename.spectralOut"


cd ./postProc

addCauchy $filename_inc*.txt
#addMises -s Cauchy $filename_inc*.txt

#addStrainTensors --left --logarithmic $filename_inc*.txt
#addMises -e "ln(V)" $filename_inc*.txt

#addDisplacement --nodal $filename_inc*.txt

cd ..

postResults --cr f,p,phase "$filename.spectralOut" --prefix="ttl"

cd ./postProc

addCauchy ttl$filename_inc*.txt
#addMises -s Cauchy ttl$filename_inc*.txt

#addStrainTensors --left --logarithmic ttl$filename_inc*.txt
#addMises -e "ln(V)" ttl$filename_inc*.txt

     



"""

    post_proc_script_short = """
    rm -r ./postProc

    filename="{0}_{0}"

    echo "$filename.spectralOut"

    postResults --cr f,p "$filename.spectralOut" --prefix="ttl"

    cd ./postProc

    addCauchy $filename_inc*.txt
    addMises -s Cauchy $filename_inc*.txt

    addStrainTensors --left --logarithmic $filename_inc*.txt
    addMises -e "ln(V)" $filename_inc*.txt
    """
