import os
import numpy as np
 
class Taxel:
    def __init__(self, pos, nrm, id):
        self.pos = pos
        self.nrm = nrm
        self.id = id
 
class SkinPart:
    def __init__(self):
        self.name = "unknown_skin_part"
        self.size = 0
        self.version = "unknown_version"
        self.spatial_sampling = "taxel"
        self.taxels = []
        self.taxel2Repr = []
        self.repr2TaxelList = {}
 
    def set_taxel_poses_from_file(self, file_path, spatial_sampling="default"):
        filename = os.path.basename(file_path)
        self.set_name_and_version(filename)
        self.set_spatial_sampling(spatial_sampling)
        calibration = self.read_calibration_data(file_path)
        if not calibration:
            return self.set_taxel_poses_from_file_old(file_path)
        self.size = len(calibration) - 1
        for i in range(1, self.size + 1):
            taxel_data = calibration[i]
            pos_nrm = np.array(taxel_data[:6], dtype=float)
            pos = pos_nrm[:3]
            nrm = pos_nrm[3:]
            if np.linalg.norm(nrm) != 0 or np.linalg.norm(pos) != 0:
                self.taxels.append(Taxel(pos, nrm, i - 1))
 
        self.read_taxel_mapping(calibration)
    def set_name_and_version(self, filename):
        name_version_map = {
            "left_forearm_mesh.txt": ("left_forearm", "V1"),
            "left_forearm_nomesh.txt": ("left_forearm", "V1"),
            "left_forearm_V2.txt": ("left_forearm", "V2"),
            "right_forearm_mesh.txt": ("right_forearm", "V1"),
            "right_forearm_nomesh.txt": ("right_forearm", "V1"),
            "right_forearm_V2.txt": ("right_forearm", "V2"),
            "left_hand_V2_1.txt": ("left_hand", "V2.1"),
            "right_hand_V2_1.txt": ("right_hand", "V2.1"),
            "left_arm_mesh.txt": ("left_upper_arm", "V1"),
            "right_arm_mesh.txt": ("right_upper_arm", "V1"),
            "torso.txt": ("front_torso", "V1")
        }
        if filename in name_version_map:
            self.name, self.version = name_version_map[filename]
        else:
            return None
 
    def set_spatial_sampling(self, spatial_sampling):
        if spatial_sampling in ["default", "taxel", "triangle"]:
            self.spatial_sampling = spatial_sampling
        else:
            raise ValueError(f"Invalid spatial sampling: {spatial_sampling}")
 
    def read_calibration_data(self, file_path):
        # This function should read the calibration data from the file and return it as a list of lists
        # The first element of the list should be the size of the calibration data
        # The remaining elements should be the calibration data
        # The calibration data should be in the following format:
        # [size, [taxel_0_data], [taxel_1_data], ..., [taxel_n_data]]
        # where [taxel_i_data] is a list of 6 elements: [x, y, z, nx, ny, nz]
        # The calibration data should be ordered by taxel id
        # The taxel id should be the index of the calibration data in the list
        # The taxel id should start from 0
        # If the file does not exist or the calibration data is invalid, return None
        calibration = []
        start_found = False
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    if start_found:
                        if not line.strip():
                            continue
                        pos_nrm = list(map(float, line.split()))
                        calibration.append(pos_nrm)
                    if line == "[calibration]\n":
                        start_found = True
            return calibration
        except FileNotFoundError:
            return None
 
    def set_taxel_poses_from_file_old(self, file_path):
        with open(file_path, 'r') as file:
            for line in file:
                if not line.strip():
                    continue
                pos_nrm = list(map(float, line.split()))
                pos = pos_nrm[:3]
                nrm = pos_nrm[3:]
                if np.linalg.norm(nrm) != 0 or np.linalg.norm(pos) != 0:
                    self.taxels.append(Taxel(pos, nrm, len(self.taxels)))
 
    def read_taxel_mapping(self, calibration):
        # Read the taxel to representative mapping from the calibration data
        self.taxel2Repr = list(range(self.size))
        self.repr2TaxelList = {i: [i] for i in range(self.size)}
 
# Usage
skin_part = SkinPart()
skin_part.set_taxel_poses_from_file("../../icub-main/app/skinGui/conf/positions/left_arm.txt")
pass