import numpy as np
import matplotlib.pyplot as plt

MODEL_PATH = "./neuromorphic_body_schema/models/icub_v2_full_body.xml"

if __name__ == '__main__':
    
# 1. read lines xml file
    # now we can open the xml robot config file 
        
    keep_processing = True
    line_counter = 0
    joint_counter = 0 

    with open(MODEL_PATH, 'r') as file:
        lines = file.readlines()

    while keep_processing:
         with open("icub_v2_full_body_kp_tuning.xml", 'a') as output_file:
        
            if "kp=" in lines[line_counter] :
            
                if "ctrllimited" in lines[line_counter] :
                    #output_file.write( lines[line_counter] )
                    line_counter += 1
                    continue
                # 2. start from line 0, read kp and ctrl values
                
                part = lines[line_counter].split('name="')[1].split('"')[0]
                kp = float(lines[line_counter].split('kp="')[1].split('"')[0])
                KP_ORIGINAL = kp

                kp = 999
                new_line = lines[line_counter].split('kp="')[0] + str('kp="') + str(kp) +  lines[line_counter].split(str(KP_ORIGINAL))[1]
                output_file.write(new_line)
                line_counter+=1
                    
            else:
                output_file.write( lines[line_counter] ) 
                line_counter += 1


         if line_counter >= len(lines):
            keep_processing = False



