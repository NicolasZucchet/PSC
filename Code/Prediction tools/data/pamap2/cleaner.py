#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

### The goal of this file is to give functions that make the pamap2 data usable

### What to do in order to make pamap2 data usable :
        # - open the file in excel and use excel_formula_generator to remove all the NaN values
        # - use data cleaner
        # - use the treated data in pamap2/data

def data_cleaner():
    # Takes the needed columns
    # Takes a subject***.txt which has lot of 0 activities
    # and creates a subject***T.txt where 0 activities are removed (except one to
    # separate other activities)
    data_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(data_path)
    raw_data_path = os.path.join(data_path,"pamap2/raw_data")
    data_t_path = os.path.join(data_path,"pamap2/data")
    to_process_files = os.listdir(raw_data_path)
    to_process_files = to_process_files[2:16]
    for file in to_process_files:
        print(file)
        if file[0] != ".":
            # extraction
            file_content = pd.read_csv(os.path.join(raw_data_path,file),delim_whitespace = True)
            matrix_content = file_content.as_matrix()
            j = 0
            k = []
            for i in range(len(matrix_content)-1):
                if not(matrix_content[i][1] == 0 and matrix_content[i+1][1] == 0):
                    k.append(i)
            print(k)
            np.savetxt(os.path.join(data_t_path,file),matrix_content[k,:14],delimiter=";")
                        
            
def excel_formula_generator():
    # Generate a formula to fill all the NaN values for heart rate
    sl = "";
    sr = "";
    lim = 30;
    for i in range(1,lim):
        if (i != lim-1):
            sl = "SI(ESTNUM(D" + str(i) + ");D" + str(i) + ";" + sl;
            sr += ")";
        else:
            s = sl + "D1"  + sr
    print(s)
    
excel_formula_generator()