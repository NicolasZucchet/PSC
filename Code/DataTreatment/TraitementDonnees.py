#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 15:56:24 2017

@author: arthurloison
"""

import csv

tab=[[],[]]

with open('psc-heart_rate.csv', newline='') as csvfile:
    rd = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
    for row in rd:
        if (row[6]!="'-3'" and row[6]!="'0'" and row[6]!='heart_rate'):
            tab[0].append(int(row[9][1:-1]))
            tab[1].append(int(row[6][1:-1]))