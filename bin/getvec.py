# -*- coding: utf-8 -*-
"""
Created on Wed Oct 7 09:46:21 2020

@author: sgmjin2
"""

import sys
import re
import os
import logging, logging.handlers

sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')

import pandas as pd

from splunk import setupSplunkLogger
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators



@Configuration(local=True)
class Getvec(StreamingCommand):
    textfield = Option(
        require=True,
        doc='''
        **Syntax:** **textfield=***<fieldname>*
        **Description:** Name of the field that will contain the text to search against''',
        )

    def stream(self, records):
        coordinates = pd.read_csv('coordinates.csv')
        xs=[]
        ys=[]
        zs=[]
        for index, row in coordinates.iterrows():
            xs.append(row['xs'])
            ys.append(row['ys'])
            zs.append(row['zs'])

        for record, x, y, z in zip(records, xs, ys, zs):
            record['x']=x
            record['y']=y
            record['z']=z
            yield record

dispatch(Getvec, sys.argv, sys.stdin, sys.stdout, __name__)