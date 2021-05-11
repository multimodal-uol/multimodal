# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 19:35:15 2021

@author: sgmjin2
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys

from warnings import filterwarnings
filterwarnings("ignore")

from splunklib.searchcommands import dispatch, GeneratingCommand, Configuration, Option, validators

sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')

SAVE_BASE_DIR = '/opt/splunk/etc/apps/Multimodal/appserver/static/videos/'
VISIT_BASE = 'http://35.246.69.64:8000/static/app/Multimodal/videos/'

@Configuration()
class FindReport(GeneratingCommand):
    
    video_id = Option(require=True)
    
    def generate(self):
        target_folder = os.path.join(SAVE_BASE_DIR,self.video_id)
        
        reports = []
        for subdir, dirs, files in os.walk(target_folder):
            for f in files:
                if f.endswith(".html"):
                    reports.append(f)
        
        for i in range(len(reports)):
            yield{
                'report #': i+1,
                'report link' : os.path.join(VISIT_BASE, self.video_id, reports[i])
            }

dispatch(FindReport, sys.argv, sys.stdin, sys.stdout, __name__)