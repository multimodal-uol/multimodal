# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 19:35:15 2021

@author: sgmjin2
"""
from __future__ import unicode_literals
import youtube_dl
import sys

video_url = sys.argv[1]
save_path = sys.argv[2]

ydl_opts = {'outtmpl': save_path}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    ydl.download([video_url])