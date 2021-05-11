#!/usr/bin/env python
# coding=utf-8
#
# Copyright © 2011-2015 Splunk, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License"): you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')

splunkhome = os.environ['SPLUNK_HOME']
sys.path.append(os.path.join(splunkhome, 'etc', 'apps', 'searchcommands_app', 'lib'))
from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators
from splunklib import six
from splunklib.six.moves import range

import urllib.request
from PIL import Image
from urllib.request import urlopen
import xlsxwriter
from io import BytesIO
import uuid

def get_img_size(url):
    image = Image.open(urllib.request.urlopen(url))
    width, height = image.size
    return width,height

SAVE_BASE_DIR='/opt/splunk/etc/apps/Multimodal/appserver/static/excel/'
#SAVE_BASE_DIR='/opt/splunk/etc/apps/Multimodal/bin/insert_img_excel/'
filename = uuid.uuid4().hex + '.xlsx'
save_path = SAVE_BASE_DIR + filename
workbook = xlsxwriter.Workbook(save_path)
text_format = workbook.add_format({'text_wrap': True})

CELL_WIDTH = 320 # column width: 320 pixels
CELL_HEIGHT = 200 # row height: 200 pixels

@Configuration()
class Url2img(StreamingCommand):
    
    ori_text = Option(
        require=True,
        validate=validators.Fieldname())
    
    twitter_img_urls = Option(
        require=True,
        validate=validators.Fieldname())
    
    newspaper_img_urls = Option(
        require=True,
        validate=validators.Fieldname())
    

    def stream(self, records):
        worksheet = workbook.add_worksheet()

        worksheet.write(0, 0, 'ori_text')
        worksheet.write(0, 1, 'twitter_image')
        worksheet.write(0, 2, 'newspaper_image')
        
        worksheet.set_column(0, 0, 100)
        worksheet.set_column(1, 2, 45)
        
        index = 1
        for record in records:
            text = record[self.ori_text]
            twitter_url = record[self.twitter_img_urls]
            newspaper_url = record[self.newspaper_img_urls]
            
            worksheet.set_row(index, 150)
            
            worksheet.write(index, 0, text,text_format)
            
            try:
                twitter_image_data = BytesIO(urlopen(twitter_url).read())
                width,height = get_img_size(twitter_url)
                scale = min(CELL_WIDTH/width, CELL_HEIGHT/height)*0.75
                worksheet.insert_image(index,1, twitter_url, 
                                       {'image_data': twitter_image_data,
                                        'x_scale': scale,
                                        'y_scale': scale})
            except Exception:
                pass
                
            try:
                newspaper_image_data = BytesIO(urlopen(newspaper_url).read())
                width,height = get_img_size(newspaper_url)
                scale = min(CELL_WIDTH/width, CELL_HEIGHT/height)*0.75
                worksheet.insert_image(index,2, newspaper_url, 
                                       {'image_data': image_data,
                                        'x_scale': scale,
                                        'y_scale': scale})
            except Exception:
                pass
                
            index+=1
            record['filename'] = filename
            
            yield record    
            
        workbook.close()
        

dispatch(Url2img, sys.argv, sys.stdin, sys.stdout, __name__)
