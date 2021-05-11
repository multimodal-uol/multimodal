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

import spacy
nlp = spacy.load('en_core_web_lg')


@Configuration()
class MyCommand(StreamingCommand):
    doc=""
    
    textfield = Option(
        require=True,
        doc="",
        validate=validators.Fieldname())


    def stream(self, records):

        for record in records:
            data=nlp(record[self.textfield])
            record[self.textfield]=[word.text for word in data.ents]
            record['tag']=[word.label_ for word in data.ents]
            
            yield record
            

dispatch(MyCommand, sys.argv, sys.stdin, sys.stdout, __name__)
