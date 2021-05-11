# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 19:35:15 2021

@author: sgmjin2
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys


from warnings import filterwarnings
filterwarnings("ignore")

sys.path.append('/opt/splunk/etc/apps/multimodal-datagen/bin')
sys.path.append(r'/opt/anaconda3/lib/python3.7/site-packages')
sys.path.append(r'/opt/anaconda3/lib/python3.7/lib-dynload')

from clarifai_grpc.grpc.api import service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
import traceback

from scenedetect import VideoManager
from scenedetect import SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

import subprocess
import pytube
import urllib.parse as urlparse

import dominate
from dominate.tags import *

from splunklib.searchcommands import dispatch, StreamingCommand, Configuration, Option, validators

SAVE_BASE_DIR='/opt/splunk/etc/apps/Multimodal/appserver/static/videos/'

def detectImage(imageURL, modelType, confidenceScore, addImageDescription):
    model = ''
    if modelType == 'NSFW':
        model = 'e9576d86d2004ed1a38ba0cf39ecb4b1'
    elif modelType == 'Apparel':
        model = '72c523807f93e18b431676fb9a58e6ad'
    elif modelType == 'Celebrity':
        model = 'cfbb105cb8f54907bb8d553d68d9fe20'
    elif modelType == 'Face':
        model = 'f76196b43bbd45c99b4f3cd8e8b40a8a'
    elif modelType == 'Logo':
        model = 'c443119bf2ed4da98487520d01a0b1e3'
    elif modelType == 'Color':
        model = 'eeed0b6733a644cea07cf4c60f87ebb7'
    allConcepts=''
    concepts = ''
    if addImageDescription =='True':
        if modelType== 'Demographics' or modelType== 'Travel' or modelType== 'Food' or modelType== 'General':
            try:
                stub = service_pb2_grpc.V2Stub(ClarifaiChannel.get_grpc_channel())
                metadata = (('authorization', 'Key ed69d46b58a24e96aa7184ed2f635931'),)
                post_workflow_results_response = stub.PostWorkflowResults(
                    service_pb2.PostWorkflowResultsRequest(
                        workflow_id=modelType,
                        inputs=[
                            resources_pb2.Input(
                                data=resources_pb2.Data(
                                    image=resources_pb2.Image(
                                        url=imageURL
                                    )
                                )
                            )
                        ]
                    ),
                    metadata=metadata
                )
                if post_workflow_results_response.status.code != status_code_pb2.SUCCESS:
                    raise Exception("Post workflow results failed, status: " +
                                    post_workflow_results_response.status.description)
 
                results = post_workflow_results_response.results[0]
 
                if modelType== 'Demographics':
                    concepts = results.outputs[2].data.regions[0].data.concepts
                elif modelType== 'Travel' or modelType== 'Food' or modelType == 'General':
                    concepts = results.outputs[0].data.concepts
 
                for concept in concepts:
                    if concept.value > float(confidenceScore):
                        allConcepts=allConcepts+ ' '+concept.name
            except:
                f = open("/tmp/commands.txt", "w")
                f.write(str(traceback.format_exc()))
                f.close()
        else:
            try:
                channel = ClarifaiChannel.get_grpc_channel()
                stub = service_pb2_grpc.V2Stub(channel)
                metadata = (('authorization', 'Key ed69d46b58a24e96aa7184ed2f635931'),)
                post_model_outputs_response = stub.PostModelOutputs(
                    service_pb2.PostModelOutputsRequest(
                        model_id=model,
                        #version_id="{THE_MODEL_VERSION_ID}",  # This is optional. Defaults to the latest model version.
                        inputs=[
                            resources_pb2.Input(
                                data=resources_pb2.Data(
                                    image=resources_pb2.Image(
                                        url=imageURL
                                    )
                                )
                            )
                        ]
                    ),
                    metadata=metadata
                )
                if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
                    raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)
                output = post_model_outputs_response.outputs[0]
                if modelType == 'Logo' or modelType== 'Apparel' or modelType== 'Face':
                    for region in output.data.regions:
                        for concept in region.data.concepts:
                            if concept.value > confidenceScore:
                                allConcepts=allConcepts+ ' '+concept.name
                elif modelType == 'Color':
                    concepts = output.data.colors
                elif modelType == 'Celebrity' or modelType== 'NSFW' :
                    concepts = output.data.concepts
                for concept in concepts:
                    if concept.value > float(confidenceScore):
                        if modelType == 'Color':
                            allConcepts=allConcepts+ ' '+concept.w3c. name
                        else:
                            allConcepts=allConcepts+ ' '+concept.name
            except:
                f = open("/tmp/commands.txt", "w")
                f.write(str(traceback.format_exc()))
                f.close()
    return allConcepts

def get_video_id(value):
    """
    Examples:
    - http://youtu.be/SA2iWivDJiE
    - http://www.youtube.com/watch?v=_oPAwA_Udwc&feature=feedu
    - http://www.youtube.com/embed/SA2iWivDJiE
    - http://www.youtube.com/v/SA2iWivDJiE?version=3&amp;hl=en_US
    """
    query = urlparse.urlparse(value)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    if query.hostname in ('www.youtube.com', 'youtube.com','www.youtube-nocookie.com'):
        if query.path == '/watch':
            p = urlparse.parse_qs(query.query)
            return p['v'][0]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
    # fail?
    return None


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    scene_manager = SceneManager()
    video_manager = VideoManager([video_path])
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    base_timecode = video_manager.get_base_timecode()

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode),video_manager

def generate_html(scenes, image_list, clarifai_descriptions, model_name):
    css ='''
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 1em;
            font-family: sans-serif;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }

        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: center;
        }

        .styled-table th {
            background: #3C444D;
            position: sticky;
            top: 0; /* Don't forget this, required for the stickiness */
            text-align: center;
        }

        .styled-table td {
            padding: 12px 15px;
        }

        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }

        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }

        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #3C444D;
        }
        '''
    _html = html()
    _head, _body = _html.add(head(title('Scene Detection Results')), body())
    _head.add(style(css))
    _table = _body.add(table(cls="styled-table"))
    _thead, _tbody = _table.add(thead(),tbody())
    
    if len(clarifai_descriptions) > 0:
        columns=['Scene #','Timecode', 'Start Frame', 'End Frame', model_name]
        _htr=_thead.add(tr())
        _htr.add([th(col) for col in columns])

        for i in range(len(scenes)):
            _tr = _tbody.add(tr())
            scene_number,timecode,start, end, description= _tr.add(td(i+1),td(),td(),td(),td())
            timecode.add(str(scenes[i][0])+' - '+ str(scenes[i][1]))
            start_link = start.add(a(href=image_list[2*i],target="_blank"))
            start_link.add(img(src=image_list[2*i],width="200"))
            end_link = end.add(a(href=image_list[2*i+1],target="_blank"))
            end_link.add(img(src=image_list[2*i+1],width="200"))
            description.add(clarifai_descriptions[i])
    else:
        columns=['Scene #','Timecode', 'Start Frame', 'End Frame']
        _htr=_thead.add(tr())
        _htr.add([th(col) for col in columns])

        for i in range(int(len(image_list)/2)):
            _tr = _tbody.add(tr())
            scene_number,timecode,start, end = _tr.add(td(i+1),td(),td(),td())
            timecode.add(str(scenes[i][0])+' - '+ str(scenes[i][1]))
            start_link = start.add(a(href=image_list[2*i],target="_blank"))
            start_link.add(img(src=image_list[2*i],width="200"))
            end_link = end.add(a(href=image_list[2*i+1],target="_blank"))
            end_link.add(img(src=image_list[2*i+1],width="200"))
        
    return str(_html)

@Configuration()
class SceneDetect(StreamingCommand):
    
    textfield = Option(require=True, validate=validators.Fieldname())
    
    yolo = Option(require=True, validate=validators.Boolean())
    
    clarifai = Option(require=True, validate=validators.Boolean())
    
    model = Option(require=False)
    
    confidenceScore = Option(require=False)
    
    def stream(self, records):
        for record in records:
            video_url = record[self.textfield]
        
            # get youtube video id from url
            video_id = get_video_id(video_url)

            # create sub folder in static
            BASE_DIR = SAVE_BASE_DIR+video_id
            if not os.path.exists(BASE_DIR):
                os.makedirs(BASE_DIR)

            # download video
            youtube = pytube.YouTube(video_url)
            video = youtube.streams.first()
            if youtube.length <= 240:
                video.download(output_path=BASE_DIR, filename=video_id)
            else:
                continue

            # get video path and detect scenes
            video_path = os.path.join(BASE_DIR, video_id+'.mp4')
            scenes, video_manager = find_scenes(video_path)


            images_path = ''
            VISIT_BASE = ''
            report_name = ''
            model_name = ''
            if self.clarifai:
                model_name = self.model+'_'

            if self.yolo:
                images_path = os.path.join(BASE_DIR, 'yolo_images')
                VISIT_BASE = '/static/app/Multimodal/videos/' + video_id + '/yolo_images/'
                report_name = model_name+'yolo_scene_report.html'
            else:
                images_path = os.path.join(BASE_DIR, 'images')
                VISIT_BASE = '/static/app/Multimodal/videos/' + video_id + '/images/'
                report_name = model_name+'scene_report.html'

            # create sub folder and save images    
            if not os.path.exists(images_path):
                os.makedirs(images_path)

            save_images(scenes,video_manager,num_images=2,output_dir=images_path)

            image_names = []
            for subdir, dirs, files in os.walk(images_path):
                for f in files:
                    if f.endswith(".jpg"):
                        image_names.append(f)

            # YOLO object detection
            if self.yolo:
                # get all images path
                images_list = [os.path.join(images_path, img) for img in image_names]

                for path in images_list:
                    subprocess.call(['/opt/anaconda3/bin/python3.7 scene_detection/yolo.py --image '+ path],shell=True)


            # get all images path (static)
            html_images_list=[os.path.join(VISIT_BASE, img) for img in image_names]
            IP = 'http://35.246.69.64:8000'

            list_scene_number=[]
            list_timecode=[]
            list_start_frame=[]
            list_end_frame=[]
            clarifai_descriptions=[]
            for i in range(len(scenes)):
                scene_number = i+1
                timecode = str(scenes[i][0].get_timecode())+' - '+ str(scenes[i][1].get_timecode())
                start_frame = IP + html_images_list[2*i]
                end_frame = IP + html_images_list[2*i+1]
                description = 'null'

                list_scene_number.append(scene_number)
                list_timecode.append(timecode)
                list_start_frame.append(start_frame)
                list_end_frame.append(end_frame)

                if self.clarifai:
                    description = detectImage(end_frame, self.model, self.confidenceScore, 'True')
                clarifai_descriptions.append(description)
        
            record['video_id'] = video_id
            record['report_name'] = report_name
            record['scene_number'] = [e for e in list_scene_number]
            record['timecode'] = [e for e in list_timecode]
            record['start_frame'] = [e for e in list_start_frame]
            record['end_frame'] = [e for e in list_end_frame]
            record['description'] = [e for e in clarifai_descriptions]
            yield record
#             yield{'video_id': video_id,
#                   'report_name': report_name, 
#                   'scene_number': scene_number,
#                   'timecode': timecode,
#                   'start_frame': start_frame,
#                   'end_frame': end_frame,
#                   'description': description
#                  }
            
            # generate html report
            html = generate_html(scenes, html_images_list, clarifai_descriptions, self.model)

            report_path = os.path.join(BASE_DIR, report_name)
            with open(report_path, "w") as file:
                file.write(html)
        

dispatch(SceneDetect, sys.argv, sys.stdin, sys.stdout, __name__)