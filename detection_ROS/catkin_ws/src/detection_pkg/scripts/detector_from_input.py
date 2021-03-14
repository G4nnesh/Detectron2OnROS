#!/home/elmehdi_c/anaconda3/envs/Deep/bin/python

#For rospy
import rospy
from std_msgs.msg import String
import cv_bridge
from sensor_msgs.msg import Image

#For script
import multiprocessing as mp
from collections import deque
import cv2
import torch
import os

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

from PredictorFromModel import Predicteur, AsyncPredictor


# Utilisation du prédicteur
class Visualization:
    def __init__(self, instance_mode=ColorMode.IMAGE, parallel=False):

        cfg = get_cfg() #initialisation d'une cfg par défaut pour fournir les noms de class au Catlog visualisé
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu") #servira pour copier les arrays images, numpy n'étant pas compatible avec le gpu
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(num_gpus=num_gpu)
        else:
            self.predictor = Predicteur()

    def _frame_from_video(self, video): # Lecture des videos en frame par frame
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    #Utilisation de la panoptic_seg de panoptic lab pour afficher les masques des objets sur une video
    def run_on_video(self, video):

        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                vis_frame = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            elif "instances" in predictions:
                predictions = predictions["instances"].to(self.cpu_device)
                vis_frame = video_visualizer.draw_instance_predictions(frame, predictions)
            elif "sem_seg" in predictions:
                vis_frame = video_visualizer.draw_sem_seg(
                    frame, predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )

            # conversion vers BGR pour openCv
            vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
            return vis_frame

        frame_gen = self._frame_from_video(video)

        if self.parallel: #le rendering / plusieurs gpus est activé, pas encore supporté ici
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor.predictions(frame))

        #TODO une output Texte direct si le tracking sur une video est lent sur le kit !

camera = 0          #port camera

def detector_from_input_1():
    pub = rospy.Publisher('obstacles', Image, queue_size=10)       #Publication dans le topic obstacles
    rospy.init_node('detector_from_input_1', anonymous=True)        #init de la node
    rate = rospy.Rate(10)                                           #rate

    #Log
    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()

    objectPredictor = Visualization()

    # Camera
    assert camera != -1, "Bad camera port!"
    cam = cv2.VideoCapture(camera)
    cam_ros_type = cv_bridge.CvBridge() #Conversion vers le format image de ros

    while not rospy.is_shutdown():

        for vis in objectPredictor.run_on_video(cam):
            pub.publish(cam_ros_type.cv2_to_imgmsg(vis) 

if __name__ == '__main__':
    try:
        detector_from_input_1()
    except rospy.ROSInterruptException:
        pass