# Social_DistancingThis algorithm estimates social distance between humans. Social distancing is very important for protecting against COVID-19. 
We must follow the rules for out health. This algorithm created for analyzing social distancing rules. 
Algorithm calculates distance between humans. Distance is bounded by 180 centimeters. If humans close each other fewer than 150 cm, you can see humans in red bounding box.
Otherwise between 150-180, you can see humans in yellow bounding box. Others safe zone in green bounding box. This algorithm uses YOLO-v3 for human detection. 
Used CLAHE preprocessing algorithm for better detections. 

Requirements
Python 3.6
Opencv 4.2.0 or above
numpy 1.14.5
argparse


Weights not added for sizes you can download from this link: https://pjreddie.com/media/files/yolov3.weights
<img src=“https://github.com/KubraTurker/Social_Distancing/blob/master/example/example1.png 1”>
