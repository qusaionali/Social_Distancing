#imports
import cv2
import numpy as np
import time
import argparse
import utills, plot

confid = 0.5
thresh = 0.5
write_video = False # If you want to write video change True
write_bird_eye_video = False # If you want to write bird eye video change True
write_frames = False # If you want to write outputs(image and bird eye screen) change True
CLAHE_preprocessing = True # If you dont want to preprocess image change False

# I used CLAHE preprocessing algorithm for detect humans better.
# HSV (Hue, Saturation, and Value channel). CLAHE uses value channel.
# Value channel refers to the lightness or darkness of a colour. An image without hue or saturation is a grayscale image.
def improveContrastWithCLAHE(bgr_image: np.array) -> np.array:
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Calculates Social Distancing
# Inputs = video path, network for human detection, output directory for writing frames, 
# output directory for output video and bird eye view's video.
def calculateSocialDistancing(vid_path, net, output_dir, output_vid, ln1):
    
    count = 0
    video_capture = cv2.VideoCapture(vid_path)    

    # Get video height, width and fps
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    
    # Set scale for birds eye view
    # Bird's eye view will only show ROI
    scale_w, scale_h = utills.getScale(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    if write_video:
        output_movie = cv2.VideoWriter("./output_vid/distancing.avi", fourcc, fps, (width, height))
    if write_bird_eye_video:
        bird_movie = cv2.VideoWriter("./output_vid/bird_eye_view.avi", fourcc, fps, (int(width * scale_w), int(height * scale_h)))
        
    points = []
    global image
    
    while True:

        (grabbed, frame) = video_capture.read()

        if CLAHE_preprocessing:
            frame = improveContrastWithCLAHE(frame)

        if not grabbed:
            print('here')
            break
            
        (H, W) = frame.shape[:2]

        # This array created for determine detection area and help to get perspective transform for bird eye view.
        # First 4 points represents detection area and other 3 points for social distance pixel distance.      
        points = [(0, 0), (0, H), (W, H), (W, 0), (80, 20), (100, 0), (180, 0)]                       

        # Using first 4 points or coordinates for perspective transformation.
        # Polygon that used with first 4 points shaped ROI is then warped into a rectangle which becomes the bird eye view. 
        # Bird eye view provides us to estimate true measurement for detections distances. 
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # Using next 3 points for horizontal and vertical unit length(in this case 180 cm)
        pts = np.float32(np.array([points[4:7]])) 
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 180 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 180cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
  
        # YOLO v3
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []   
    
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # detecting humans in frame
                if classID == 0:

                    if confidence > confid:

                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = utills.getTransformedPoints(boxes1, prespective_transform)
        
        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = utills.getDistances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.getCount(distances_mat)
    
        frame1 = np.copy(frame)
        
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = plot.birdEyeView(frame, distances_mat, person_points, scale_w, scale_h, risk_count)
        img = plot.socialDistancingView(frame1, bxs_mat, boxes1, risk_count)
        
        # Show/write image and videos
        if count != 0:
            if write_video:
                output_movie.write(img)
            if write_bird_eye_video:
                bird_movie.write(bird_image)
    
            cv2.imshow('Bird Eye View', bird_image)
            cv2.imshow('Output', img)

            if write_frames:
                cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
                cv2.imwrite(output_dir+"bird_eye_view/frame%d.jpg" % count, bird_image)
    
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    video_capture.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default='./data/example.mp4' ,
                    help='Path for input video')
                    
    parser.add_argument('-o', '--outputDirectory', action='store', dest='outputDirectory', default='./output/' ,
                    help='Path for Output images')
    
    parser.add_argument('-O', '--outputVid', action='store', dest='outputVid', default='./output_vid/' ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    modelPath = values.model
    if modelPath[len(modelPath) - 1] != '/':
        modelPath = modelPath + '/'
        
    outputDirectory = values.outputDirectory
    if outputDirectory[len(outputDirectory) - 1] != '/':
        outputDirectory = outputDirectory + '/'
    
    outputVid = values.outputVid
    if outputVid[len(outputVid) - 1] != '/':
        outputVid = outputVid + '/'


    # load Yolov3 weights
    
    weightsPath = modelPath + "yolov3.weights"
    configPath = modelPath + "yolov3.cfg"

    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]
        
    calculateSocialDistancing(values.video_path, net_yl, outputDirectory, outputVid, ln1)



