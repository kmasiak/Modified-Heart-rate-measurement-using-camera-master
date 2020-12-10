import cv2
import numpy as np
import dlib
from imutils import face_utils
import imutils
from face_recognition import api
import time

class FaceDetection(object):
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.fa = face_utils.FaceAligner(self.predictor, desiredFaceWidth=256)
        self.facecount = 0
        self.framecount = 0
        self.starttime = 0

    def face_detect(self, frame):
        if frame is None:
            #print("facecount:" + str(self.facecount))
            #print("framecount:" + str(self.framecount))
            return
        frame = imutils.resize(frame, width=1000)
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        status = False

        frame_shape_x = frame.shape[1]
        frame_shape_y = frame.shape[0]
        rgb_frame = frame[:, :, ::-1]
        # detect faces in the rgb image
        rects = api.face_detector(rgb_frame)
        xscale = 640 / frame_shape_x
        yscale = 480 / frame_shape_y



        if len(rects)>0:
            status = True

            #(x, y, w, h) = face_utils.rect_to_bb(rects[0])

            #if y<0:
                #print("a")
                #return frame, face_frame, ROI1, ROI2, status, mask

            #face_frame = frame[y:y+h,x:x+w]

            #if(face_frame.shape[:2][1] != 0):
                #face_frame = imutils.resize(face_frame,width=256)
                
            face_frame = self.fa.align(frame,rgb_frame,rects[0])
            rgb_frame_f = face_frame[:, :, ::-1]
            #rectsf = api.face_detector(rgb_frame_f)
            r = dlib.rectangle(0, 0, 256, 256)
            if len(rects) > 0:

                shape = self.predictor(rgb_frame_f, r)
                shape = face_utils.shape_to_np(shape)

                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks

                #cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 0, 255), 1)

                cv2.rectangle(face_frame,(shape[54][0], shape[29][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[33][1]), (0,255,0), 0)
                cv2.rectangle(face_frame, (shape[4][0], shape[29][1]),
                        (shape[48][0],shape[33][1]), (0,255,0), 0)
                
                ROI1 = face_frame[shape[29][1]:shape[33][1], #right cheek
                        shape[54][0]:shape[12][0]]
                        
                ROI2 =  face_frame[shape[29][1]:shape[33][1], #left cheek
                        shape[4][0]:shape[48][0]]    


                rshape = np.zeros_like(shape) 
                rshape = self.face_remap(shape)


                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))
            
                cv2.fillConvexPoly(mask, rshape[0:27], 1) 

        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False

        frame = cv2.resize(frame, (640, 480))
        return frame, face_frame, ROI1, ROI2, status, mask


    def face_detect_artery(self, frame):
        if frame is None:
            #print("facecount:" + str(self.facecount))
            #print("framecount:" + str(self.framecount))
            return
        frame = imutils.resize(frame, width=1000)
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        status = False

        frame_shape_x = frame.shape[1]
        frame_shape_y = frame.shape[0]
        rgb_frame = frame[:, :, ::-1]
        # detect faces in the rgb image
        rects = api.face_detector(rgb_frame)
        xscale = 640 / frame_shape_x
        yscale = 480 / frame_shape_y



        if len(rects)>0:
            status = True

            #(x, y, w, h) = face_utils.rect_to_bb(rects[0])

            #if y<0:
                #print("a")
                #return frame, face_frame, ROI1, ROI2, status, mask

            #face_frame = frame[y:y+h,x:x+w]

            #if(face_frame.shape[:2][1] != 0):
                #face_frame = imutils.resize(face_frame,width=256)

            face_frame = self.fa.align(frame,rgb_frame,rects[0])
            rgb_frame_f = face_frame[:, :, ::-1]
            #rectsf = api.face_detector(rgb_frame_f)
            r = dlib.rectangle(0, 0, 256, 256)
            if len(rects) > 0:

                shape = self.predictor(rgb_frame_f, r)
                shape = face_utils.shape_to_np(shape)

                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks

                #cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 0, 255), 1)

                #print('face_frame', face_frame)

                cv2.rectangle(face_frame,(shape[25][0], shape[15][1]), #draw rectangle on right and left cheeks
                        (shape[12][0],shape[6][1]), (0,255,0), 0)
                #cv2.rectangle(face_frame, (shape[27][0], shape[15][1]),  # draw rectangle on right and left cheeks
                #              (shape[13][0], shape[5][1]), (0, 255, 0), 0)
                cv2.rectangle(face_frame, (shape[18][0], shape[15][1]),
                        (shape[2][0],shape[6][1]), (0,255,0), 0)

                ROI1 = face_frame[shape[15][1]:shape[6][1], #right cheek
                        shape[25][0]:shape[12][0]]

                #ROI2 =  face_frame[shape[15][1]:shape[6][1], #left cheek
                #        shape[20][0]:shape[2][0]]
                ROI2 = face_frame[shape[29][1]:shape[33][1],  # left cheek
                       shape[4][0]:shape[48][0]]

                rshape = np.zeros_like(shape)
                rshape = self.face_remap(shape)


                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))

                cv2.fillConvexPoly(mask, rshape[0:27], 1)

        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False

        frame = cv2.resize(frame, (640, 480))
        return frame, face_frame, ROI1, ROI2, status, mask

    def face_detect_forehead(self, frame):
        if frame is None:
            #print("facecount:" + str(self.facecount))
            #print("framecount:" + str(self.framecount))
            return
        frame = imutils.resize(frame, width=1000)
        face_frame = np.zeros((10, 10, 3), np.uint8)
        mask = np.zeros((10, 10, 3), np.uint8)
        ROI1 = np.zeros((10, 10, 3), np.uint8)
        ROI2 = np.zeros((10, 10, 3), np.uint8)
        status = False

        frame_shape_x = frame.shape[1]
        frame_shape_y = frame.shape[0]
        rgb_frame = frame[:, :, ::-1]
        # detect faces in the rgb image
        rects = api.face_detector(rgb_frame)
        xscale = 640 / frame_shape_x
        yscale = 480 / frame_shape_y



        if len(rects)>0:
            status = True

            #(x, y, w, h) = face_utils.rect_to_bb(rects[0])

            #if y<0:
                #print("a")
                #return frame, face_frame, ROI1, ROI2, status, mask

            #face_frame = frame[y:y+h,x:x+w]

            #if(face_frame.shape[:2][1] != 0):
                #face_frame = imutils.resize(face_frame,width=256)

            face_frame = self.fa.align(frame,rgb_frame,rects[0])
            rgb_frame_f = face_frame[:, :, ::-1]
            #rectsf = api.face_detector(rgb_frame_f)
            r = dlib.rectangle(0, 0, 256, 256)
            if len(rects) > 0:

                shape = self.predictor(rgb_frame_f, r)
                shape = face_utils.shape_to_np(shape)

                for (a, b) in shape:
                    cv2.circle(face_frame, (a, b), 1, (0, 0, 255), -1) #draw facial landmarks

                #cv2.rectangle(frame, (x, y), (w + x, h + y), (0, 0, 255), 1)

                #print('face_frame', face_frame)

                cv2.rectangle(face_frame,(shape[21][0], shape[23][1]), #draw rectangle on right and left cheeks
                        (shape[22][0],shape[28][1]), (0,255,0), 0)
                #cv2.rectangle(face_frame, (shape[27][0], shape[15][1]),  # draw rectangle on right and left cheeks
                #              (shape[13][0], shape[5][1]), (0, 255, 0), 0)
                cv2.rectangle(face_frame, (shape[21][0], shape[23][1]),
                        (shape[22][0],shape[28][1]), (0,255,0), 0)

                ROI1 = face_frame[shape[23][1]:shape[28][1], #right cheek
                        shape[21][0]:shape[22][0]]

                #ROI2 =  face_frame[shape[15][1]:shape[6][1], #left cheek
                #        shape[20][0]:shape[2][0]]
                ROI2 = face_frame[shape[23][1]:shape[28][1],  # left cheek
                       shape[21][0]:shape[22][0]]

                rshape = np.zeros_like(shape)
                rshape = self.face_remap(shape)


                mask = np.zeros((face_frame.shape[0], face_frame.shape[1]))

                cv2.fillConvexPoly(mask, rshape[0:27], 1)

        else:
            cv2.putText(frame, "No face detected",
                       (200,200), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255),2)
            status = False

        frame = cv2.resize(frame, (640, 480))
        return frame, face_frame, ROI1, ROI2, status, mask

    # some points in the facial landmarks need to be re-ordered
    def face_remap(self,shape):
        remapped_image = shape.copy()
        # left eye brow
        remapped_image[17] = shape[26]
        remapped_image[18] = shape[25]
        remapped_image[19] = shape[24]
        remapped_image[20] = shape[23]
        remapped_image[21] = shape[22]
        # right eye brow
        remapped_image[22] = shape[21]
        remapped_image[23] = shape[20]
        remapped_image[24] = shape[19]
        remapped_image[25] = shape[18]
        remapped_image[26] = shape[17]
        # neatening 
        remapped_image[27] = shape[0]
        
        remapped_image = cv2.convexHull(shape)
        return remapped_image       
        
        
        
        
        
