from PyQt5.QtWidgets import (QWidget, QScrollBar, QVBoxLayout, QLabel, QApplication)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from scipy.spatial.transform import Rotation as R
import sys, cv2, uuid
import numpy as np
import matplotlib.image as Mimage
# from PIL import Image, ImageDraw, ImageFont

# testing of viewpoint transform from cam 1 into cam 2 [virtual camera]
    # focusing on extrinsic (yaw pitch roll)
    # translation [cam height] is kept unchanged
    # intrinsic is kept unchanged [FOV cam 1 = FOV cam 2]
# camera 1 : camera that took the original image
# camera 2 : virtual camera having the same camera center as camera 1, then rotating around to test

class RotationTuning(QWidget):
    def __init__(self):
        super().__init__()                
        # # XYZ_ego's origin is on the ground (rear axle center orthogonal-projection on the ground)
        # # X ego : rear to front
        # # Y ego : right to left
        # # Y ego :

        self.wheelradius      = 0
        # XYZ_ego's origin is on the ground (rear axle center orthogonal-projection on the ground)
        # X ego : rear to front
        # Y ego : right to left
        # Z ego : upside toward the sky
        self.trans_x          = 0.8571999277380136 # -1.4m to be at wheelbase center
        self.trans_y          = -0.1526472937107812
        self.trans_z          = 1.623767855306573  # already + 0.362 [wheel radius]
        self.translation      = np.array([self.trans_x, self.trans_y, self.trans_z])        
                
        ## CAMERA 1 # || cam 1 - vehicle matrix
        # self.cam1_roll       = -1.58159
        # self.cam1_pitch      =  0.03591
        # self.cam1_yaw        = -1.59909        
        
        self.cam1_roll       = -1.58159
        self.cam1_pitch      =  0.03591
        self.cam1_yaw        = -1.59909       
        
        self.cam1_vehicle    = np.identity(4)
        self.cam1_vehicle[:3,:3] \
                              = (R.from_euler('ZYX', [self.cam1_yaw, self.cam1_pitch, self.cam1_roll])).as_matrix()
        self.cam1_vehicle[:3, 3] \
                              = self.translation
                              
                    # || vehicle - cam 1 matrix
        self.vehicle_cam1    = np.linalg.inv(self.cam1_vehicle)    
        
        # print('self.cam1_vehicle \n', self.cam1_vehicle )
        # print('self.vehicle_cam1 \n', self.vehicle_cam1)
      

        ## CAMERA 2 # || cam 2 - vehicle matrix
        self.cam2_roll     = -1.58159
        self.cam2_pitch    =  0.03591
        self.cam2_yaw      = -1.59909
        
        self.cam2_vehicle     = np.identity(4)
        self.cam2_vehicle[:3,:3] \
                              = (R.from_euler('ZYX', [self.cam2_yaw, self.cam2_pitch, self.cam2_roll])).as_matrix()
        self.cam2_vehicle[:3, 3] \
                              = self.translation
                              
                    # || vehicle - cam 2 matrix
        self.vehicle_cam2    = np.linalg.inv(self.cam2_vehicle)
        
        # print('self.cam2_vehicle \n', self.cam2_vehicle )
        # print('self.vehicle_cam2 \n', self.vehicle_cam2)
        
        
        ## || intrinsic matrix, cam 1 - cam 2 has the same
        self.fx             = 2617.92
        self.fy             = 2617.84
        self.mtx            = np.array([[self.fx, 0.000000, 952.30],
                                        [0.000000, self.fy, 551.57],
                                        [0.000000, 0.000000, 1.000000]])        
        self.dist           = np.array([-0.333839, 0.114792, -0.003408, 0.000579, 0.0])
        
        self.K_inverse      = np.linalg.inv(self.mtx)
        print('self.K_inverse ', self.K_inverse)        
        
        ## Load original image taken by camera 1
        self.image          = np.float32(cv2.imread('frame0002.jpg', cv2.IMREAD_COLOR))
        self.image          = np.uint8(self.image)

                    # Image undistort"""
        self.image          = cv2.undistort(self.image, self.mtx, self.dist)
        
        # # CAM1-image-points SET
        #   u v 
        self.cam1_impoints = np.array([[74,    1170],                   # P0 # bottom-left,
                                        [1556,  1204],                  # P1 # bottom-right,
                                        [1030,   868],                  # P2 # top-right,
                                        [814,    868]], dtype=np.int32) # P3 # top-left
        
        """ IMG-points-Cam1 >> ego/world """
        """ np.asarray(pointProjected2D).astype('int')"""
        
        self.homopoints_cam1coordinates_4x_ = np.empty((0,4)) #np.array([[]]) # set of homo points in camera 1 coordinate [4 corners point]
        self.homopoints_egocoordinates_4x_  = np.empty((0,4)) #np.array([[]]) # set of homo points in ego coordinate [4 corners point]
        
        img_point                       = np.array([0, 0, 1], np.float32) # 1 image point [u v 1]
        for i in range(self.cam1_impoints.shape[0]):
            # 1 image point [u v 1]            
            img_point[0:2]      = self.cam1_impoints[i, :]    

            # 1 point in camera-1's homogenenous plan, expressed in camera 1 coordinates
            # [x y z=1]
            homopoint_cam1coordinates \
                                = np.matmul(self.K_inverse, img_point.T)                                
            homopoint_cam1coordinates_4x1 \
                                = np.append(homopoint_cam1coordinates, 1)            

            # homopoint_cam1coordinates converted into world coordinates
            homopoint_egocoordinates_4x1\
                                = np.matmul(self.cam1_vehicle, homopoint_cam1coordinates_4x1)                                
            
            # append in to point array
            self.homopoints_cam1coordinates_4x_\
                                = np.append(self.homopoints_cam1coordinates_4x_, [homopoint_cam1coordinates_4x1], axis=0)
            
            self.homopoints_egocoordinates_4x_\
                                = np.append(self.homopoints_egocoordinates_4x_, [homopoint_egocoordinates_4x1], axis=0)                             
            #
            # self.homopoints_cam1coordinates_4x_ = [] # set of homo points in camera 1 coordinate [4 corners point]
            # self.homopoints_egocoordinates_4x_  = [] # set of homo points in ego coordinate [4 corners point]
            # # append to set of points
            # self.homopoints_cam1coordinates_4x_.append(homopoint_cam1coordinates_4x1) 
            # self.homopoints_egocoordinates_4x_.append(homopoint_egocoordinates_4x1)     
            #
            # print('homopoint_cam1coordinates_4x1 ', homopoint_cam1coordinates_4x1)
            # print('homopoint_egocoordinates_4x1 ', homopoint_egocoordinates_4x1)
            
        # Project image
        self.visualizeProjection()

        # Init the GUI
        self.initUI()

    def initUI(self):
        vbox = QVBoxLayout()
        
    # # *the scroller UI can set only positive value, ex [0..628*2*2], so need to center value of yaw/pitch/roll [-pi..+pi]
    #           in a positive scale of the scroller UI
    #           scroller's value  
        
        """# roll"""
        sld1 = QScrollBar(Qt.Horizontal, self)
        sld1.setRange(0, 628*2*2)
        sld1.setValue(int((self.cam2_roll + np.pi)*400))
        sld1.setFocusPolicy(Qt.NoFocus)
        sld1.setPageStep(1)
        sld1.valueChanged.connect(self.updateLabelRoll)
        self.label1 = QLabel('roll X-ego 3rd-rotation --pitch effect | ' + str(round((self.cam2_roll + np.pi)/np.pi*180, 2)) + ' degree', self)
        self.label1.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label1.setMinimumWidth(80)        
        vbox.addWidget(sld1)
        vbox.addSpacing(5)
        vbox.addWidget(self.label1)

        """# pitch"""
        sld2 = QScrollBar(Qt.Horizontal, self)
        sld2.setRange(0, 628*2*2)
        sld2.setValue(int((self.cam2_pitch + np.pi)*400))    
        sld2.setFocusPolicy(Qt.NoFocus)
        sld2.setPageStep(1)
        sld2.valueChanged.connect(self.updateLabelPitch)
        self.label2 = QLabel('pitch Y-ego 2nd-rotation --roll effect | ' + str(round((self.cam2_pitch + np.pi)/np.pi*180, 2)) + ' degree', self)
        self.label2.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label2.setMinimumWidth(80)
        vbox.addWidget(sld2)
        vbox.addSpacing(5)
        vbox.addWidget(self.label2)
        
        """# yaw"""
        sld3 = QScrollBar(Qt.Horizontal, self)
        sld3.setRange(0, 628*2*2)
        sld3.setValue(int((self.cam2_yaw + np.pi)*400))             
        sld3.setFocusPolicy(Qt.NoFocus)
        sld3.setPageStep(1)
        sld3.valueChanged.connect(self.updateLabelYaw)
        self.label3 = QLabel('yaw Z-ego 1st-rotation --yaw effect | ' + str(round((self.cam2_yaw + np.pi)/np.pi*180, 2)) + ' degree', self)
        self.label3.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.label3.setMinimumWidth(80)    
        vbox.addWidget(sld3)
        vbox.addSpacing(5)
        vbox.addWidget(self.label3)
         
        #
        self.setLayout(vbox)
        self.setGeometry(300, 300, 700, 250)
        self.setWindowTitle('Rotation tuning')        
        self.show()

    # # *the scroller UI can set only positive value, ex [0..628*2*2], so need to center value of yaw/pitch/roll [-pi..+pi]
    #           in a positive scale of the scroller UI
    #           scroller's value
    
    def updateLabelRoll(self, value):
        # self.label1.setText('roll ' + str(round(value/400.0  - np.pi, 5)) + ' rad')
        self.label1.setText('roll X-ego 3rd rotation --pitch effect | ' + str(round((value/400.0 - np.pi)/np.pi*180, 2)) + ' degree')
        self.cam2_roll = value/400.0 - np.pi
        self.visualizeProjection()
    
    def updateLabelPitch(self, value):
        # self.label2.setText('pitch ' + str(round(value/400.0  - np.pi, 5)) + ' rad')
        self.label2.setText('pitch Y-ego 2nd rotation --roll effect | ' + str(round((value/400.0 - np.pi)/np.pi*180, 2)) + ' degree')
        self.cam2_pitch = value/400.0 - np.pi
        self.visualizeProjection()

    def updateLabelYaw(self, value):
        # self.label3.setText('yaw ' + str(round(value/400.0  - np.pi, 5)) + ' rad')
        self.label3.setText('yaw Z-ego 1st rotation --yaw effect | ' + str(round((value/400.0 - np.pi)/np.pi*180, 2)) + ' degree')
        self.cam2_yaw = value/400.0 - np.pi
        self.visualizeProjection()
               
    def visualizeProjection(self):
        # update camera 2 extrinsic zith new input values
        # # rotation matrix updated, trnaslation unchanged
        self.cam2_vehicle     = np.identity(4)
        self.cam2_vehicle[:3,:3] \
                              = (R.from_euler('ZYX', [self.cam2_yaw, self.cam2_pitch, self.cam2_roll])).as_matrix()
        self.cam2_vehicle[:3, 3] \
                              = self.translation
                              
                    # || vehicle - cam 2 matrix
        self.vehicle_cam2    = np.linalg.inv(self.cam2_vehicle)
        
        # print('self.cam2_vehicle \n', self.cam2_vehicle )
        # print('self.vehicle_cam2 \n', self.vehicle_cam2)
        
        # ego/world projection into camera 2
        """ ego/world projection >> camera 2 """
        self.homopoints_cam2coordinates_4x_ = np.empty((0,4)).astype(np.float32)
        self.cam2_impoints                  = np.empty((0,2)).astype(np.int32)
        
        # # convert homopoints_egocoordinates_4x_ into camera-2 's coordinates
        for i in range(self.homopoints_egocoordinates_4x_.shape[0]):
            # 1 image point [u v 1]            
            _point_4x1          = self.homopoints_egocoordinates_4x_[i, :]

            # 1 point of homopoints_egocoordinates_4x_ converted in camera-2 's coordinates
            # [x y z 1]
            _cam2coordinates_4x1 \
                                = np.matmul(self.vehicle_cam2, _point_4x1)
            _cam2coordinates_3x1 \
                                = np.array([_cam2coordinates_4x1[0:3]], np.float32)
                                
            print('_cam2coordinates_3x1 \n', _cam2coordinates_3x1)
                                
            # # project _cam2coordinates_4x1 >> camera-2's homogeneous plan
            # [x/z y/z 1] 3x1 vector of projected _cam2coordinates_4x1 into camera-2's homogeneous
            x                   = _cam2coordinates_4x1[0]/_cam2coordinates_4x1[2]
            y                   = _cam2coordinates_4x1[1]/_cam2coordinates_4x1[2]
            z                   = 1
            homo2_cam2coordinates_3x1 \
                                = np.array([x, y, 1], np.float32)
            
            # # image point of camera-2 created by homo2_cam2coordinates_3x1
            # [u v 1] pixel point created by homo2_cam2coordinates_3x1
            # u v are integers            
            img_point_cam2      = (np.matmul(self.mtx, _cam2coordinates_3x1.T)).astype(np.int32)
            
            # print('img_point_cam2 \n', img_point_cam2)
            # print('img_point_cam2[0:2].T \n', img_point_cam2[0:2].T)
            
            
            self.homopoints_cam2coordinates_4x_ \
                                = np.append(self.homopoints_cam2coordinates_4x_, [_cam2coordinates_4x1], axis=0)
            
            print('self.homopoints_cam2coordinates_4x_ \n', self. homopoints_cam2coordinates_4x_)
            
            self.cam2_impoints \
                                = np.append(self.cam2_impoints, img_point_cam2[0:2].T, axis=0)

        print('self.cam2_impoints \n', self.cam2_impoints)
        
        # # warp perspectives        
        #   source-points on the original image taken by camera-1
                #       openCV takes tuple for a point
        self.cam1_srcpoints     = [tuple(x.astype(np.int32)) for x in self.cam1_impoints[:4]] # 4 sources point for image warping       

        image_cp                = np.copy(self.image) # Copy image
        for i in range(len(self.cam1_srcpoints)):
            if self.cam1_srcpoints[i][0] > 0 and self.cam1_srcpoints[i][1] > 0:
                colori = (0, 0, 255)      
                cv2.circle(image_cp, self.cam1_srcpoints[i], 4, colori, -1)

        self.cam1_image           = cv2.resize(image_cp, (960, 640))
        cv2.namedWindow('Camera 1 - Original image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera 1 - Original image', 960, 640)    #resize the window
        cv2.imshow('Camera 1 - Original image', self.cam1_image)                       
                # cv2.imwrite('camera_1_original_image' + '.png', image_cp)
                ## cv2.waitKey(5000)
                # cv2.destroyAllWindows()
                
        # # destination-points on the original image taken by camera-1
                #       openCV takes tuple for a point
        self.cam2_dstpoints     = [tuple(x.astype(np.int32)) for x in self.cam2_impoints[:4]] # 4 sources point for image warping
        # image_cp                = np.zeros_like(self.image) # [0 0 0] image
        
        """" warping image camera-1 >> camera-2"""
        # # Compute the perspective transform M
        size                    = (self.image.shape[1], self.image.shape[0])
        print('size', size)
        
        
        self.cam1_srcpoints     = np.float32(self.cam1_srcpoints) # convert array of tuple into 2 dimensions matrix, float32 
        self.cam2_dstpoints     = np.float32(self.cam2_dstpoints) # convert array of tuple into 2 dimensions matrix, float32
        
        print('self.cam1_srcpoints \n', self.cam1_srcpoints)
        print('self.cam2_dstpoints \n', self.cam2_dstpoints)
        
            # # *cv2.getPerspectiveTransform takes float32 as input, not intergers
        M                       = cv2.getPerspectiveTransform(self.cam1_srcpoints , self.cam2_dstpoints)
        self.cam2_image         = cv2.warpPerspective(self.image, M, size, flags=1)
        
        for i in range(len(self.cam2_dstpoints)):
            if self.cam2_dstpoints[i][0] > 0 and self.cam2_dstpoints[i][1] > 0:
                colori = (0, 255, 255)      
                cv2.circle(self.cam2_image, tuple(np.int32(self.cam2_dstpoints[i])), 4, colori, -1)
                
        self.cam2_image           = cv2.resize(self.cam2_image, (960, 640))
        cv2.namedWindow('Camera 2 - Virtual image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera 2 - Virtual image', 960, 640)    #resize the window
        cv2.imshow('Camera 2 - Virtual image', self.cam2_image)
        
        # print('self.homopoints_cam1coordinates_4x_ \n', self.homopoints_cam1coordinates_4x_)
        # print('self.homopoints_egocoordinates_4x_  \n', self.homopoints_egocoordinates_4x_)
        # print('self.homopoints_cam2coordinates_4x_ \n', self.homopoints_cam2coordinates_4x_)
        
def main():
    app         = QApplication(sys.argv)
    ex          = RotationTuning()
    app.exec_() # sys.exit(app.exec_())
    
    # # save iamge when exit    
    cv2.imwrite('camera_1_original_image' + '.png', ex.cam1_image)
    cv2.imwrite('camera_2_virtual_image' + '.png', ex.cam2_image)
    cv2.destroyAllWindows()
    
    # # system exit
    sys.exit()
    
if __name__ == '__main__':
    main()