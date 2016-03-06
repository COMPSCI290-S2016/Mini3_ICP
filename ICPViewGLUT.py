from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.arrays import vbo

import sys
sys.path.append("S3DGLPy")
from Primitives3D import *
from PolyMesh import *
from Cameras3D import *
from struct import *
from sys import exit, argv
import numpy as np
import scipy.io as sio
import os
import math
import time
from time import sleep
import matplotlib.pyplot as plt
from ICP import *

class ICPViewerCanvas(object):
    def __init__(self, xmesh, ymesh):
        #GLUT Variables
        self.GLUTwindow_height = 800
        self.GLUTwindow_width = 800
        self.GLUTmouse = [0, 0]
        self.GLUTButton = [0, 0, 0, 0, 0]
        self.GLUTModifiers = 0
        self.bbox = BBox3D()
        
        self.xmesh = xmesh
        self.ymesh = ymesh
        self.displayMeshEdges = False
        self.displayMeshFaces = True
        self.displayMeshPoints = True
        self.displayCorrespondences = True
        self.currCx = np.array([[0, 0, 0]]).T #Current X Centroid
        self.currCy = np.array([[0, 0, 0]]).T #Current Y Centroid
        self.currRx = np.eye(3) #Current rotation
        self.CxList = []
        self.CyList = []
        self.RxList = []
        self.corridx = np.zeros([]) #Current correspondences
        self.corridxbuff = None #Correspondence vertex buffer
        self.maxItersTxt = None
        #Animation variables
        self.animating = False
        self.frameIdx = 0
        self.nearDist = 0.01
        self.farDist = 1000.0
        self.outputPrefixTxt = None
    
    def GLUTResize(self, w, h):
        glViewport(0, 0, w, h)
        self.GLUTwindow_height = 800
        self.GLUTwindow_width = 800
        #Update camera parameters based on new size
        self.camera = MousePolarCamera(w, h)
        self.camera.centerOnBBox(self.bbox, math.pi/2, math.pi/2)
    
    def handleMouseStuff(self, x, y):
        y = self.GLUTwindow_height - y
        self.GLUTmouse[0] = x
        self.GLUTmouse[1] = y
        self.GLUTmodifiers = glutGetModifiers()

    def GLUTMouse(self, button, state, x, y):
        buttonMap = {GLUT_LEFT_BUTTON:0, GLUT_MIDDLE_BUTTON:1, GLUT_RIGHT_BUTTON:2, 3:3, 4:4}
        if state == GLUT_DOWN:
            self.GLUTButton[buttonMap[button]] = 1
        else:
            self.GLUTButton[buttonMap[button]] = 0
        self.handleMouseStuff(x, y)
        glutPostRedisplay()

    def GLUTMotion(self, x, y):
        lastX = self.GLUTmouse[0]
        lastY = self.GLUTmouse[1]
        self.handleMouseStuff(x, y)
        dX = self.GLUTmouse[0] - lastX
        dY = self.GLUTmouse[1] - lastY
        if self.GLUTButton[2] == 1:
            self.camera.zoom(-dY)#Want to zoom in as the mouse goes up
        elif self.GLUTButton[1] == 1:
            self.camera.translate(dX, dY)
        else:
            self.camera.orbitLeftRight(dX)
            self.camera.orbitUpDown(dY)
        glutPostRedisplay()
    
    def displayMeshFacesCheckbox(self, evt):
        self.displayMeshFaces = evt.Checked()
        glutPostRedisplay()

    def displayMeshPointsCheckbox(self, evt):
        self.displayMeshPoints = evt.Checked()
        glutPostRedisplay()

    def displayMeshEdgesCheckbox(self, evt):
        self.displayMeshEdges = evt.Checked()
        glutPostRedisplay()
        
    def displayCorrespondencesCheckbox(self, evt):
        self.displayCorrespondences = evt.Checked()
        glutPostRedisplay()
    
    def getBBoxs(self):
        #Make Y bounding box
        Vy = self.ymesh.getVerticesCols() - self.currCy
        ybbox = BBox3D()
        ybbox.fromPoints(Vy.T)
        
        #Make X bounding box
        Vx = self.xmesh.getVerticesCols() - self.currCx
        Vx = self.currRx.dot(Vx)
        xbbox = BBox3D()
        xbbox.fromPoints(Vx.T)
        
        bboxall = BBox3D()
        bboxall.fromPoints(np.concatenate((Vx, Vy), 1).T)
        self.farDist = bboxall.getDiagLength()*20
        self.nearDist = self.farDist/10000.0
        return (xbbox, ybbox)
    
    #Move the camera to look at the Y mesh (default)
    def viewYMesh(self, evt):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(ybbox, theta = -math.pi/2, phi = math.pi/2)
        glutPostRedisplay()

    #Move the camera to look at the X mesh, taking into consideration
    #current transformation
    def viewXMesh(self, evt):
        (xbbox, ybbox) = self.getBBoxs()
        self.camera.centerOnBBox(xbbox, theta = -math.pi/2, phi = math.pi/2)
        glutPostRedisplay()
    
    def updateCorrBuffer(self):
        X = self.xmesh.VPos.T - self.currCx
        X = self.currRx.dot(X)
        Y = self.ymesh.VPos.T - self.currCy
        idx = self.corridx
        N = idx.size
        C = np.zeros((N*2, 3))
        C[0::2, :] = X.T
        C[1::2, :] = Y.T[idx, :]
        self.corridxbuff = vbo.VBO(np.array(C, dtype=np.float32))
    
    #Call the students' centroid centering code and update the display
    def centerOnCentroids(self, evt):
        self.currCx = getCentroid(self.xmesh.getVerticesCols())
        self.currCy = getCentroid(self.ymesh.getVerticesCols())
        if self.corridxbuff: #If correspondences have already been found
            self.updateCorrBuffer()
        self.viewYMesh(None)

    def findCorrespondences(self, evt):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        self.corridx = getCorrespondences(X, Y, self.currCx, self.currCy, self.currRx)
        self.updateCorrBuffer()
        glutPostRedisplay()
    
    def doProcrustes(self, evt):
        if not self.corridxbuff:
            wx.MessageBox('Must compute correspondences before doing procrustes!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        (self.currCx, self.currCy, self.currRx) = getProcrustesAlignment(X, Y, self.corridx)
        self.updateCorrBuffer()
        glutPostRedisplay()

    def doICP(self, evt):
        X = self.xmesh.getVerticesCols()
        Y = self.ymesh.getVerticesCols()
        MaxIters = 200
        if self.maxItersTxt:
            MaxIters = int(self.maxItersTxt.GetValue())
        (self.CxList, self.CyList, self.RxList) = doICP(X, Y, MaxIters)
        self.currRx = self.RxList[-1]
        self.corridxbuff = None
        self.viewYMesh(None)

    def doAnimation(self, evt):
        if len(self.RxList) == 0:
            wx.MessageBox('Must compute ICP before playing animation!', 'Error', wx.OK | wx.ICON_ERROR)
            return
        self.currRx = self.RxList[0]
        self.animating = True
        self.frameIdx = 0
        glutPostRedisplay()

    def drawPoints(self, mesh):
        glEnableClientState(GL_VERTEX_ARRAY)
        mesh.VPosVBO.bind()
        glVertexPointerf(mesh.VPosVBO)
        glDisable(GL_LIGHTING)
        glPointSize(POINT_SIZE)
        glDrawArrays(GL_POINTS, 0, mesh.VPos.shape[0])
        mesh.VPosVBO.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)

    def drawLines(self, buff, NLines):
        glEnableClientState(GL_VERTEX_ARRAY)
        buff.bind()
        glVertexPointerf(buff)
        glDisable(GL_LIGHTING)
        glPointSize(POINT_SIZE)
        glDrawArrays(GL_LINES, 0, NLines*2)
        buff.unbind()
        glDisableClientState(GL_VERTEX_ARRAY)        

    def setupPerspectiveMatrix(self, nearDist = -1, farDist = -1):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if nearDist == -1:
            farDist = self.camera.eye - self.bbox.getCenter()
            farDist = np.sqrt(farDist.dot(farDist)) + self.bbox.getDiagLength()
            nearDist = farDist/50.0
        gluPerspective(180.0*self.camera.yfov/M_PI, float(self.GLUTwindow_width)/self.GLUTwindow_height, nearDist, farDist)

    def repaint(self):
        if np.isnan(self.camera.eye[0]):
            #TODO: Patch for a strange bug that I can't quite track down
            #where camera eye is initially NaNs (likely a race condition)
            self.viewYMesh(None)
        self.setupPerspectiveMatrix(self.nearDist, self.farDist)
        
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        glEnable(GL_LIGHTING)
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, [0.8, 0.8, 0.8, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [0.2, 0.2, 0.2, 1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, 64)
        
        #glLightfv(GL_LIGHT1, GL_POSITION, np.array([0, 0, 1, 1]))
        self.camera.gotoCameraFrame()     
        P = np.zeros(4)
        P[0:3] = self.camera.eye   
        glLightfv(GL_LIGHT0, GL_POSITION, P)
        
        #Draw the Y mesh
        TYC = np.eye(4)
        TYC[0:3, 3] = -self.currCy.flatten()
        glPushMatrix()
        glMultMatrixd((TYC.T).flatten())
        self.ymesh.renderGL(self.displayMeshEdges, False, self.displayMeshFaces, False, False, True, False)
        if self.displayMeshPoints:
            glColor3f(1.0, 0, 0)
            self.drawPoints(self.ymesh)
        glPopMatrix()
        
        #Draw the X mesh transformed        
        Rx = np.eye(4)
        Rx[0:3, 0:3] = self.currRx
        #Translation to move X to its centroid
        TXC = np.eye(4)
        TXC[0:3, 3] = -self.currCx.flatten()
        T = Rx.dot(TXC)
        glPushMatrix()
        #Note: OpenGL is column major
        glMultMatrixd((T.T).flatten())
        self.xmesh.renderGL(self.displayMeshEdges, False, self.displayMeshFaces, False, False, True, False)
        if self.displayMeshPoints:
            glColor3f(0, 0, 1.0)
            self.drawPoints(self.xmesh)
        glPopMatrix()
        
        if self.displayCorrespondences and self.corridxbuff:
            self.drawLines(self.corridxbuff, self.xmesh.VPos.shape[0])
        
        if self.animating:
            if self.outputPrefixTxt and not(self.outputPrefixTxt.GetValue() == ""):
                #Ouptut screenshots
                prefix = self.outputPrefixTxt.GetValue()
                saveImageGL(self, "%s%i.png"%(prefix, self.frameIdx))
            self.frameIdx += 1
            if self.frameIdx == len(self.RxList):
                self.animating = False
            else:
                self.currCx = self.CxList[self.frameIdx]
                self.currCy = self.CyList[self.frameIdx]
                self.currRx = self.RxList[self.frameIdx]
                glutPostRedisplay()
        glutSwapBuffers()

    def initGL(self):        
        
#        #Buttons to go to a default view
#        viewPanel = wx.BoxSizer(wx.HORIZONTAL)
#        YmeshButton = wx.Button(self, -1, "Y Mesh")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewYMesh, YmeshButton)
#        viewPanel.Add(YmeshButton, 0, wx.EXPAND)
#        XMeshButton = wx.Button(self, -1, "X Mesh")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.viewXMesh, XMeshButton)
#        viewPanel.Add(XMeshButton, 0, wx.EXPAND)
#        self.rightPanel.Add(wx.StaticText(self, label="Views"), 0, wx.EXPAND)
#        self.rightPanel.Add(viewPanel, 0, wx.EXPAND)
#        
#        #Checkboxes for displaying data
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label="Display Options"), 0, wx.EXPAND)
#        self.displayMeshFacesCheckbox = wx.CheckBox(self, label = "Display Mesh Faces")
#        self.displayMeshFacesCheckbox.SetValue(True)
#        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshFacesCheckbox, self.displayMeshFacesCheckbox)
#        self.rightPanel.Add(self.displayMeshFacesCheckbox, 0, wx.EXPAND)
#        
#        self.displayMeshPointsCheckbox = wx.CheckBox(self, label = "Display Mesh Points")
#        self.displayMeshPointsCheckbox.SetValue(True)
#        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshPointsCheckbox, self.displayMeshPointsCheckbox)
#        self.rightPanel.Add(self.displayMeshPointsCheckbox, 0, wx.EXPAND)

#        self.displayMeshEdgesCheckbox = wx.CheckBox(self, label = "Display Mesh Edges")
#        self.displayMeshEdgesCheckbox.SetValue(False)
#        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayMeshEdgesCheckbox, self.displayMeshEdgesCheckbox)
#        self.rightPanel.Add(self.displayMeshEdgesCheckbox, 0, wx.EXPAND)

#        self.displayCorrespondencesCheckbox = wx.CheckBox(self, label = "Display Correspondences")
#        self.displayCorrespondencesCheckbox.SetValue(True)
#        self.Bind(wx.EVT_CHECKBOX, self.glcanvas.displayCorrespondencesCheckbox, self.displayCorrespondencesCheckbox)
#        self.rightPanel.Add(self.displayCorrespondencesCheckbox, 0, wx.EXPAND)

#        #Buttons to test ICP algorithm step by step
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label="ICP Algorithm Step By Step"))
#        CentroidButton = wx.Button(self, -1, "Center Meshes on Centroids")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.centerOnCentroids, CentroidButton)
#        self.rightPanel.Add(CentroidButton, 0, wx.EXPAND)
#        CorrespButton = wx.Button(self, -1, "Find Correspondences")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.findCorrespondences, CorrespButton)
#        self.rightPanel.Add(CorrespButton, 0, wx.EXPAND)
#        ProcrustesButton = wx.Button(self, -1, "Do Procrustes Alignment")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.doProcrustes, ProcrustesButton)
#        self.rightPanel.Add(ProcrustesButton, 0, wx.EXPAND)        
#        
#        #Buttons to compute and test ICP in its entirety
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label=""))
#        self.rightPanel.Add(wx.StaticText(self, label="ICP Algorithm Full"))
#        ComputeICPButton = wx.Button(self, -1, "Compute ICP")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.doICP, ComputeICPButton)
#        self.rightPanel.Add(ComputeICPButton, 0, wx.EXPAND)
#        AnimateICPButton = wx.Button(self, -1, "Animate ICP")
#        self.Bind(wx.EVT_BUTTON, self.glcanvas.doAnimation, AnimateICPButton)
#        self.rightPanel.Add(AnimateICPButton, 0, wx.EXPAND)
#        self.rightPanel.Add(wx.StaticText(self, label="Maximum Iterations"))  
#        self.glcanvas.maxItersTxt = wx.TextCtrl(self)
#        self.glcanvas.maxItersTxt.SetValue("200")
#        self.rightPanel.Add(self.glcanvas.maxItersTxt)
#        self.rightPanel.Add(wx.StaticText(self, label="Output Prefix"))  
#        self.glcanvas.outputPrefixTxt = wx.TextCtrl(self)
#        self.rightPanel.Add(self.glcanvas.outputPrefixTxt)
#        
#        #Finally add the two main panels to the sizer        
#        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
#        self.sizer.Add(self.glcanvas, 2, wx.EXPAND)
#        self.sizer.Add(self.rightPanel, 0, wx.EXPAND)
#        
#        self.SetSizer(self.sizer)
#        self.Layout()
#        self.glcanvas.Show()

        glutInit('')
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(self.GLUTwindow_width, self.GLUTwindow_height)
        glutInitWindowPosition(50, 50)
        glutCreateWindow('ICP Viewer')
        glutReshapeFunc(self.GLUTResize)
        glutDisplayFunc(self.repaint)
        #glutKeyboardFunc(self.GLUTKeyboard)
        #glutKeyboardUpFunc(self.GLUTKeyboardUp)
        #glutSpecialFunc(self.GLUTSpecial)
        #glutSpecialUpFunc(self.GLUTSpecialUp)
        glutMouseFunc(self.GLUTMouse)
        glutMotionFunc(self.GLUTMotion)
        
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1.0])
        glEnable(GL_LIGHT0)
        glLightfv(GL_LIGHT1, GL_DIFFUSE, [0.5, 0.5, 0.5, 1.0])
        glEnable(GL_LIGHT1)
        glEnable(GL_NORMALIZE)
        glEnable(GL_LIGHTING)
        
        glEnable(GL_DEPTH_TEST)
        
        glutMainLoop()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: python ICPViewer.py <mesh to align file> <target mesh file>"
        sys.exit(0)
    (xmeshfile, ymeshfile) = (sys.argv[1], sys.argv[2])
    xmesh = PolyMesh()
    print "Loading %s..."%xmeshfile
    (xmesh.VPos, xmesh.VColors, xmesh.ITris) = loadOffFileExternal(xmeshfile)
    xmesh.performDisplayUpdate(True)
    
    ymesh = PolyMesh()
    print "Loading %s..."%ymeshfile
    (ymesh.VPos, ymesh.VColors, ymesh.ITris) = loadOffFileExternal(ymeshfile)
    ymesh.performDisplayUpdate(True)
    
    viewer = ICPViewerCanvas(xmesh, ymesh)
    viewer.initGL()
