#Purpose: Code that students fill in to implement Procrustes Alignment
#and the Iterative Closest Points Algorithm
import numpy as np

#Purpose: To compute the centroid of a point cloud
#Inputs:
#PC: 3 x N matrix of points in a point cloud
#Returns: A 3 x 1 matrix of the centroid of the point cloud
def getCentroid(PC):
    print "TODO"
    return np.zeros((3, 1))

#Purpose: Given an estimate of the aligning matrix Rx that aligns
#X to Y, as well as the centroids of those two point clouds, to
#find the nearest neighbors of X to points in Y
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of Y
#Returns:
#idx: An array of size M which stores the indices of the corresponding
#points in Y to every point in X
def getCorrespondences(X, Y, Cx, Cy, Ry):
    print "TODO"
    #TODO: CHANGE THIS!  This is mapping all points in X to the 
    #first point of Y currently
    idx = np.zeros(X.shape[0])
    return idx

#Purpose: Given correspondences between two point clouds, to center
#them on their centroids and compute the Procrustes alignment to
#align one to the other
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Cx: 3 x 1 matrix of the centroid of X
#Cy: 3 x 1 matrix of the centroid of Y
#idx: An array of size M which stores the indices of the corresponding
#points in Y to every point in X
#Returns:
#Rx: A 3x3 rotation matrix to rotate and align X
#with Y once they have been centered on their centroids
def getProcrustesAlignment(X, Y, Cx, Cy, idx):
    print "TODO"
    #TODO: CHANGE THIS!  It should not be returning the identity
    #matrix each time, as it currently is
    return np.eye(3)

#Purpose: To implement the loop which ties together correspondence finding
#and procrustes alignment to implment the interative closest points algorithm
#Do until convergence (i.e. as long as the correspondences haven't changed)
#Inputs:
#X: 3 x M matrix of points in X
#Y: 3 x N matrix of points in Y (the target point cloud)
#Returns: A tuple of (X centroid, Y centroid, RxList),
#where RxList is a list of rotations matrices Rx that are produced
#as the algorithm runs
#This is all of the information needed to animate exactly
#what the ICP algorithm did
def doICP(X, Y):
    print "TODO"
    RxList = [np.eye(3)]
    #Return centroids of X and Y, as well as a list of rotation matrices
    #that the algorithm produces as it iterates
    return (getCentroid(X), getCentroid(Y), RxList)
