import zipfile
from pathlib import PurePath
import numpy as np
from PIL import Image
from io import BytesIO

import numpy as np
from scipy import ndimage as nd
from glob import glob

def fill(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'. True cells set where data
                 value should be replaced.
                 If None (default), use: invalid  = np.isnan(data)

    Output: 
        Return a filled array. 
    """
    #import numpy as np
    #import scipy.ndimage as nd

    if invalid is None: invalid = np.isnan(data)

    ind = nd.distance_transform_cdt(invalid, metric='taxicab',return_distances=False, return_indices=True)
    return data[tuple(ind)]


class CacheObject(object):

  def clear(self,exceptions=set()):
    for i in (x for x in dir(self) if x.startswith('_cache') and ((x not in exceptions) and "_cache_" + x not in exceptions)):
      delattr(self,i)

from functools import update_wrapper


class LazyProperty(property):
    def __init__(self, method, fget=None, fset=None, fdel=None, doc=None):

        self.method = method
        self.cache_name = "_cache_{}".format(self.method.__name__)

        doc = doc or method.__doc__
        super(LazyProperty, self).__init__(fget=fget, fset=fset, fdel=fdel, doc=doc)

        update_wrapper(self, method)

    def __get__(self, instance, owner):

        if instance is None:
            return self

        if hasattr(instance, self.cache_name):
            result = getattr(instance, self.cache_name)
        else:
            if self.fget is not None:
                result = self.fget(instance)
            else:
                result = self.method(instance)

            setattr(instance, self.cache_name, result)

        return result




import tempfile

class TartanAirFrame(CacheObject):
  def __init__(self,idx,traj):
    self.idx = idx
    self.traj = traj
    self.sidx = str(idx).zfill(6)
  
  @property
  def zips(self):
    return self.traj.zips
  
  @LazyProperty
  def names(self):
    return [x for x in self.traj.names if x.split('/')[-1].startswith(self.sidx)]

  @LazyProperty
  def namedict(self):
    r = {}
    #['seg_right.zip', 'flow_mask.zip', 'seg_left.zip', 'image_right.zip', 'flow_flow.zip', 'depth_left.zip', 'depth_right.zip', 'image_left.zip']
    for n in self.names:
      cn = n.split('/')[-2]
      r[cn] = n
    return r
  
  @LazyProperty
  def image_right(self):
    with self.zips['image_right'].open(self.namedict['image_right']) as p:
      return Image.open(p)
  
  @LazyProperty
  def image_left(self):
    with tempfile.TemporaryDirectory() as tfd:
      p = self.zips['image_left'].extract(self.namedict['image_left'],tfd)
      return Image.open(p)
  
  @LazyProperty
  def seg_right(self):
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['seg_right'].extract(self.namedict['seg_right'],tfd)
        return np.loadtxt(p)
  
  @LazyProperty
  def seg_left(self):
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['seg_left'].extract(self.namedict['seg_left'],tfd)
        return np.loadtxt(p)
  
  @LazyProperty
  def depth_left(self):
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['depth_left'].extract(self.namedict['depth_left'],tfd)
        return np.loadtxt(p)
  
  @LazyProperty
  def depth_right(self):
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['depth_right'].extract(self.namedict['depth_right'],tfd)
        return np.loadtxt(p)
  
  @LazyProperty
  def flow_mask(self):
    with self.zips['flow_mask'].open(self.namedict['flow'][:-8] + "mask.npy") as f:
        return np.load(BytesIO(f.read()))
  
  @LazyProperty
  def flow_flow(self):
    with self.zips['flow_flow'].open(self.namedict['flow']) as f:
        return np.load(BytesIO(f.read()))

  @property
  def flow_mask_same_dim(self):
    return np.repeat(self.flow_mask[:,:,np.newaxis],2,axis=2)

  @LazyProperty
  def flow_mask_filled(self):
    return fill(self.flow_flow,(self.flow_mask_same_dim > 0))

  @LazyProperty
  def flow_variance(self):
    # with self:
        return np.var(np.std(self.flow_mask_filled,axis=2))




class TartanAirTrajectory(CacheObject):
  def __init__(self,idx,scene):
    self.idx = idx
    self.sidx = "P" + str(idx).zfill(3)
    self.scene = scene
    tmplst = (x.split('/') for x in self.names)
    self.indices = [int(x[-1].split('_')[0]) for x in tmplst if not x[-1].startswith('pose')]
    self._cache_frames = {}
  
  @property
  def zips(self):
    return self.scene.zips

  @LazyProperty
  def names(self):
    return [x for x in self.scene.names if self.sidx in x]
  
  @LazyProperty
  def pose_left_name(self):
    return next(iter((x for x in self.names if x.endswith('pose_left.txt'))))
  
  @LazyProperty
  def pose_left(self):
    n = self.pose_left_name
    zf = next(iter(self.scene.zips.items()))[1]
    r = None
    with tempfile.TemporaryDirectory() as tfd:
      r = np.loadtxt(zf.extract(n,tfd))
    return r

  @LazyProperty
  def pose_right_name(self):
    return next(iter((x for x in self.names if x.endswith('pose_right.txt'))))
  
  @LazyProperty
  def pose_right(self):
    n = self.pose_right_name
    zf = next(iter(self.scene.zips.items()))[1]
    r = None
    with tempfile.TemporaryDirectory() as tfd:
      r = np.loadtxt(zf.extract(n,tfd))
    return r

  @property
  def poses(self):
    return pose_left,pose_right
  
  def __len__(self):
    return len(self.indices)
  
  def __getitem__(self,idx):
      if not idx in self.indices:
        raise IndexError()
      if not hasattr(self,'_cache_frames'):
        self._cache_frames = {}
      if not idx in self._cache_frames:
        self._cache_frames[idx] = TartanAirFrame(idx,self)
      return self._cache_frames[idx]
    

class TartanAirScene(CacheObject):
  def __init__(self,path,pathsandzips=None):
    self.path = path
    self.zips = {}
    self.expected = ['seg_right.zip', 'flow_mask.zip', 'seg_left.zip', 'image_right.zip', 'flow_flow.zip', 'depth_left.zip', 'depth_right.zip', 'image_left.zip']
    if pathsandzips is None:
      self.pathsfound = [x for x in self.expected if glob(path + '/' + "*" + x) != []]
      self.zipfilepaths = [glob(path + '/' + "*" + x)[0] for x in self.pathsfound]
    else:
      self.pathsfound = pathsandzips[0]
      self.zipfilepaths = pathsandzips[1]
    
    #self.scene_name = "_".join(self.zipfilepaths[0].split('/')[-1].split('_')[0:2])
    self.zipfiles = [zipfile.ZipFile(x) for x in self.zipfilepaths]
    self.zips = dict(zip([x.split('.')[0] for x in self.pathsfound],self.zipfiles))
    self.indices = np.unique([int(x.split('/')[3][1:]) for x in next(iter(self.zips.items()))[1].namelist()]).tolist()
    self.purepath = PurePath(next(iter(self.zips.items()))[1].namelist()[0]).parents[2]
    # print(list(self.purepath.parents))
    self.scene_name = str(list(self.purepath.parents)[-2])
    self.names = []
    for i in (x[1].namelist() for x in self.zips.items()):
      self.names.extend(i)
    self._cache_trajectories = {}

  def __len__(self):
    return len(self.indices)
  
  def __getitem__(self,idx):
    if isinstance(idx,str):
      return self.zips[idx]
    if isinstance(idx,int):
      if not idx in self.indices:
        raise IndexError()
      if not hasattr(self,'_cache_trajectories'):
        self._cache_trajectories = {}
      if not idx in self._cache_trajectories:
        self._cache_trajectories[idx] = TartanAirTrajectory(idx,self)
      return self._cache_trajectories[idx]

   def __iter__(self):
     for x in self.indices:
       yield self[x]



def _calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift
    
def flow2vis(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = _calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if ( mask is not None ):
        mask = mask > 0
        rgb[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return rgb