import zipfile
from pathlib import PurePath
import numpy as np
from PIL import Image

class CacheObject(object):

  def clear(self):
    for i in (x for x in dir(self) if x.startswith('_cache')):
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
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['flow_mask'].extract(self.namedict['flow_mask'],tfd)
        return np.loadtxt(p)
  
  @LazyProperty
  def flow_flow(self):
    with tempfile.TemporaryDirectory() as tfd:
        p = self.zips['flow_flow'].extract(self.namedict['flow_flow'],tfd)
        return np.loadtxt(p)



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
  def __init__(self,path):
    self.path = path
    self.zips = {}
    self.expected = ['seg_right.zip', 'flow_mask.zip', 'seg_left.zip', 'image_right.zip', 'flow_flow.zip', 'depth_left.zip', 'depth_right.zip', 'image_left.zip']
    self.pathsfound = [x for x in self.expected if glob(path + '/' + "*" + x) != []]
    self.zipfilepaths = [glob(path + '/' + "*" + x)[0] for x in self.pathsfound]
    self.scene_name = "_".join(self.zipfilepaths[0].split('/')[-1].split('_')[0:2])
    self.zipfiles = [zipfile.ZipFile(x) for x in self.zipfilepaths]
    self.zips = dict(zip([x.split('.')[0] for x in self.pathsfound],self.zipfiles))
    self.indices = np.unique([int(x.split('/')[3][1:]) for x in next(iter(self.zips.items()))[1].namelist()]).tolist()
    self.purepath = PurePath(next(iter(self.zips.items()))[1].namelist()[0]).parents[2]
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
    

