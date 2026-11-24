from dataclasses import dataclass, field

# 位置，速度-全局坐标系，角度-弧度
@dataclass
class Vector3D:
    x:float = 0
    y:float = 0
    z:float = 0
    def get_state(self):
        return {"x":self.x, "y":self.y,"z":self.z}

@dataclass
class Location:
    x:float = 0
    y:float = 0
    z:float = 0
    def get_state(self):
        return {"x":self.x, "y":self.y,"z":self.z}
@dataclass
class Rotation:
    pitch:float = 0  # radian
    yaw:float = 0       # radian
    roll:float = 0      # radian
    def get_state(self):
        return {"pitch":self.pitch, "yaw":self.yaw, "roll":self.roll}

@dataclass
class Transform:
    location: Location = field(default_factory=Location)
    rotation: Rotation = field(default_factory=Rotation)
    def get_state(self):
        return {"location":self.location.get_state(), "rotation":self.rotation.get_state()}
@dataclass
class BoundingBox:

    extent:Vector3D =   field(default_factory=Vector3D)
    def get_state(self):
        return {"extent":self.extent.get_state()}
@dataclass
class Actor:
    name:str = ''
    type:str = ''
    transform: Transform = field(default_factory=Transform)
    velocity: Vector3D = field(default_factory=Vector3D)
    bounding_box: BoundingBox = field(default_factory=BoundingBox)
    is_alive: bool = False
    timestamp: float = 0
    def get_state(self):
        return {"name":self.name, "type":self.type, "transform":self.transform.get_state(),
                "velocity":self.velocity.get_state(),
                "bounding_box":self.bounding_box.get_state(),
                "is_alive":self.is_alive, "timestamp":self.timestamp
                }




# a = Actor(is_alive = True)
# log.info(a.get_state())
# # log.info(a)

