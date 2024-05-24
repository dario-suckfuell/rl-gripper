from PIL import Image
import pybullet as p
import random
import yaml


def render_rgba_flat(width, height, rgba_flat_array):
    image = Image.new("RGBA", (width, height))

    rgba_array = [(rgba_flat_array[i], rgba_flat_array[i + 1], rgba_flat_array[i + 2], rgba_flat_array[i + 3])
                  for i in range(0, len(rgba_flat_array), 4)]

    image.putdata(rgba_array)
    image.show()


def spawn_random_cubes(count):
    for i in range(count):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        cubeID = p.loadURDF("model/cube.urdf", [x, y, .8], p.getQuaternionFromEuler([0, 0, 0]))


def load_config(path='rl_gripper/config/config.yaml'):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    return config
