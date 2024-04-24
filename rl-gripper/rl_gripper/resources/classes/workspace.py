class Workspace:
    def __init__(self):
        self.area = 0.3 #Seitenlänge der rechteckigen Arbeitsfläche (Mittelpunkt bei (0.4, 0.0))
        self.xMin = None
        self.xMax = None
        self.yMin = None
        self.yMax = None

    def define_workspace(self, cube_position, curriculum):
        if cube_position == 'FIX' or curriculum:
            self.xMin = 0.4
            self.xMax = 0.4
            self.yMin = 0.0
            self.yMax = 0.0
        elif cube_position == 'RANDOM':
            self.xMin = 0.4 - self.area/2
            self.xMax = 0.4 + self.area/2
            self.yMin = 0.0 - self.area/2
            self.yMax = 0.0 + self.area/2

