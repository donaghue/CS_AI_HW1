import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, mode=0):
        self.floor = np.zeros((10, 10))
        self.walls= np.zeros((10, 10, 4)) # 0 = north, 1 = east, 2= south, 3 = west
        self.walls[:, 9, 0] = np.ones(10) # north
        self.walls[9, :, 1] = np.ones(10) # east
        self.walls[:, 0, 2] = np.ones(10) # south
        self.walls[0, :, 3] = np.ones(10) # west
        if mode==1:
            #north
            self.walls[0:2, 4, 0] = np.ones(2)
            self.walls[3:7, 4, 0] = np.ones(4)
            self.walls[8:10, 4, 0] = np.ones(2)
            #east
            self.walls[4, 0:2, 1] = np.ones(2)
            self.walls[4, 3:7, 1] = np.ones(4)
            self.walls[4, 8:10, 1] = np.ones(2)
            #south
            self.walls[0:2, 5, 2] = np.ones(2)
            self.walls[3:7, 5, 2] = np.ones(4)
            self.walls[8:10, 5, 2] = np.ones(2)
            # west
            self.walls[5, 0:2, 3] = np.ones(2)
            self.walls[5, 3:7, 3] = np.ones(4)
            self.walls[5, 8:10, 3] = np.ones(2)

class Agent:
    def __init__(self, env, home, position, orientation):
        self._environment=env
        self._home= home
        self.position = home
        self.orientation = orientation  # 0 = north, 1 = east, 2= south, 3 = west

    def sensor_clean(self):
        return self._environment.floor[self.position[0], self.position[1]] #1 = clean, 0 = dirty

    def sensor_wall(self):
        return self._environment.walls[self.position[0], self.position[1], self.orientation] #1 = wall, 0 no wall

    def sensor_home(self):
        if self.position == self._home:
            return 1 #home
        return 0 #not home


    def clean(self):
        self._environment.floor[self.position] = 1

    def rotate_right(self):
        self.orientation = (self.orientation + 1)%4

    def rotate_left(self):
        self.orientation = (self.orientation - 1)%4

    def move_forward(self):
        if self.sensor_wall() == 0:
            if self.orientation == 0: #north
                self.position = (self.position[0], self.position[1] + 1)
            elif self.orientation == 1: #east
                self.position = (self.position[0] + 1, self.position[1])
            elif self.orientation ==2: #south
                self.position = (self.position[0], self.position[1] - 1)
            elif self.orientation == 3: #west
                self.position = (self.position[0] -1, self.position[1])
        else:
            raise Exception('Wall!')

class AgentProgram_reflex:
    def __init__(self, agent):
        self.agent = agent
        self._on = 1

    def turnoff(self):
        self._on = 0

    def on(self):
        return self._on == 1

    def proceed(self):
        sensor_status = self.agent.sensor_home()*2**2 + self.agent.sensor_wall()*2 + self.agent.sensor_clean()
        #0, not home, no wall, not clean
        #1, ', ', clean,
        #2 ', wall, dirty,
        #3 ', wall, clean
        if sensor_status%2 == 0:
            self.agent.clean()
            print('cleaning')
        elif sensor_status%4 == 1:
            self.agent.move_forward()
            print('moving')
        elif sensor_status == 3:
            self.agent.rotate_right()
            print('right')
        elif sensor_status ==7:
            self.turnoff()
            print('off')

class AgentProgram_stocastic:
    def __init__(self, agent):
        self.agent = agent
        self._on = 1

        # a_0 = prob clean
        # a_1 = prob right
        # a_2 = prob left
        # a_3 = prob forward
        # a_4 = prob off

        status_0_prob = ()

    def turnoff(self):
        self._on = 0

    def on(self):
        return self._on == 1

    def proceed(self):
        sensor_status = self.agent.sensor_home()*2**2 + self.agent.sensor_wall()*2 + self.agent.sensor_clean()
        #0, not home, no wall, not clean
        #1, ', ', clean,
        #2 ', wall, dirty,
        #3 ', wall, clean
        if sensor_status%2 == 0:
            self.agent.clean()
            print('cleaning')
        elif sensor_status%4 == 1:
            self.agent.move_forward()
            print('moving')
        elif sensor_status == 3:
            self.agent.rotate_right()
            print('right')
        elif sensor_status ==7:
            self.turnoff()
            print('off')


def render(env, agent, history= []):
    plt.axis('off')
    ax = plt.gca()
    plt.axis('equal')
    #plt.grid(True)
    #draw tiles
    for i in range(0, 10):
        for j in range(0, 10):
            # walls
            if env.walls[i, j, 0] == 1: # north
                plt.plot([i, i+1], [j+1, j+1], 'k')
            if env.walls[i, j, 1] == 1: # east
                plt.plot([i+1, i+1], [j, j+1], 'k')
            if env.walls[i, j, 2] == 1: # south
                plt.plot([i, i+1], [j, j], 'k')
            if env.walls[i, j, 3] == 1: # west
                plt.plot([i, i], [j, j+1], 'k')
            # clean
            if env.floor[i,j] ==0:
                ax.add_artist(plt.Rectangle((i, j), 1, 1, color=(0.9, 0.9, 0.9)))
    #draw agent
    ax.add_artist(plt.Circle((agent.position[0] + 0.5, agent.position[1] + 0.5), 0.2, color='blue'))
    if agent.orientation == 0:
        ax.add_artist(plt.Circle((agent.position[0] + 0.5, agent.position[1] + 0.6), 0.1, color='red'))
    if agent.orientation == 1:
        ax.add_artist(plt.Circle((agent.position[0] + 0.6, agent.position[1] + 0.5), 0.1, color='red'))
    if agent.orientation == 2:
        ax.add_artist(plt.Circle((agent.position[0] + 0.5, agent.position[1] + 0.4), 0.1, color='red'))
    if agent.orientation == 3:
        ax.add_artist(plt.Circle((agent.position[0] + 0.4, agent.position[1] + 0.5), 0.1, color='red'))
    plt.show()

env = Environment(0)
robot = Agent(env, (0,0), (0,0), 0)
AI = AgentProgram_reflex(robot)
history_x = []
history_y = []
while AI.on():
    AI.proceed()

render(env, robot)