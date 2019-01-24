import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


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
        elif sensor_status%4 == 1:
            self.agent.move_forward()
        elif sensor_status == 3:
            self.agent.rotate_right()
        elif sensor_status ==7:
            self.turnoff()
class AgentProgram_reflex_mem:
    def __init__(self, agent):
        self.agent = agent
        self._on = 1
        self._actions=[self.agent.clean, self.agent.rotate_right, self.agent.rotate_left, self.agent.move_forward,
                       self.turnoff]
        self.memory = 0

    def turnoff(self):
        self._on = 0

    def on(self):
        return self._on == 1

    def proceed(self):
        if self.agent.sensor_clean() == 0:
            self.agent.clean()

        elif self.agent.sensor_clean() + self.agent.sensor_wall() + self.agent.sensor_home() ==3:
            self.turnoff()

        elif self.agent.sensor_wall() == 1:
            if self.memory == 0:
                self.agent.rotate_right()
                self.memory = 1
            elif self.memory ==2:
                self.agent.rotate_right()
                self.memory = 3
            elif self.memory == 3:
                self.agent.rotate_left()
                self.memory = 4
            elif self.memory == 4:
                self.agent.rotate_left()
                self.memory = 5
            elif self.memory ==5:
                self.agent.rotate_left()
                self.memory = 3

        elif self.agent.sensor_clean() ==1:
            if self.memory == 0:
                self.agent.move_forward()

            elif self.memory == 1:
                self.agent.move_forward()
                self.memory = 2
            elif self.memory == 2:
                self.agent.rotate_right()
                self.memory = 3
            elif self.memory == 3:
                self.agent.move_forward()
            elif self.memory == 4:
                self.agent.move_forward()
                self.memory = 5
            elif self.memory == 5:
                self.agent.rotate_left()
                self.memory = 0
class AgentProgram_stocastic:
    def __init__(self, agent):
        self.agent = agent
        self._on = 1
        self._actions=[self.agent.clean, self.agent.rotate_right, self.agent.rotate_left, self.agent.move_forward,
                       self.turnoff]
        self._action_probabilities = np.array([[1, 0, 0, 0, 0],  #dirty
                                [0, 0.138, 0.088, 1 - 0.138 - 0.088, 0],  #clean
                                [1, 0, 0, 0, 0],  # dirty, wall
                                [0, 1, 0, 0, 0],  #clean, wall
                                [1, 0, 0, 0, 0],  # dirty, home
                                [0, 0, 0, 1, 0],  # clean, home
                                [1, 0, 0, 0, 0],  # dirty, wall, home
                                [0, 1, 0, 0, 0.0]]) #clean, wall, home

    def turnoff(self):
        self._on = 0

    def on(self):
        return self._on == 1

    def proceed(self):
        sensor_status = int(self.agent.sensor_home()*2**2 + self.agent.sensor_wall()*2 + self.agent.sensor_clean())
        #0, not home, no wall, not clean
        #1, ', ', clean,
        #2 ', wall, dirty,
        #3 ', wall, clean
        r =  np.random.random_sample()
        self._actions[0]()
        for j in range(0,4):
            s = sum(self._action_probabilities[ sensor_status, 0:j+1])
            if r <= s:
                self._actions[j]()
                return

class AgentProgram_stocastic_learner:
    def __init__(self, agent, alpha=0.15, beta=0.15, gamma=0.5):
        self.agent = agent
        self._on = 1
        self._actions=[self.agent.clean, self.agent.rotate_right, self.agent.rotate_left, self.agent.move_forward,
                       self.turnoff]
        self._action_probabilities = np.array([[1, 0, 0, 0, 0],  #dirty
                                [0, alpha, beta, 1 - alpha - beta, 0],  #clean
                                [1, 0, 0, 0, 0],  # dirty, wall
                                [0, gamma, 1 - gamma, 0, 0],  #clean, wall
                                [1, 0, 0, 0, 0],  # dirty, home
                                [0, 0, 0, 1, 0],  # clean, home
                                [1, 0, 0, 0, 0],  # dirty, wall, home
                                [0, 0.75, 0, 0, 0.25]]) #clean, wall, home

    def turnoff(self):
        self._on = 0

    def on(self):
        return self._on == 1

    def proceed(self):
        sensor_status = int(self.agent.sensor_home()*2**2 + self.agent.sensor_wall()*2 + self.agent.sensor_clean())
        #0, not home, no wall, not clean
        #1, ', ', clean,
        #2 ', wall, dirty,
        #3 ', wall, clean
        r =  np.random.random_sample()
        self._actions[0]()
        for j in range(0,4):
            s = sum(self._action_probabilities[ sensor_status, 0:j+1])
            if r <= s:
                self._actions[j]()
                return

def stocastic_trainer():
    #define default parameters
    # alpha = 0.138, beta = 0.08, avg ~ 720
    best_parameters = np.array([0.1, 0.1, 0.5, 0])
    avg_steps = 0
    tr=500
    for test in range(0, tr):
        env_no_wall = Environment(0)
        robot_no_wall = Agent(env_no_wall, (0, 0), (0, 0), 0)
        AI_no_walls = AgentProgram_stocastic_learner(robot_no_wall, best_parameters[0], best_parameters[1], best_parameters[2])
        env_wall = Environment(1)
        robot_wall = Agent(env_wall, (0, 0), (0, 0), 0)
        AI_walls = AgentProgram_stocastic_learner(robot_wall, best_parameters[0], best_parameters[1], best_parameters[2])
        s = 0
        t = 0
        while np.sum(env_no_wall.floor) < 90:
            s += 1
            AI_no_walls.proceed()


        while np.sum(env_wall.floor) < 90:
            t += 1
            AI_walls.proceed()

        avg_steps += (s + t)
    best_parameters[3] = avg_steps/(2*tr)
    print(best_parameters)

    #vary parameters and tests
    for trial in range(0, 10):
        print(trial)
        best_outcome = best_parameters
        for run in range(0, 10):
            alpha = 0.05*(np.random.random_sample() - 0.5) + best_parameters[0]
            beta = 0.05*(np.random.random_sample() - 0.5) +  best_parameters[1]
            gamma= 0.05*(np.random.random_sample() -0.5) + best_parameters[2]
            avg_steps = 0
            for test in range(0, tr):
                env_no_wall = Environment(0)
                robot_no_wall = Agent(env_no_wall, (0, 0), (0, 0), 0)
                AI_no_walls = AgentProgram_stocastic_learner(robot_no_wall, alpha, beta, gamma)
                env_wall = Environment(1)
                robot_wall = Agent(env_wall, (0, 0), (0, 0), 0)
                AI_walls = AgentProgram_stocastic_learner(robot_wall, alpha, beta, gamma)
                s = 0
                t = 0
                while np.sum(env_no_wall.floor) < 90:
                    s += 1
                    AI_no_walls.proceed()

                while np.sum(env_wall.floor) < 90:
                    t += 1
                    AI_walls.proceed()

                avg_steps += (s + t)
            avg_steps = avg_steps/(2*tr)
            if avg_steps < best_outcome[3]:
                best_outcome[0] = alpha
                best_outcome[1] = beta
                best_outcome[2] = gamma
                best_outcome[3] = avg_steps
        best_parameters = best_outcome
        print(best_parameters)

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

def animate(env, AI, intv=50):
    fig, (ax, ax2) = plt.subplots(1 ,2)
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)

    ax.set_xlim([0, 10])
    ax.set_ylim([0,10])
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')

    ax2.set_xlim([0, 500])
    ax2.set_ylim([0,100])
    #plt.axis('equal')
    #plt.grid(True)

    #line
    linex = [AI.agent.position[0] + 0.5]
    liney = [AI.agent.position[1] + 0.5]
    line, = ax.plot(linex, liney, color='blue')

    #clean
    progy = [np.sum(env.floor)]
    progx = [0]
    prog, = ax2.plot(progx, progy, color='red')

    #walls
    for i in range(0, 10):
        for j in range(0, 10):
            if env.walls[i, j, 0] == 1:  # north
                ax.plot([i, i + 1], [j + 1, j + 1], 'k')
            if env.walls[i, j, 1] == 1:  # east
                ax.plot([i + 1, i + 1], [j, j + 1], 'k')
            if env.walls[i, j, 2] == 1:  # south
                ax.plot([i, i + 1], [j, j], 'k')
            if env.walls[i, j, 3] == 1:  # west
                ax.plot([i, i], [j, j + 1], 'k')


    #robot
    robot = []
    robot.append(plt.Circle((AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.5, ), 0.2, color='blue'))
    if AI.agent.orientation == 0:
        robot.append(plt.Circle((AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.6), 0.1, color='red'))
    if AI.agent.orientation == 1:
        robot.append(plt.Circle((AI.agent.position[0] + 0.6, AI.agent.position[1] + 0.5), 0.1, color='red'))
    if AI.agent.orientation == 2:
        robot.append(plt.Circle((AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.4), 0.1, color='red'))
    if AI.agent.orientation == 3:
        robot.append(plt.Circle((AI.agent.position[0] + 0.4, AI.agent.position[1] + 0.5), 0.1, color='red'))
    ax.add_patch(robot[0])
    ax.add_patch(robot[1])


    def update(i):
        AI.proceed()
        robot[0].center = (AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.5,)
        if AI.agent.orientation == 0:
            robot[1].center = (AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.6)
        if AI.agent.orientation == 1:
            robot[1].center = (AI.agent.position[0] + 0.6, AI.agent.position[1] + 0.5)
        if AI.agent.orientation == 2:
            robot[1].center = (AI.agent.position[0] + 0.5, AI.agent.position[1] + 0.4)
        if AI.agent.orientation == 3:
            robot[1].center = (AI.agent.position[0] + 0.4, AI.agent.position[1] + 0.5)
        linex.append(AI.agent.position[0] + 0.5)
        liney.append(AI.agent.position[1] + 0.5)
        line.set_xdata(linex)
        line.set_ydata(liney)
        progy.append(np.sum(env.floor))
        progx.append(len(progy) - 1)
        prog.set_xdata(progx)
        prog.set_ydata(progy)
        ax2.set_xlim([0, 100 + 100*np.floor(len(progy)/100)])
        return robot[0], robot[1], line, prog

    anim = animation.FuncAnimation(fig, update, interval=intv, blit=True)
    plt.show()
