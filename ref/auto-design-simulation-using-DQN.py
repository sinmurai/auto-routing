# original: https://github.com/shibuiwilliam/maze_solver

# Environment

import random
import copy


class Maze(object):
    def __init__(self, size=10, blocks_rate=0.1):
        self.size = size if size > 3 else 10
        self.blocks = int((size**2) * blocks_rate)
        self.s_list = []
        self.maze_list = []
        self.e_list = []

    def generate_maze(self):
        s_r = random.randint(1, (self.size / 2) - 1)
        for i in range(0, self.size):
            if i == s_r: self.s_list.extend("S")
            else: self.s_list.extend("#")
        start_point = [0, s_r]

        e_r = random.randint((self.size / 2) + 1, self.size - 2)
        for j in range(0, self.size):
            if j == e_r: self.e_list.extend([50])
            else: self.e_list.extend("#")
        goal_point = [self.size - 1, e_r]

        for k in range(0, self.size):
            self.__create_mid_lines(k)

        for k in range(self.blocks):
            self.__insert_blocks(k, s_r, e_r)

        return self.maze_list, start_point, goal_point

    def __create_mid_lines(self, k):
        if k == 0: self.maze_list.append(self.s_list)
        elif k == self.size - 1: self.maze_list.append(self.e_list)
        else:
            tmp_list = []
            for l in range(0, self.size):
                if l == 0: tmp_list.extend("#")
                elif l == self.size - 1: tmp_list.extend("#")
                else:
                    a = random.randint(-1, 0)
                    tmp_list.extend([a])
            self.maze_list.append(tmp_list)

    def __insert_blocks(self, k, s_r, e_r):
        b_y = random.randint(1, self.size - 2)
        b_x = random.randint(1, self.size - 2)
        if [b_y, b_x] == [1, s_r] or [b_y, b_x] == [self.size - 2, e_r]:
            k = k - 1
        else:
            self.maze_list[b_y][b_x] = "#"


class Field(object):
    def __init__(self, maze, start_point, goal_point):
        self.maze = maze
        self.start_point = start_point
        self.goal_point = goal_point
        self.__movable_vec = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    def get_actions(self, at):
        movables = []
        if at == self.start_point:
            y = at[0] + 1
            x = at[1]
            a = [[y, x]]
            return a
        else:
            for v in self.__movable_vec:
                y = at[0] + v[0]
                x = at[1] + v[1]
                if not (0 < x < len(self.maze) and 0 <= y <= len(self.maze) - 1
                        and self.maze[y][x] != "#" and self.maze[y][x] != "S"):
                    continue
                movables.append([y, x])
            if len(movables) != 0:
                return movables
            else:
                return None

    # returns (reward, done)
    def get_value(self, action):
        y, x = action
        if action == self.start_point:
            return 0, False
        else:
            v = float(self.maze[y][x])
            if action == self.goal_point:
                return v, True
            else:
                return v, False


# Solver

import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
import numpy as np

LEARNING_RATE = 0.0001
GAMMA = 0.9
E_DECAY = 0.9999
E_MIN = 0.01


def build_maze_solver():
    model = Sequential()
    model.add(Dense(128, input_shape=(2, 2), activation='tanh'))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss="mse", optimizer=RMSprop(lr=LEARNING_RATE))
    return model


class Solver:
    def __init__(self):
        self.memory = deque(maxlen=100000)
        self.epsilon = 1.0
        self.model = build_maze_solver()

    def remember_memory(self, state, action, reward, next_state, next_movables,
                        done):
        self.memory.append((state, action, reward, next_state, next_movables,
                            done))

    def choose_action(self, at, movables):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(at, movables)

    def choose_best_action(self, at, movables):
        best_actions = []
        max_act_value = -100
        for a in movables:
            np_action = np.array([[at, a]])
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [
                    a,
                ]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def replay_experience(self, batch_size, dump_and_stop=False):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        x = []
        y = []
        for i in range(batch_size):
            state, action, reward, next_state, next_movables, done = minibatch[
                i]
            input_action = [state, action]
            if done:
                target_f = reward
            else:
                next_rewards = []
                for a in next_movables:
                    np_next_s_a = np.array([[next_state, a]])
                    next_rewards.append(self.model.predict(np_next_s_a))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + GAMMA * np_n_r_max
            x.append(input_action)
            y.append(target_f)

        if (dump_and_stop):
            print("dump state")
            print(x, y)
            raise 'stop'

        self.model.fit(np.array(x), np.array([y]).T, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > E_MIN:
            self.epsilon *= E_DECAY


# --- RUN ---

# setup env

MAZE_SIZE = 10
MAZE_BARRIAR_RATE = 0.1


def build_field():
    maze_1 = Maze(MAZE_SIZE, MAZE_BARRIAR_RATE)
    maze, start_point, goal_point = maze_1.generate_maze()
    field = Field(maze, start_point, goal_point)
    return field


def display_field(field, point=None):
    field_data = copy.deepcopy(field.maze)
    if not point is None:
        y, x = point
        field_data[y][x] = "@@"
    else:
        point = ""
    for line in field_data:
        print("\t" + "%3s " * len(line) % tuple(line))


field = build_field()
display_field(field)

# train

EPISODE_COUNT = 10000
MAX_WALK_COUNT = 1000

solver = Solver()

for e in range(EPISODE_COUNT):
    at = field.start_point
    score = 0
    for time in range(MAX_WALK_COUNT):
        movables = field.get_actions(at)
        action = solver.choose_action(at, movables)
        reward, done = field.get_value(action)
        score = score + reward
        next_state = action
        next_movables = field.get_actions(next_state)
        solver.remember_memory(at, action, reward, next_state, next_movables,
                               done)
        if done or time == (MAX_WALK_COUNT - 1):
            if e % 500 == 0:
                print("episode: {}/{}, score: {}, e: {:.2} \t @ {}".format(
                    e, EPISODE_COUNT, score, solver.epsilon, time))
            break
        at = next_state

    solver.replay_experience(32)

# exec trained model

at = field.start_point
score = 0
steps = 0
while True:
    steps += 1
    movables = field.get_actions(at)
    action = solver.choose_best_action(at, movables)
    print("current state: {0} -> action: {1} ".format(at, action))
    reward, done = field.get_value(action)
    display_field(field, at)
    score = score + reward
    at = action
    print("current step: {0} \t score: {1}\n".format(steps, score))
    if done:
        display_field(field, action)
        print("goal!")
        break