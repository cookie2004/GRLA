import time
#import retro
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
#import keyboard
import gym
import matplotlib.pylab as plt
from IPython.display import clear_output
#retro.data.list_games()

#obs, rew, done, info = env.step(env.action_space.sample())
#(224, 240, 3)
#MultiBinary(9)                          

class ANN():
  def __init__(self, action_space):
    self.action_space = len(action_space)
    num_actions = self.action_space
    #NN layers
    image_input = layers.Input(shape=(84,84,4))
    #preprocessor = layers.experimental.preprocessing.Resizing(84, 84, interpolation='bilinear', name=None)(image_input)
    # Convolutions on the frames on the screen
    #data_format='channels_first'
    layer1 = layers.Conv2D(32, 8, strides=4, activation="relu")(image_input)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    #Define NN parameters.
    self.toymodel = keras.Model(inputs=image_input, outputs=action)
    self.loss_fn = tf.keras.losses.Huber()
    self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    self.toymodel.compile(self.optimizer, self.loss_fn)

  def trainStep(self, sample_X, sample_Y):
    with tf.GradientTape() as tape:
      old_q = self.toymodel(sample_X, training=True)
      loss_value = self.loss_fn(sample_Y, old_q)
    grads = tape.gradient(loss_value, self.toymodel.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.toymodel.trainable_weights))
    return loss_value.numpy()

  def train(self, x_input, y_input, batchsize=64):
    loss_history = []
    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))
    dataset = dataset.shuffle(buffer_size=1024).batch(batchsize)
    for steps, (x, y) in enumerate(dataset):
      loss_history.append(self.trainStep(x,y))
    return loss_history

  def forward(self, x_input):
    return self.toymodel(x_input)

class Atari_ANN():
  def __init__(self, action_space):
    self.action_space = len(action_space)
    num_actions = self.action_space
    #NN layers
    image_input = layers.Input(shape=(84,84,4))
    #preprocessor = layers.experimental.preprocessing.Resizing(84, 84, interpolation='bilinear', name=None)(image_input)
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 8, strides=8, activation="relu")(image_input)
    layer2 = layers.Conv2D(64, 4, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 3, strides=1, activation="relu")(layer2)
    layer4 = layers.Flatten()(layer3)
    layer5 = layers.Dense(512, activation="relu")(layer4)
    action = layers.Dense(num_actions, activation="linear")(layer5)

    #Define NN parameters.
    self.toymodel = keras.Model(inputs=image_input, outputs=action)
    self.loss_fn = tf.keras.losses.Huber()
    self.optimizer = keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    self.toymodel.compile(self.optimizer, self.loss_fn)

  def trainStep(self, sample_X, sample_Y):
    with tf.GradientTape() as tape:
      old_q = self.toymodel(sample_X, training=True)
      loss_value = self.loss_fn(sample_Y, old_q)
    grads = tape.gradient(loss_value, self.toymodel.trainable_weights)
    self.optimizer.apply_gradients(zip(grads, self.toymodel.trainable_weights))
    return loss_value.numpy()

  def train(self, x_input, y_input, batchsize=64):
    loss_history = []
    dataset = tf.data.Dataset.from_tensor_slices((x_input, y_input))
    dataset = dataset.shuffle(buffer_size=1024).batch(batchsize)
    for steps, (x, y) in enumerate(dataset):
      loss_history.append(self.trainStep(x,y))
    return loss_history

  def forward(self, x_input):
    return self.toymodel(x_input)

def detect_input():
    #["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
    output = [0,0,0,0,0,0,0,0,0]
    print ('Waiting for human input...')
    while True:
        if keyboard.is_pressed('/'):
            if keyboard.is_pressed('w'): output[4] = 1
            if keyboard.is_pressed('s'): output[5] = 1
            if keyboard.is_pressed('a'): output[6] = 1
            if keyboard.is_pressed('d'): output[7] = 1
            if keyboard.is_pressed('v'): output[8] = 1
            if keyboard.is_pressed('b'): output[0] = 1
            break
        else:
            None
    print (output)
    time.sleep(0.1)
    return output

def load_history(action_history, state_history, next_state_history, reward_history, done_history, return_history):
    action_history = np.load(runname + 'action_history' + '.npy')
    state_history = np.load(runname + 'state_history' + '.npy')
    #next_state_history = np.load(runname + 'next_state_history' + '.npy')
    #reward_history = np.load(runname + 'reward_history' + '.npy')
    #done_history = np.load(runname + 'done_history' + '.npy')
    #return_history = np.load(runname + 'return_history' + '.npy') 


#behavior.toymodel = keras.models.load_model('120227_SMB')


class Agent_4Frame():
    def __init__(self,runname, env, action_space):
        self.action_space = action_space
        self.env = env
        self.steps_taken = 0 
        self.runname = runname
        self.epsilon = 0
        self.epsilon_max = 1.0
        self.epsilon_min = 0.1
        self.annealing_time = 1000000
        self.len_of_episode = 10000
        self.gamma = 0.99
        self.max_memory_len = 1000000
        self.batch_size = 32
        self.loss_history = []
        self.action_history = []
        self.state_history= []
        self.next_state_history = []
        self.reward_history = []
        self.done_history = []
        self.episodic_return = []
        self.return_history = [] 
        self.behavior = ANN(self.action_space)
        self.target = ANN(self.action_space)
    
        #["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]


    def preprocess(self, image):
        output = np.average(np.array(image), axis=2)[25:205]
        #return output
        return cv2.resize(output, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)/255.0

    def popback(self, state_block, incoming_state):
        state_block.pop(0)
        state_block.append(incoming_state)
        return state_block

    def gradient_update(self, 
                        runname,
                        state_history, 
                        next_state_history,
                        rewards_history,
                        action_history,
                        loss_history,
                        model,
                        target_model,
                        gamma,
                        batch_size,
                        done_history,
                        action_space):
    
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            next_state_sample = np.array([next_state_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices])
            future_rewards = target_model.toymodel.predict(next_state_sample)
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
      
            updated_q_values = updated_q_values *(1-done_sample) - done_sample  
                         
            masks = tf.one_hot(action_sample, len(action_space))
            with tf.GradientTape() as tape:  
                q_values = model.toymodel(state_sample)
                q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = model.loss_fn(updated_q_values, q_actions)
                
            loss_history.append(loss)
            grads = tape.gradient(loss, model.toymodel.trainable_variables)
            model.toymodel.optimizer.apply_gradients(zip(grads, model.toymodel.trainable_variables))

                        
    def save_history(self,
                     runname,
                     action_history,
                     state_history,
                     next_state_history,
                     reward_history,
                     done_history,
                     return_history):            
        np.save(runname + 'action_history',action_history)
        np.save(runname + 'state_history', state_history)
        np.save(runname + 'next_state_history', next_state_history)
        np.save(runname + 'reward_history', reward_history)
        np.save(runname + 'done_history', done_history)
        np.save(runname + 'return_history', return_history)   
        
    def memory_manager(self,array, mem_size):
        num_delete = len(array) - mem_size
        if num_delete < 0:
            None
        else:
            del array[:num_delete]
                
    def episode(self,num_episodes):    #Double Deep Q
        for i in range (num_episodes):
            self.epsilon = self.epsilon_max - (((self.epsilon_max-self.epsilon_min)/self.annealing_time)*self.steps_taken)
            print ('Epsilon is at ', np.max([self.epsilon, self.epsilon_min]), ' as of step ', self.steps_taken)
            epi_return = 0 
            done = False
            lives = self.env.unwrapped.ale.lives()
            
            s = []
            s.append(self.preprocess(self.env.reset()))
            #Prime the state s with 3 frames.
            
            for i in range(3):
                frame, reward, done, info = self.env.step(self.action_space[1])
                epi_return += reward
                s.append(self.preprocess(frame))

            s_channeled = np.dstack((s[0],s[1], s[2], s[3]))
            #Choose an initial action.
            if np.random.random() < np.max([self.epsilon,self.epsilon_min]):
                a = np.random.choice(np.arange(len(self.action_space)))
            else: 
                a_probs = self.behavior.toymodel(np.expand_dims(s_channeled,0), training=False)
                a = tf.argmax(a_probs[0]).numpy()
        
            while not done:
                new_frame, reward, done, info = self.env.step(self.action_space[a])
                s_prime = self.popback(s, self.preprocess(new_frame))      
                s_prime_channeled = np.dstack((s_prime[0],s_prime[1], s_prime[2], s_prime[3]))
                epi_return += reward
                self.env.render()
                
                #Check if a life was lost. Is so, make the reward -1. 
                if not (lives == self.env.unwrapped.ale.lives()):
                    reward = -1
                    lives = self.env.unwrapped.ale.lives()
                    
                #Check is episode is done. Assign the reward for reaching the terminal state as -1. 
                if done:
                    self.done_history[-1] = True
                    self.return_history.append(epi_return)
                    break
                if np.random.random() < np.max([self.epsilon,self.epsilon_min]):
                    a_prime = np.random.choice(np.arange(len(self.action_space)))
                else:
                    a_probs = self.behavior.toymodel(np.expand_dims(s_prime_channeled,0), training=False)
                    a_prime = tf.argmax(a_probs[0]).numpy()
                    #print ('Nonrandom action taken. ', a_prime)
                #Save to history
                self.reward_history.append(reward)
                self.state_history.append(s_channeled)
                self.action_history.append(a)
                self.next_state_history.append(s_prime_channeled)
                self.done_history.append(done)
                 
                if len(self.reward_history)>32 and self.steps_taken%4==0:
                    self.gradient_update(self.runname,
                                         self.state_history, 
                                         self.next_state_history,
                                         self.reward_history,
                                         self.action_history,
                                         self.loss_history,
                                         self.behavior, 
                                         self.target,
                                         self.gamma,
                                         self.batch_size,
                                         self.done_history,
                                         self.action_space)
                if self.steps_taken%10000==0:
                    self.target.toymodel.set_weights(self.behavior.toymodel.get_weights())
                    print ('Target model has been updated.')
                s = s_prime
                a = a_prime
                self.steps_taken += 1
                self.memory_manager(self.action_history, self.max_memory_len)
                self.memory_manager(self.state_history, self.max_memory_len)
                self.memory_manager(self.next_state_history, self.max_memory_len)
                self.memory_manager(self.reward_history, self.max_memory_len)
                self.memory_manager(self.done_history, self.max_memory_len)
            #self.save_history(self.runname,
            #                  self.action_history,
            #                  self.state_history,
            #                  self.next_state_history,
            #                  self.reward_history,
            #                  self.done_history,
            #                  self.return_history)
            self.episodic_return.append(epi_return)
            #print ("Episode complete.")
            #self.env.close()
            #clear_output()
            #plt.figure(figsize=(5,5))
            #plt.plot(np.arange(len(self.episodic_return)), self.episodic_return)
            #plt.xlabel('Episode')
            #plt.ylabel('Return')
            #plt.show()
            #plt.figure(figsize=(5,5))
            #plt.plot(np.arange(len(self.loss_history)), self.loss_history)
            #plt.xlabel('Update steps')
            #plt.ylabel('Huber Loss')
            #plt.yscale('log')
            #plt.show()
            print ('Episode', len(self.episodic_return))
            print ('Huber loss', self.loss_history[-1])
            print ('Episodic Reward', self.episodic_return[-1])
        self.behavior.toymodel.save('120301_Breakout')

env = retro.make(game='SuperMarioBros-Nes')

SMB_action_space = [[0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,1,0,1],
                [0,0,0,0,0,0,0,1,1]]

env1 = gym.make('BreakoutNoFrameskip-v4')
atari_action_space = np.arange(4)

#agent = Agent_4Frame('210303_1', env1, atari_action_space)
agent = Agent_4Frame('210309_1', env, SMB_action_space)
agent.episode(10000000)
