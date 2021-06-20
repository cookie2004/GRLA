import time
import retro
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
#retro.data.list_games()

#obs, rew, done, info = env.step(env.action_space.sample())
#(224, 240, 3)
#MultiBinary(9)
#["B", null, "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
action_space = [[0,0,0,0,0,0,0,0,0],
                [1,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,1,0,0],
                [0,0,0,0,0,0,0,1,0],
                [0,0,0,0,0,0,0,0,1],
                [1,0,0,0,0,0,1,0,0],
                [1,0,0,0,0,0,0,1,0],
                [1,0,0,0,0,0,0,0,1],
                [0,0,0,0,0,0,1,1,0],
                [0,0,0,0,0,0,1,0,1],
                [0,0,0,0,0,0,0,1,1],
                [1,0,0,0,0,0,0,1,1],
                [1,0,0,0,0,0,1,0,1],
                [1,0,0,0,0,0,1,1,1],
                [0,0,0,0,0,1,0,0,0],
                [1,0,0,0,0,1,0,0,0],
                [0,0,0,0,0,1,0,0,1],
                [0,0,0,0,0,0,1,1,0]]

class ANN():
  def __init__(self):
    num_actions = 18
    #NN layers
    image_input = layers.Input(shape=(4,84,84,))
    #preprocessor = layers.experimental.preprocessing.Resizing(84, 84, interpolation='bilinear', name=None)(image_input)
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(32, 4, strides=4, activation="relu")(image_input)
    layer2 = layers.Conv2D(64, 1, strides=2, activation="relu")(layer1)
    layer3 = layers.Conv2D(64, 1, strides=1, activation="relu")(layer2)
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

def preprocess(image):
    output = np.average(np.array(image), axis=2)[25:205]
    #return output
    return cv2.resize(output, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)

def popback(state_block, incoming_state):
    state_block.pop(0)
    state_block.append(incoming_state)
    return state_block

def gradient_update(state_history, 
                    next_state_history,
                    rewards_history,
                    action_history,
                    loss_history,
                    model, 
                    gamma,
                    batch_size):
    
            # Get indices of samples for replay buffers
            indices = np.random.choice(range(len(done_history)), size=batch_size)
            # Using list comprehension to sample from replay buffer
            state_sample = np.array([state_history[i] for i in indices])
            next_state_sample = np.array([next_state_history[i] for i in indices])
            rewards_sample = [rewards_history[i] for i in indices]
            action_sample = [action_history[i] for i in indices]
            done_sample = tf.convert_to_tensor(
                [float(done_history[i]) for i in indices])
            print ('Memory contains ', len(action_history), 'states.')
            future_rewards = model.toymodel.predict((np.array(next_state_sample)))
            updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
            masks = tf.one_hot(action_sample, 18)
            with tf.GradientTape() as tape:  
                q_values = model.forward((np.array(state_sample)))
                q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                loss = model.loss_fn(updated_q_values, q_actions)
            loss_history.append(loss)
            grads = tape.gradient(loss, model.toymodel.trainable_variables)
            print ('Gradient updated. Loss at',float(loss))
            model.toymodel.optimizer.apply_gradients(zip(grads, model.toymodel.trainable_variables))
            model.toymodel.save('120223_SMB')
                                         
   
behavior = ANN()
#behavior.toymodel = keras.models.load_model('120223_SMB')
env = retro.make(game='SuperMarioBros-Nes')

epsilon = 0.9
epsilon_min = 0.05
gamma = 0.99
max_memory_len = 100000
batch_size = 32

loss_history = []
action_history = []
state_history= []
next_state_history = []
reward_history = []
done_history = []
episodic_return = []

step_counter = 0

for episodes in range(1000):
    s = []
    s.append(preprocess(env.reset()))
    #Prime the state s with 3 frames.
    prelim_reward = 0
    epi_return = 0 
    for i in range(3):
        frame, reward, done, info = env.step(action_space[0])
        prelim_reward += reward
        s.append(preprocess(frame))
    done = False
    #Choose an initial action.
    if np.random.random() < np.max([epsilon,epsilon_min]):
        a = np.random.choice(np.arange(len(action_space)))
    else:
        a = np.argmax(behavior.forward(np.expand_dims(s,0)))
    while not done:
        #env.render()
        new_frame, reward, done, info = env.step(action_space[a])
        s_prime = popback(s, preprocess(new_frame))
        epi_return += reward
        #env.render()
        if done:
            #Add gradient minimization step here.
            return_history.append(epi_return+prelim_reward)
            break
        if np.random.random() < np.max([epsilon,epsilon_min]):
            a_prime = np.random.choice(np.arange(len(action_space)))
        else:
            a_prime = np.argmax(behavior.forward(np.expand_dims(s,0)))
        #Save to history
        reward_history.append(prelim_reward)
        state_history.append(s)
        action_history.append(a)
        next_state_history.append(s_prime)
        done_history.append(done)
        if len(reward_history)>32 and step_counter > 10:
            gradient_update(state_history, 
                            next_state_history,
                            reward_history,
                            action_history,
                            loss_history,
                            behavior, 
                            gamma,
                            batch_size)
            step_counter = 0
        else:
            step_counter += 1
        s = s_prime
        a = a_prime
        epsilon -= 0.00009

        if len(reward_history)>1000000:
            action_history.pop(0)
            state_history.pop(0)
            next_state_history.pop(0)
            reward_history.pop(0)
            done_history.pop(0)
            
