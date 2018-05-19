
# Project: Train a Quadcopter How to Fly

Design an agent to fly a quadcopter, and then train it using a reinforcement learning algorithm of your choice! 

Try to apply the techniques you have learnt, but also feel free to come up with innovative ideas and test them.

## Instructions

Take a look at the files in the directory to better understand the structure of the project. 

- `task.py`: Define your task (environment) in this file.
- `agents/`: Folder containing reinforcement learning agents.
    - `policy_search.py`: A sample agent has been provided here.
    - `agent.py`: Develop your agent here.
- `physics_sim.py`: This file contains the simulator for the quadcopter.  **DO NOT MODIFY THIS FILE**.

For this project, you will define your own task in `task.py`.  Although we have provided a example task to get you started, you are encouraged to change it.  Later in this notebook, you will learn more about how to amend this file.

You will also design a reinforcement learning agent in `agent.py` to complete your chosen task.  

You are welcome to create any additional files to help you to organize your code.  For instance, you may find it useful to define a `model.py` file defining any needed neural network architectures.

## Controlling the Quadcopter

We provide a sample agent in the code cell below to show you how to use the sim to control the quadcopter.  This agent is even simpler than the sample agent that you'll examine (in `agents/policy_search.py`) later in this notebook!

The agent controls the quadcopter by setting the revolutions per second on each of its four rotors.  The provided agent in the `Basic_Agent` class below always selects a random action for each of the four rotors.  These four speeds are returned by the `act` method as a list of four floating-point numbers.  

For this project, the agent that you will implement in `agents/agent.py` will have a far more intelligent method for selecting actions!


```python
import random

class Basic_Agent():
    def __init__(self, task):
        self.task = task
    
    def act(self):
        new_thrust = random.gauss(450., 25.)
        return [new_thrust + random.gauss(0., 1.) for x in range(4)]
```

Run the code cell below to have the agent select actions to control the quadcopter.  

Feel free to change the provided values of `runtime`, `init_pose`, `init_velocities`, and `init_angle_velocities` below to change the starting conditions of the quadcopter.

The `labels` list below annotates statistics that are saved while running the simulation.  All of this information is saved in a text file `data.txt` and stored in the dictionary `results`.  


```python
%load_ext autoreload
%autoreload 2

import csv
import numpy as np
from task import Task

# Modify the values below to give the quadcopter a different starting position.
runtime = 50.                                     # time limit of the episode
init_pose = np.array([0., 0., 20., 0., 0., 0.])  # initial pose
init_velocities = np.array([0., 0., 0.])         # initial velocities
init_angle_velocities = np.array([0., 0., 0.])   # initial angle velocities
file_output = 'data.txt'                         # file name for saved results

# Setup
task = Task(init_pose, init_velocities, init_angle_velocities, runtime)
agent = Basic_Agent(task)
done = False
labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
results = {x : [] for x in labels}

# Run the simulation, and save the results.
with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    while True:
        rotor_speeds = agent.act()
        _, _, done = task.step(rotor_speeds)
        to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
        for ii in range(len(labels)):
            results[labels[ii]].append(to_write[ii])
        writer.writerow(to_write)
        if done:
            break
```

Run the code cell below to visualize how the position of the quadcopter evolved during the simulation.


```python
import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(results['time'], results['x'], label='x')
plt.plot(results['time'], results['y'], label='y')
plt.plot(results['time'], results['z'], label='z')
plt.legend()
_ = plt.ylim()
```


![png](output_6_0.png)


The next code cell visualizes the velocity of the quadcopter.


```python
plt.plot(results['time'], results['x_velocity'], label='x_hat')
plt.plot(results['time'], results['y_velocity'], label='y_hat')
plt.plot(results['time'], results['z_velocity'], label='z_hat')
plt.legend()
_ = plt.ylim()
```


![png](output_8_0.png)


Next, you can plot the Euler angles (the rotation of the quadcopter over the $x$-, $y$-, and $z$-axes),


```python
plt.plot(results['time'], results['phi'], label='phi')
plt.plot(results['time'], results['theta'], label='theta')
plt.plot(results['time'], results['psi'], label='psi')
plt.legend()
_ = plt.ylim()
```


![png](output_10_0.png)


before plotting the velocities (in radians per second) corresponding to each of the Euler angles.


```python
plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
plt.legend()
_ = plt.ylim()
```


![png](output_12_0.png)


Finally, you can use the code cell below to print the agent's choice of actions.  


```python
plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 revolutions / second')
plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 revolutions / second')
plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 revolutions / second')
plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 revolutions / second')
plt.legend()
_ = plt.ylim()
```


![png](output_14_0.png)


When specifying a task, you will derive the environment state from the simulator.  Run the code cell below to print the values of the following variables at the end of the simulation:
- `task.sim.pose` (the position of the quadcopter in ($x,y,z$) dimensions and the Euler angles),
- `task.sim.v` (the velocity of the quadcopter in ($x,y,z$) dimensions), and
- `task.sim.angular_v` (radians/second for each of the three Euler angles).


```python
# the pose, velocity, and angular velocity of the quadcopter at the end of the episode
print(task.sim.pose)
print(task.sim.v)
print(task.sim.angular_v)
```

    [-106.39033885   15.48406938    0.            0.11467936    0.62164775
        0.        ]
    [-114.42905099   -2.26720888  -70.31642297]
    [-0.00227616 -0.06922731  0.        ]
    

In the sample task in `task.py`, we use the 6-dimensional pose of the quadcopter to construct the state of the environment at each timestep.  However, when amending the task for your purposes, you are welcome to expand the size of the state vector by including the velocity information.  You can use any combination of the pose, velocity, and angular velocity - feel free to tinker here, and construct the state to suit your task.

## The Task

A sample task has been provided for you in `task.py`.  Open this file in a new window now. 

The `__init__()` method is used to initialize several variables that are needed to specify the task.  
- The simulator is initialized as an instance of the `PhysicsSim` class (from `physics_sim.py`).  
- Inspired by the methodology in the original DDPG paper, we make use of action repeats.  For each timestep of the agent, we step the simulation `action_repeats` timesteps.  If you are not familiar with action repeats, please read the **Results** section in [the DDPG paper](https://arxiv.org/abs/1509.02971).
- We set the number of elements in the state vector.  For the sample task, we only work with the 6-dimensional pose information.  To set the size of the state (`state_size`), we must take action repeats into account.  
- The environment will always have a 4-dimensional action space, with one entry for each rotor (`action_size=4`). You can set the minimum (`action_low`) and maximum (`action_high`) values of each entry here.
- The sample task in this provided file is for the agent to reach a target position.  We specify that target position as a variable.

The `reset()` method resets the simulator.  The agent should call this method every time the episode ends.  You can see an example of this in the code cell below.

The `step()` method is perhaps the most important.  It accepts the agent's choice of action `rotor_speeds`, which is used to prepare the next state to pass on to the agent.  Then, the reward is computed from `get_reward()`.  The episode is considered done if the time limit has been exceeded, or the quadcopter has travelled outside of the bounds of the simulation.

In the next section, you will learn how to test the performance of an agent on this task.

## The Agent

The sample agent given in `agents/policy_search.py` uses a very simplistic linear policy to directly compute the action vector as a dot product of the state vector and a matrix of weights. Then, it randomly perturbs the parameters by adding some Gaussian noise, to produce a different policy. Based on the average reward obtained in each episode (`score`), it keeps track of the best set of parameters found so far, how the score is changing, and accordingly tweaks a scaling factor to widen or tighten the noise.

Run the code cell below to see how the agent performs on the sample task.


```python
import sys
import pandas as pd
from agents.policy_search import PolicySearch_Agent
from task import Task

num_episodes = 200
target_pos = np.array([0., 0., 40.])
task = Task(target_pos=target_pos)
agent = PolicySearch_Agent(task) 

for i_episode in range(1, num_episodes+1):
    state = agent.reset_episode() # start a new episode
    while True:
        action = agent.act(state)
        next_state, reward, done = task.step(action)
        agent.step(reward, done)
        state = next_state
        if done:
            print("\rEpisode = {:4d}, score = {:7.3f} (best = {:7.3f}), noise_scale = {}".format(
                i_episode, agent.score, agent.best_score, agent.noise_scale), end="")  # [debug]
            break
    sys.stdout.flush()
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-9-df7b82e5c05b> in <module>()
         13     while True:
         14         action = agent.act(state)
    ---> 15         next_state, reward, done = task.step(action)
         16         agent.step(reward, done)
         17         state = next_state
    

    /home/workspace/task.py in step(self, rotor_speeds)
         37         pose_all = []
         38         for _ in range(self.action_repeat):
    ---> 39             done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
         40             reward += self.get_reward()
         41             pose_all.append(self.sim.pose)
    

    /home/workspace/physics_sim.py in next_timestep(self, rotor_speeds)
        120     def next_timestep(self, rotor_speeds):
        121         self.calc_prop_wind_speed()
    --> 122         thrusts = self.get_propeler_thrust(rotor_speeds)
        123         self.linear_accel = self.get_linear_forces(thrusts) / self.mass
        124 
    

    /home/workspace/physics_sim.py in get_propeler_thrust(self, rotor_speeds)
        111             V = self.prop_wind_speed[prop_number]
        112             D = self.propeller_size
    --> 113             n = rotor_speeds[prop_number]
        114             J = V / n * D
        115             # From http://m-selig.ae.illinois.edu/pubs/BrandtSelig-2011-AIAA-2011-1255-LRN-Propellers.pdf
    

    IndexError: index 1 is out of bounds for axis 0 with size 1


This agent should perform very poorly on this task.  And that's where you come in!

## Define the Task, Design the Agent, and Train Your Agent!

Amend `task.py` to specify a task of your choosing.  If you're unsure what kind of task to specify, you may like to teach your quadcopter to takeoff, hover in place, land softly, or reach a target pose.  

After specifying your task, use the sample agent in `agents/policy_search.py` as a template to define your own agent in `agents/agent.py`.  You can borrow whatever you need from the sample agent, including ideas on how you might modularize your code (using helper methods like `act()`, `learn()`, `reset_episode()`, etc.).

Note that it is **highly unlikely** that the first agent and task that you specify will learn well.  You will likely have to tweak various hyperparameters and the reward function for your task until you arrive at reasonably good behavior.

As you develop your agent, it's important to keep an eye on how it's performing. Use the code above as inspiration to build in a mechanism to log/save the total rewards obtained in each episode to file.  If the episode rewards are gradually increasing, this is an indication that your agent is learning.


```python
## TODO: Train your agent here.
from agents.agent import DDPG
from task import Task

num_episodes = 200
runtime = 5.
init_pose = np.array([0., 0., 25., 0., 0., 0.])
init_velocities = np.array([0., 0., 0.])
init_angle_velocities = np.array([0., 0., 0.])
target_pos = np.array([0., 0., 30.])

task = Task(init_pose=init_pose, init_velocities = init_velocities, init_angle_velocities = init_angle_velocities,
            runtime = runtime, target_pos=target_pos)
my_agent = DDPG(task)
score_tot = []

for i_episode in range(1, num_episodes + 1):
    state = my_agent.reset_episode()
    count, reward_tot, score = 0, 0, 0
    while True:
        count += 1
        action = my_agent.act(state)
        action = [min(900, action[0])]
        rotors_speeds = action*4
        next_state, reward, done = task.step(rotors_speeds)
        my_agent.step(action, reward, next_state, done)
        state = next_state
        reward_tot += reward
        if done:
            score = reward_tot/count
            score_tot.append(reward_tot/count)
            print("\rEpisode = {:4d}, score = {:7.3f}".format(i_episode, score))
            break
    sys.stdout.flush()
```

    Episode =    1, score = -18.234
    Episode =    2, score = -56.796
    Episode =    3, score = -56.098
    Episode =    4, score = -56.084
    Episode =    5, score = -56.083
    Episode =    6, score = -56.074
    Episode =    7, score = -56.086
    Episode =    8, score = -56.092
    Episode =    9, score = -56.097
    Episode =   10, score = -56.093
    Episode =   11, score = -56.095
    Episode =   12, score = -56.079
    Episode =   13, score = -56.093
    Episode =   14, score = -56.080
    Episode =   15, score = -56.091
    Episode =   16, score = -56.104
    Episode =   17, score = -56.090
    Episode =   18, score = -56.078
    Episode =   19, score = -56.072
    Episode =   20, score = -56.092
    Episode =   21, score = -56.070
    Episode =   22, score = -56.076
    Episode =   23, score = -56.096
    Episode =   24, score = -56.084
    Episode =   25, score = -56.077
    Episode =   26, score = -56.085
    Episode =   27, score = -56.089
    Episode =   28, score = -56.097
    Episode =   29, score = -56.091
    Episode =   30, score = -56.087
    Episode =   31, score = -56.088
    Episode =   32, score = -56.094
    Episode =   33, score = -56.090
    Episode =   34, score = -56.081
    Episode =   35, score = -56.079
    Episode =   36, score = -56.098
    Episode =   37, score = -56.089
    Episode =   38, score = -56.089
    Episode =   39, score = -56.078
    Episode =   40, score = -56.083
    Episode =   41, score = -56.074
    Episode =   42, score = -56.084
    Episode =   43, score = -56.103
    Episode =   44, score = -56.065
    Episode =   45, score = -56.091
    Episode =   46, score = -56.090
    Episode =   47, score = -56.084
    Episode =   48, score = -56.097
    Episode =   49, score = -56.078
    Episode =   50, score = -56.085
    Episode =   51, score = -56.089
    Episode =   52, score = -56.090
    Episode =   53, score = -56.095
    Episode =   54, score = -56.088
    Episode =   55, score = -56.063
    Episode =   56, score = -56.104
    Episode =   57, score = -56.086
    Episode =   58, score = -56.088
    Episode =   59, score = -56.084
    Episode =   60, score = -56.064
    Episode =   61, score = -56.062
    Episode =   62, score = -56.082
    Episode =   63, score = -56.092
    Episode =   64, score = -56.083
    Episode =   65, score = -56.094
    Episode =   66, score = -56.087
    Episode =   67, score = -56.091
    Episode =   68, score = -56.086
    Episode =   69, score = -56.080
    Episode =   70, score = -56.089
    Episode =   71, score = -56.095
    Episode =   72, score = -56.077
    Episode =   73, score = -56.088
    Episode =   74, score = -56.091
    Episode =   75, score = -56.095
    Episode =   76, score = -56.087
    Episode =   77, score = -56.073
    Episode =   78, score = -56.082
    Episode =   79, score = -56.094
    Episode =   80, score = -56.099
    Episode =   81, score = -56.085
    Episode =   82, score = -56.101
    Episode =   83, score = -56.098
    Episode =   84, score = -56.088
    Episode =   85, score = -56.092
    Episode =   86, score = -56.066
    Episode =   87, score = -56.099
    Episode =   88, score = -56.092
    Episode =   89, score = -56.091
    Episode =   90, score = -56.086
    Episode =   91, score = -56.096
    Episode =   92, score = -56.083
    Episode =   93, score = -56.088
    Episode =   94, score = -56.074
    Episode =   95, score = -56.087
    Episode =   96, score = -56.083
    Episode =   97, score = -56.097
    Episode =   98, score = -56.101
    Episode =   99, score = -56.096
    Episode =  100, score = -56.079
    Episode =  101, score = -56.090
    Episode =  102, score = -56.086
    Episode =  103, score = -56.099
    Episode =  104, score = -56.078
    Episode =  105, score = -56.080
    Episode =  106, score = -56.094
    Episode =  107, score = -56.078
    Episode =  108, score = -56.090
    Episode =  109, score = -56.091
    Episode =  110, score = -56.089
    Episode =  111, score = -56.087
    Episode =  112, score = -56.090
    Episode =  113, score = -56.090
    Episode =  114, score = -56.099
    Episode =  115, score = -56.099
    Episode =  116, score = -56.095
    Episode =  117, score = -56.071
    Episode =  118, score = -56.083
    Episode =  119, score = -56.084
    Episode =  120, score = -56.078
    Episode =  121, score = -56.080
    Episode =  122, score = -56.087
    Episode =  123, score = -56.086
    Episode =  124, score = -56.080
    Episode =  125, score = -56.087
    Episode =  126, score = -56.085
    Episode =  127, score = -56.075
    Episode =  128, score = -56.089
    Episode =  129, score = -56.095
    Episode =  130, score = -56.084
    Episode =  131, score = -56.078
    Episode =  132, score = -56.097
    Episode =  133, score = -56.100
    Episode =  134, score = -56.097
    Episode =  135, score = -56.086
    Episode =  136, score = -56.089
    Episode =  137, score = -56.101
    Episode =  138, score = -56.089
    Episode =  139, score = -56.094
    Episode =  140, score = -56.084
    Episode =  141, score = -56.096
    Episode =  142, score = -56.075
    Episode =  143, score = -56.102
    Episode =  144, score = -56.097
    Episode =  145, score = -56.103
    Episode =  146, score = -56.088
    Episode =  147, score = -56.099
    Episode =  148, score = -56.081
    Episode =  149, score = -56.098
    Episode =  150, score = -56.090
    Episode =  151, score = -56.091
    Episode =  152, score = -56.100
    Episode =  153, score = -56.087
    Episode =  154, score = -56.096
    Episode =  155, score = -56.098
    Episode =  156, score = -56.084
    Episode =  157, score = -56.086
    Episode =  158, score = -56.090
    Episode =  159, score = -56.091
    Episode =  160, score = -56.103
    Episode =  161, score = -56.082
    Episode =  162, score = -56.086
    Episode =  163, score = -56.091
    Episode =  164, score = -56.087
    Episode =  165, score = -56.097
    Episode =  166, score = -56.093
    Episode =  167, score = -56.083
    Episode =  168, score = -56.083
    Episode =  169, score = -56.091
    Episode =  170, score = -56.087
    Episode =  171, score = -56.082
    Episode =  172, score = -56.097
    Episode =  173, score = -56.082
    Episode =  174, score = -56.075
    Episode =  175, score = -56.079
    Episode =  176, score = -56.091
    Episode =  177, score = -56.084
    Episode =  178, score = -56.072
    Episode =  179, score = -56.088
    Episode =  180, score = -56.083
    Episode =  181, score = -56.082
    Episode =  182, score = -56.102
    Episode =  183, score = -56.093
    Episode =  184, score = -56.079
    Episode =  185, score = -56.085
    Episode =  186, score = -56.079
    Episode =  187, score = -56.094
    Episode =  188, score = -56.075
    Episode =  189, score = -56.080
    Episode =  190, score = -56.080
    Episode =  191, score = -56.083
    Episode =  192, score = -56.090
    Episode =  193, score = -56.096
    Episode =  194, score = -56.084
    Episode =  195, score = -56.078
    Episode =  196, score = -56.085
    Episode =  197, score = -56.073
    Episode =  198, score = -56.092
    Episode =  199, score = -56.084
    Episode =  200, score = -56.085
    

## Plot the Rewards

Once you are satisfied with your performance, plot the episode rewards, either from a single run, or averaged over multiple runs. 


```python
## TODO: Plot the rewards.
import pandas as pd
reward_df = pd.DataFrame({"episode":[c for c in range(1,num_episodes + 1)], "score":score_tot})
plt.plot(reward_df['episode'], reward_df["score"])
```




    [<matplotlib.lines.Line2D at 0x7f9eb20a66a0>]




![png](output_24_1.png)


## Reflections

**Question 1**: Describe the task that you specified in `task.py`.  How did you design the reward function?

**Answer**: my goal is try to training my agent to get takeoff, so intuitively, my reward function is as same as the example, since we are in a classic euclidean space, therefore i can measure the reward as a norm bewteen two points.

**Question 2**: Discuss your agent briefly, using the following questions as a guide:

- What learning algorithm(s) did you try? What worked best for you?
- What was your final choice of hyperparameters (such as $\alpha$, $\gamma$, $\epsilon$, etc.)?
- What neural network architecture did you use (if any)? Specify layers, sizes, activation functions, etc.

**Answer**:
- since both the action space and the states space are continuous, so i used the DDPG algorithme.
- I set 0.05 for Critic's learning rate and 0.01 for Actor's, 0.9 for discounted rate, and 0.001 for the soft replacement, i think  a small tau will guarantee the convergence.
- For the Actor, i have 3 hidden layers with 32, 164, 32 units inside, each of them active by relu function. For Critic, i have 2 layers for the V and the Q, and connect them by a full layer with relu function. 

**Question 3**: Using the episode rewards plot, discuss how the agent learned over time.

- Was it an easy task to learn or hard?
- Was there a gradual learning curve, or an aha moment?
- How good was the final performance of the agent? (e.g. mean rewards over the last 10 episodes)

**Answer**:
- it seems like a impossible mission for my agent, it stuck after 3 episode, and the reward stay -56.

**Question 4**: Briefly summarize your experience working on this project. You can use the following prompts for ideas.

- What was the hardest part of the project? (e.g. getting started, plotting, specifying the task, etc.)
- Did you find anything interesting in how the quadcopter or your agent behaved?

**Answer**:
- i think it's coding part, i understood each part before this project, and i even do demostrations for some formula, but when i try to put them together, i can't remember anything.....



```python

```
