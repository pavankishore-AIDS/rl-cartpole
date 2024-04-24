# CART POLE BALANCING

## AIM
To develop and fine tune the Monte Carlo algorithm to stabilize the Cart Pole.

## PROBLEM STATEMENT
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in “Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem”. A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces in the left and right direction on the cart.

The action is a ndarray with shape (1,) which can take values {0, 1} indicating the direction of the fixed force the cart is pushed with.

i) 0: Push cart to the left

ii) 1: Push cart to the right

## MONTE CARLO CONTROL ALGORITHM FOR CART POLE BALANCING
```
Name: Pavan Kishore.M
Reg No: 212221230076
```

```Python
def create_bins(n_bins=g_bins, n_dim=4):

    bins = [
        np.linspace(-4.8, 4.8, n_bins),
        np.linspace(-4, 4, n_bins),
        np.linspace(-0.418, 0.418, n_bins),
        np.linspace(-4, 4, n_bins)
    ]

    return bins
```

```Python
def discretize_state(observation, bins):

    binned_state = []

    for i in range(len(observation)):
        d = np.digitize(observation[i], bins[i])
        binned_state.append( d - 1)

    return tuple(binned_state)
```

```Python
def decay_schedule(
    init_value, min_value, decay_ratio,
    max_steps, log_start = -2, log_base=10):
    decay_steps = int(max_steps*decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(
      log_start, 0, decay_steps,
      base = log_base, endpoint = True)[::-1]
    values = (values -values.min())/(values.max() - values.min())
    values = (init_value - min_value)*values +min_value
    values = np.pad(values, (0, rem_steps), 'edge')

    return values
```

```Python
def generate_trajectory(
    select_action, Q, epsilon,
    env, max_steps=200):
    done, trajectory = False, []
    bins = create_bins(g_bins)
    
    observation,_ = env.reset()
    state = discretize_state(observation, bins)
    
    for t in count():
        action = select_action(state, Q, epsilon)
        observation, reward, done, _, _ = env.step(action)
        next_state = discretize_state(observation, bins)
        if not done:                
            if t >= max_steps-1:
                break
            experience = (state, action,
                    reward, next_state, done)                            
            trajectory.append(experience)                
        else:
            experience = (state, action,
                    -100, next_state, done)
            trajectory.append(experience)                
            #time.sleep(2)
            break
        state = next_state

    return np.array(trajectory, dtype=object)
```

## MONTE CARLO CONTROL FUNCTION
```Python
def mc_control (env,n_bins=g_bins, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 3000, max_steps = 200, first_visit = True, init_Q=None):
    
    nA = env.action_space.n
    discounts = np.logspace(0, max_steps,
                            num = max_steps, base = gamma,
                            endpoint = False)
    alphas = decay_schedule(init_alpha, min_alpha,
                            0.9999, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                            0.99, n_episodes)
    pi_track = []
    global Q_track
    global Q
    
    
    if init_Q is None:
        Q = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    else:
        Q = init_Q
        
    n_elements = Q.size
    n_nonzero_elements = 0
    
    Q_track = np.zeros([n_episodes] + [n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
    select_action = lambda state, Q, epsilon: np.argmax(Q[tuple(state)]) if np.random.random() > epsilon else np.random.randint(len(Q[tuple(state)]))

    progress_bar = tqdm(range(n_episodes), leave=False)
    steps_balanced_total = 1
    mean_steps_balanced = 0
    for e in progress_bar:        
        trajectory = generate_trajectory(select_action, Q, epsilons[e],
                                    env, max_steps)
        
        steps_balanced_total = steps_balanced_total + len(trajectory)
        mean_steps_balanced = 0
        
        visited = np.zeros([n_bins]*env.observation_space.shape[0] + [env.action_space.n],dtype =np.float64)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            #if visited[tuple(state)][action] and first_visit:
            #    continue    
            visited[tuple(state)][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps]*trajectory[t:, 2])
            Q[tuple(state)][action] = Q[tuple(state)][action]+alphas[e]*(G - Q[tuple(state)][action])
        Q_track[e] = Q
        n_nonzero_elements = np.count_nonzero(Q)
        pi_track.append(np.argmax(Q, axis=env.observation_space.shape[0]))
        if e != 0:
            mean_steps_balanced = steps_balanced_total/e
        #progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], Steps=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}", NonZeroValues="{0}/{1}".format(n_nonzero_elements,n_elements))
        progress_bar.set_postfix(episode=e, Epsilon=epsilons[e], StepsBalanced=f"{len(trajectory)}" ,MeanStepsBalanced=f"{mean_steps_balanced:.2f}")
        
    print("mean_steps_balanced={0},steps_balanced_total={1}".format(mean_steps_balanced,steps_balanced_total))
    V = np.max(Q, axis=env.observation_space.shape[0])
    pi = lambda s:{s:a for s, a in enumerate(np.argmax(Q, axis=env.observation_space.shape[0]))}[s]

    return Q, V, pi
```

## OUTPUT:

1.Specify the average number of steps achieved within two minutes when the Monte Carlo (MC) control algorithm is initiated with zero-initialized Q-values. 
![324844181-8ccb911d-6a22-43f3-b9e2-751ed8dee898](https://github.com/pavankishore-AIDS/rl-cartpole/assets/94154941/42168043-72c1-4897-8208-595b5400061b)

2.Mention the average number of steps maintained over a four-minute period when the Monte Carlo (MC) control algorithm is executed with pretrained Q-values.
![324844268-d26e73c8-2365-478b-b099-a5041047de8f](https://github.com/pavankishore-AIDS/rl-cartpole/assets/94154941/1b104534-2c4f-47b6-9cb6-e434f41f1917)

3.In your submission text, mention the average number of steps maintained over a four-minute period when the Monte Carlo (MC) control algorithm is executed with pretrained Q-values.
![324844358-3a6d6d0a-d6bd-4605-a96d-ea40a719ecb5](https://github.com/pavankishore-AIDS/rl-cartpole/assets/94154941/eaeb538a-f909-4bcb-84ec-d0f23397d515)

## RESULT:
Thus, a Python program is developed to find the optimal policy for the given cart-pole environment using the Monte Carlo algorithm.
