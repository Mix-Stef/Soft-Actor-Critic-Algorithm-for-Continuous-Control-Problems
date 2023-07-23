# Soft Actor Critic (SAC) Algorithm for Continuous Control Problems

In this repo the "Soft Actor-Critic" algorithm is implemented from scratch in order to solve the problem of continuous control for simple (low-dimensional) enviroments. *PyTorch* is used for the neural network design and training and *Gymnasium* (OpenAI Gym) for testing the algorithm with some ready-to-go enviroments. At a later stage we will try to design custom enviroments that come from the field of Process Control.

Soft Actor-Critic is a Deep Reinforcement Learning (DRL) algorithm that aims to find an optimal policy for the agent to utilize inside the state space in order to maximize its reward. The main difference between SAC and other DRL algorithms (like DDPG and Twin Delayed DDPG) is the extra *entropy term* that is added to the loss functions. This ensures that the agent explores the environment more efficiently. 

The algorithms utilizes three types of neural networks, the *Value Network*, $V_{\psi}$, the *Target Value Network*, $V_{\bar{\psi}}$, one or more *Critic Networks*, $Q_{\theta}$ and a *Policy (or Actor) Network*, $\pi_{\phi}$. The respecive loss functions are:

* Value Network Loss: $$\displaystyle J_{V}(\psi)= E_{s_{t} \sim D} \big[\frac{1}{2}(V_{\psi}(s_{t}) - E_{a_{t} \sim \pi_{\phi}}[Q_{\theta}(s_{t}, a_{t}) - \log \pi_{\phi}(a_{t} | s_{t}])])^2\big]$$

* Critic Network Loss: $$\displaystyle J_{Q}(\theta) = E_{(s_{t}, a_{t}) \sim D} \big[ \frac{1}{2} \big(Q_{\theta}(s_{t}, a_{t}) - \hat{Q}(s_{t}, a_{t})  \big) ^ 2 \big]$$ where $$\hat{Q}(s_{t}, a_{t}) = r(s_{t}, a_{t}) + \gamma E_{s_{t+1} \sim D} \big[V_{\bar{\psi}}(s_{t+1}) \big]$$

* Actor Loss: $$\displaystyle J_{\pi}(\phi) = E_{(s_{t}, a_{t}) \sim D} \big[ \log \pi_{\phi}(a_{t}|s_{t}) - Q_{\theta}(s_{t}, a_{t) \big]$$ if we do not use the *repametrization trick*. If we reparametrize, then the actor loss becomes: $$\displaystyle \bar{J}_{\pi}(\phi) = E_{s_{t} \sim D, e_{t} \sim N} \big[ \log \pi_{\phi}(f_{\phi}(e_{t};s_{t})|s_{t}) - Q_{\theta}(f_{\phi}(e_{t}|s_{t}), a_{t) \big]$$ with $e_{t}$ a random (e.g. Gaussian) noise vector and $f_{\phi}$ some non-linearity, like tanh



