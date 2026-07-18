import numpy as np, time
from toy_wam import *
from run_probe import make_dataset, sample_scene, run_episode
t00 = time.time()
O, A, Z = make_dataset(2048, seed=1)
print('data %.1fs' % (time.time() - t00), flush=True)
t0 = time.time()
m = ToyWAM('direct', seed=3)
hist = m.train(O, A, Z, steps=4000, lr=3e-3, lr_min=3e-4, log_every=1000)
print('losses', [round(h[1], 4) for h in hist], '%.0fs' % (time.time() - t0), flush=True)
h1 = np.tanh(m._center(O[:256]) @ m.params['W1'] + m.params['b1'])
print('sat frac>0.99:', (np.abs(h1) > 0.99).mean(), flush=True)
t0 = time.time()
succ = 0
dists = []
for ep in range(60):
    rng = np.random.default_rng(10_000 + ep)
    agent, goal = sample_scene(rng)
    log = run_episode(m, agent, goal, attack=None, seed=ep)
    succ += log['success']
    dists.append(log['final_dist'])
print('direct clean success: %d/60, mean final dist %.3f, episodes %.0fs'
      % (succ, np.mean(dists), time.time() - t0), flush=True)
