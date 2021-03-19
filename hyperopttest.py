# from hyperopt import fmin, tpe, hp
# import hyperopt.pyll.stochastic
#
# space = {
#     'x': hp.uniform('x', 0, 1),
#     'y': hp.normal('y', 0, 1),
#     'name': hp.choice('name', ['alice', 'bob']),
# }
#
# print(hyperopt.pyll.stochastic.sample(space))

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt

fspace = {
    'x': hp.uniform('x', -5, 5)
}

def f(params):
    x = params['x']
    val = x**2
    return {'loss': val, 'status': STATUS_OK}

trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=1000, trials=trials)

print ('best:', best)

# print( 'trials:')
# for trial in trials.trials[:2]:
#     print (trial)

# f, ax = plt.subplots(1)
# xs = [t['tid'] for t in trials.trials]
# ys = [t['misc']['vals']['x'] for t in trials.trials]
# ax.set_xlim(xs[0]-10, xs[-1]+10)
# ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
# ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
# ax.set_xlabel('$t$', fontsize=16)
# ax.set_ylabel('$x$', fontsize=16)

f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$val$ $vs$ $x$ ', fontsize=18)
ax.set_xlabel('$x$', fontsize=16)
ax.set_ylabel('$val$', fontsize=16)
plt.show()