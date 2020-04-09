## Changes
- pair wise amounts configuration (roi, stoploss, trailing) (not implemented in hyperopt, edge, rpc/telegram)
- hyperopt with multi optimizers (explorative), or shared optimizer (exploitative)
- hyperopt with dynamic epochs and effort (a best guess on when to give up optimization, with tunable effort)
- cross validation mode for hyperopt, load parameters from evaluated epochs, test against different configuration (timerange, pairlist...)
- sampling for epochs, filter epochs by (profits, loss, etc...) sort by specified metric. (used by hyperopt-list, hyperopt CV, ability to continue optimization from sampled epochs)
- parallel pairs signal evaluation as a per strategy config
- time weighted roi calculation as a per strategy config

## Files that use amounts config 
```
- freqtradebot.py
- persistence.py
- optimize/hyperopt.py
- pairlist/PrecisionFilter.py
- strategy/interface.py
```
