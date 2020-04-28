## Changes
- heavily parallel hyperopt with per worker diverse acquisition/model setup and count/time based logging
- tuning of exploitation/exploration based on backtest results
- hyperopt with dynamic epochs and effort (a best guess on when to give up optimization, with tunable effort)
- ability to resume previous evaluation
- cross validation mode for hyperopt, load parameters from previous evaluated epochs, test against different configuration (timerange, pairlist, exchange)
- sampling for epochs, filter epochs by (profits, loss, etc...) sort by specified metric. (used by hyperopt-list, hyperopt CV), or filter by normalized best epochs, or deduplicate epochs by metric value
- parallel pairs signal evaluation as a per strategy config
- time weighted roi calculation as a per strategy config
- pair wise amounts configuration (roi, stoploss, trailing) as a per strategy config (not implemented in hyperopt, edge, rpc/telegram)
- more persistent trials storage with pandas hdf support...hyperopt class + strategy class is mapped to an hdf file, every lossfunc+params is mapped to a key (table)
- ability to `prefer_sell_signal` which prioritizes a sell when both buy and sell signals are set

## Files that use amounts config 
```
- freqtradebot.py
- persistence.py
- optimize/hyperopt.py
- pairlist/PrecisionFilter.py
- strategy/interface.py
```
