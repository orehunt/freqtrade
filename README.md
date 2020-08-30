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
- colored progressbar with statistics, colored log legel
- option to abort hyperopt if there isn't a minimum ratio for all the pairs/timerange
-------------------------------------------------------------------------------
- datahandler with parquet as backing store
- config option to avoid plotting pairs profit
- stop hyperopt when convergence ratio reached (num of same points asked over saved ones)
- add cofig optionn to increase empty count

## Files that use amounts config 
```
- freqtradebot.py
- persistence.py
- optimize/hyperopt.py
- pairlist/PrecisionFilter.py
- strategy/interface.py
```

## Sell rate configuration
Sell signals happen on open on the next candle. The rate is calculated as:
```
mean([open, low, min(open, close)])
norm_vol: volume normalized in a rolling window by average trade duration
vol_diff: 1 - norm_vol
if open > close:
       open_vol = norm_vol
       low_vol = vol_diff
else:
     open_vol = vol_diff
     low_vol = norm_vol
open * norm_vol + low * vol_diff
```
