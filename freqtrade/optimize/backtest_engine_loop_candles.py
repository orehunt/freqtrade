from numpy import ndarray
from pandas import DataFrame

from freqtrade.optimize.backtest_nb import Features, define_callbacks, iter_triggers


def _loop_candles_select_triggers(
    self, df_vals: ndarray, bought: ndarray, bought_ranges: ndarray, bts_vals: ndarray,
) -> DataFrame:
    v = self._get_vars(df_vals, bought, bought_ranges)
    (
        fl_cols,
        it_cols,
        names,
        fl,
        it,
        bl,
        cmap,
        nan_early_idx,
        roi_timeouts,
        roi_values,
        trg_range_max,
    ) = self._nb_vars(v)

    # NOTE: functions are cached based on signatures, signatures are based on types
    # to load the correct cache when enabling or disabling features, we have to pass
    # True or None, (not False) such that the type of the tuple changes and and the sig/cache does too
    feat_dict = {
        k: v[k] or None
        for k in (
            "roi_enabled",
            "stoploss_enabled",
            "trailing_enabled",
            "not_position_stacking",
        )
    }
    define_callbacks(feat_dict)
    feat = Features(**feat_dict)

    iter_triggers(
        fl_cols,
        it_cols,
        names,
        fl,
        it,
        bl,
        cmap,
        nan_early_idx,
        roi_timeouts,
        roi_values,
        trg_range_max,
        feat,
    )
    bts_vals = self._assign_triggers_vals(
        bts_vals, bought, v["triggers"], v["col_names"]
    )
    return bts_vals
