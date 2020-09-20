def eval_weights_rotate(
    state: tuple, sgn: str, params: dict, d: DataFrame, preds=[], limits={}, weights={}
) -> (np.ndarray, np.ndarray, np.ndarray):
    mode = state.mode
    cond = state.cond
    se = int(os.environ["FQT_SE"])
    defv = state.defv

    gr = np.full(len(d), defv)
    lo = np.full(len(d), defv)
    signal = []

    c = 0
    for nl, lim in enumerate(limits):
        for n in range(len(lim) // 4):
            val = params if se >= 0 and se <= n < state.k else weights
            gr = sgf(preds[n], lim[f"{sgn}.gr{n}"], val[f"{sgn}.gr{nl+n+c}"], gr)
            lo = slf(preds[n], lim[f"{sgn}.lo{n}"], val[f"{sgn}.lo{nl+n+c}"], lo)
        c += 1
    val = params if se < 0 else weights
    signal.extend([gr >= val[f"{sgn}.mean.gr0"], lo >= val[f"{sgn}.mean.lo0"]])
    return reduce(andf, signal), gr, lo


def eval_weights_limlist(
        state: tuple, sgn: str, params: dict, d: DataFrame, preds={}, limits=[], mean=0.26
):
    """"""
    signal = []
    limlists = []
    cond = state.cond
    # first compute all the limits lists int len(limits) bools
    for lim_dict in limits:
        gr = np.full(len(d), state.defv)
        lo = np.full(len(d), state.defv)
        for n in range(
            len(lim_dict) // 4
        ):  # divide by number of signals and limits (2 * 2)
            gr = sge(preds[n], lim_dict[f"{sgn}.gr{n}"], gr)
            lo = sle(preds[n], lim_dict[f"{sgn}.lo{n}"], lo)
        limlists.append(gr & lo)
    # now apply weights on those bools
    gr = np.full(len(d), 0.0)
    lo = np.full(len(d), 0.0)
    for n, ll in enumerate(limlists):
        gr = ll.astype(float) * params[f"{sgn}.gr{n}"] + gr
        lo = ll.astype(float) * params[f"{sgn}.lo{n}"] + lo
    signal.extend([gr >= mean[sgn]["gr"], lo >= mean[sgn]["lo"]])

    return reduce(andf, signal), gr, lo

