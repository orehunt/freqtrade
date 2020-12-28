import numba as nb

nb_opt = nb.types.Optional
nb_str = nb.types.unicode_type
nb_un = nb.types.UnionType
config_val = nb_opt(nb_un([nb_str, nb.types.int64, nb.types.float64, nb.types.boolean]))
config_val = nb.types.int64
config_type = nb.typeof(nb.typed.Dict.empty(key_type=nb_str, value_type=config_val))
nb_str_list = nb.typeof(nb.typed.List.empty_list(item_type=nb_str))
dates_dict_type = nb.typeof(
    nb.typed.Dict.empty(key_type=nb_str, value_type=nb.int64[:])
)
