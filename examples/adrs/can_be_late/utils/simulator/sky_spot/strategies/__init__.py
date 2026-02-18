import importlib

__all__ = []

_REQUIRED_MODULES = [
    'strategy',
    'strawman',
    'time_sliced',
    'time_sliced_by_num',
    'random_time_sliced',
    'group_time_sliced',
    'loose_time_sliced',
    'loose_time_sliced_by_num',
    'loose_time_sliced_vdt_by_num',
    'on_demand',
    'only_spot',
    'ideal_no_overhead',
    'ideal_ilp_overhead',
    'ideal_ilp_overhead_sliced_by_num',
    'ideal_ilp_overhead_sliced',
    'rc_threshold',
    'rc_dc_threshold',
    'rc_dd_threshold',
    'rc_vd_threshold',
    'rc_vdt_threshold',
    'rc_vdt_allow_idle_threshold',
    'rc_v2dt_threshold',
    'rc_vd_no_k_threshold',
    'rc_lw_threshold',
    'rc_ec_threshold',
    'rc_gec_threshold',
    'rc_cr_threshold',
    'rc_cr_no_keep_threshold',
    'rc_1cr_threshold',
    'rc_next_spot_threshold',
    'rc_next_spot_single_threshold',
    'rc_next_wait_spot_threshold',
    'quick_optimal',
    'quick_optimal_paranoid',
    'quick_optimal_sliced_by_num',
    'quick_optimal_more_sliced_by_num',
    'rc_cr_threshold_no_condition2',
    'multi_region_rc_cr_threshold',
    'multi_region_rc_cr_no_cond2',
    'multi_region_rc_cr_randomized',
    'multi_region_rc_cr_reactive',
    'lazy_cost_aware_multi',
    'evolutionary_simple_v2',
]


def _safe_import(module_name: str) -> None:
    try:
        module = importlib.import_module(f'.{module_name}', __name__)
        globals()[module_name] = module
        __all__.append(module_name)
    except ModuleNotFoundError:
        pass


for _name in _REQUIRED_MODULES:
    _safe_import(_name)
