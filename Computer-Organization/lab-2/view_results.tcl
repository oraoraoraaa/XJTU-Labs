# Source this file in Vivado, then use e.g. lab2_view 6-tb-counter-mod99.
set lab2_results_dir [file normalize [file join [file dirname [info script]] lab2-verilog]]
proc lab2_view {view} {
    global lab2_results_dir
    set tops [dict create \
        1-tb-sr-latch tb_sr_latch \
        2-tb-d-latch-reset tb_d_latch_reset \
        3-tb-d-ff-control tb_d_ff_control \
        4-tb-register-n tb_register_n \
        5-tb-shift-register tb_shift_register \
        6-tb-counter-mod99 tb_counter_mod99 \
        7-tb-counter-mod99-controls tb_counter_mod99 \
        8-tb-fsm-div3 tb_fsm_div3 \
        9-tb-sequence-01 tb_sequence_01]
    if {![dict exists $tops $view]} {error "Available views: [dict keys $tops]"}
    set top [dict get $tops $view]
    catch {close_sim}
    open_wave_database -noautoloadwcfg $lab2_results_dir/.verification/$top.wdb
    open_wave_config $lab2_results_dir/lab2-verilog.srcs/sim_1/wave/$view.wcfg
}
puts {Saved results ready. Example: lab2_view 6-tb-counter-mod99}
