set lab3_dir [file normalize [file join [file dirname [info script]] lab3-verilog]]
proc lab3_view {name} {
 global lab3_dir
 set tops [dict create 0-tb-full-adder tb_full_adder 1-tb-adders tb_adders 2-tb-alu32 tb_alu32 3-tb-alu32-flags tb_alu32 4-tb-alu32-shift tb_alu32 5-tb-alu-bus tb_alu_bus]
 set top [dict get $tops $name]
 catch {close_sim}
 open_wave_database -noautoloadwcfg $lab3_dir/.verification/$top.wdb
 open_wave_config $lab3_dir/lab3-verilog.srcs/sim_1/wave/$name.wcfg
}
