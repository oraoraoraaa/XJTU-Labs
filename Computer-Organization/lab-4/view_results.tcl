set lab4_dir [file normalize [file join [file dirname [info script]] lab4-verilog]]
proc lab4_view {name} {
 global lab4_dir
 set tops [dict create 0-tb-ram-1kx16 tb_ram_1kx16 1-tb-ram-4kx32 tb_ram_4kx32 2-tb-ram-bank-boundary tb_ram_4kx32 3-tb-ram-inout tb_ram_inout 4-tb-instruction-memory tb_instruction_memory 5-tb-register-file tb_register_file]
 set top [dict get $tops $name]
 catch {close_sim}
 open_wave_database -noautoloadwcfg $lab4_dir/.verification/$top.wdb
 open_wave_config $lab4_dir/lab4-verilog.srcs/sim_1/wave/$name.wcfg
}
