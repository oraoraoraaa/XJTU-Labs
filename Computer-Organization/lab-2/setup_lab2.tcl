# In the existing Vivado project's Tcl Console:
# source D:/Local/Github/XJTU-Labs/Computer-Organization/lab-2/setup_lab2.tcl
set lab2_dir [file normalize [file join [file dirname [info script]] lab2-verilog]]
foreach f [glob -directory $lab2_dir/lab2-verilog.srcs/sources_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sources_1 $f}
}
foreach f [glob -directory $lab2_dir/lab2-verilog.srcs/sim_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sim_1 $f}
}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

proc lab2_sim {top} {
    catch {close_sim}
    set_property top $top [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation -simset sim_1 -mode behavioral
    log_wave -r /
    run all
}
puts {Lab 2 sources loaded. Example: lab2_sim tb_counter_mod99}
