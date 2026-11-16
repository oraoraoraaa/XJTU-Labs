# In the existing Vivado project's Tcl Console:
# source D:/Local/Github/XJTU-Labs/Computer-Organization/lab-2/setup_lab3.tcl
set lab3_dir [file normalize [file join [file dirname [info script]] lab3-verilog]]
foreach f [glob -directory $lab3_dir/lab3-verilog.srcs/sources_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sources_1 $f}
}
foreach f [glob -directory $lab3_dir/lab3-verilog.srcs/sim_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sim_1 $f}
}
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

proc lab3_sim {top} {
    catch {close_sim}
    set_property top $top [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation -simset sim_1 -mode behavioral
    log_wave -r /
    run all
}
puts {Lab 3 sources loaded. Example: lab3_sim tb_alu32}
