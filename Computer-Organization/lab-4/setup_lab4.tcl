# In the existing Vivado project's Tcl Console:
# source D:/Local/Github/XJTU-Labs/Computer-Organization/lab-2/setup_lab4.tcl
set lab4_dir [file normalize [file join [file dirname [info script]] lab4-verilog]]
foreach f [glob -directory $lab4_dir/lab4-verilog.srcs/sources_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sources_1 $f}
}
foreach f [glob -directory $lab4_dir/lab4-verilog.srcs/sim_1/new *.v] {
    if {[llength [get_files -quiet $f]] == 0} {add_files -fileset sim_1 $f}
}
add_files -norecurse -fileset sources_1 $lab4_dir/lab4-verilog.srcs/sources_1/new/imem.hex
set_property file_type {Memory Initialization Files} [get_files */imem.hex]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

proc lab4_sim {top} {
    catch {close_sim}
    set_property top $top [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation -simset sim_1 -mode behavioral
    log_wave -r /
    run all
}
puts {Lab 4 sources loaded. Example: lab4_sim tb_register_file}
