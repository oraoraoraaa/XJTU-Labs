# task 1
python mcp_check.py --list --unwrap
python mcp_check.py get_time --unwrap
python mcp_check.py get_memory_info --unwrap
python host_deepseek.py

# task 2
# module 1
python mcp_check.py list_top_processes --unwrap
# linux environment only
LAB_PID=$$
python mcp_check.py get_process_memory \
  --args "{\"pid\":$LAB_PID,\"max_maps\":3}" --unwrap
