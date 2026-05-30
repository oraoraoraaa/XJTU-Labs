  out_dir=./test-logs
  mkdir -p "$out_dir"

  cc -std=c11 -O2 -Wall -Wextra -o icg main.c
  if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
  fi

  out_file="$out_dir/all.out"
  err_file="$out_dir/all.err"
  tmp_err="$out_dir/.tmp.err"
  : > "$out_file"
  : > "$err_file"

  fail=0
  fail_list=""
  for f in ../test-set/*.src; do
    header="== $f =="
    echo "$header"
    echo "" >> "$out_file"
    echo "$header" >> "$out_file"
    echo "----------------------------------------" >> "$out_file"
    ./icg "$f" >> "$out_file" 2> "$tmp_err"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
      echo "" >> "$err_file"
      echo "$header" >> "$err_file"
      echo "----------------------------------------" >> "$err_file"
      cat "$tmp_err" >> "$err_file"
      echo "STATUS: FAIL (exit $exit_code)" >> "$err_file"
      echo "FAIL: $f (exit $exit_code)"
      fail=$((fail + 1))
      fail_list="$fail_list $f"
    else
      echo "STATUS: OK" >> "$out_file"
      echo "PASS: $f"
    fi
  done

  if [ $fail -ne 0 ]; then
    echo "FAIL: $fail case(s)"
    echo "FAILED:$fail_list"
    echo "See $out_file and $err_file for details"
    exit 1
  fi
  echo "ALL PASS"
  echo "See $out_file and $err_file for details"
