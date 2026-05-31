Simple WebUI for Lab 7

Features:
- Build the Lab 6 intermediate-code generator (`icg`).
- Run `icg` on test inputs to get quadruples.
- Translate a subset of quadruples to ARM64 assembly (text output).
- Minimal single-page UI to select an input and view outputs.

Notes:
- Producing a runnable ARM64 executable requires an aarch64 toolchain on the host
  (e.g. `aarch64-linux-gnu-gcc`) or running on an aarch64 machine. This project
  will only emit assembly; assembling/linking is left to the user if the toolchain
  is not present.

Quick start (Linux):

1. Install Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Build `icg` (Lab 6 generator):

```bash
./build_icg.sh
```

3. Run the web UI:

```bash
FLASK_APP=app.py flask run --host=127.0.0.1 --port=5000
```

Open http://127.0.0.1:5000
