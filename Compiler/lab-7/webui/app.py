from flask import Flask, render_template, request, jsonify, send_file
import subprocess
import os
from pathlib import Path
import tempfile
import shlex

ROOT = Path(__file__).resolve().parents[3]  # repository root (XJTU-Labs)
ICG = Path(__file__).resolve().parent / 'bin' / 'icg'
WEBUI_DIR = Path(__file__).resolve().parent
OUT_QUADS = WEBUI_DIR / 'out.quads'
OUT_ASM = WEBUI_DIR / 'out.s'

app = Flask(__name__, template_folder='templates', static_folder='static')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/build', methods=['POST'])
def build():
    script = Path(__file__).resolve().parent / 'build_icg.sh'
    try:
        subprocess.check_output([str(script)], cwd=str(script.parent), stderr=subprocess.STDOUT)
        return jsonify({ 'ok': True, 'path': str(ICG) })
    except subprocess.CalledProcessError as e:
        return jsonify({ 'ok': False, 'error': e.output.decode() })


@app.route('/list_tests')
def list_tests():
    testdir = ROOT / 'Compiler' / 'test-set'
    files = []
    try:
        for p in sorted(testdir.iterdir()):
            if p.is_file():
                files.append(p.relative_to(ROOT).as_posix())
    except FileNotFoundError:
        return jsonify({ 'ok': False, 'error': f'test-set not found: {testdir}' }), 404
    return jsonify(files)


@app.route('/run', methods=['POST'])
def run_icg():
    data = request.json or {}
    path = data.get('path')
    if not path:
        return jsonify({ 'ok': False, 'error': 'path required' }), 400
    infile = ROOT / path
    if not infile.exists():
        return jsonify({ 'ok': False, 'error': 'input not found' }), 404
    if not ICG.exists():
        return jsonify({ 'ok': False, 'error': 'icg not built' }), 400
    try:
        out = subprocess.check_output([str(ICG), str(infile)], cwd=str(ICG.parent), stderr=subprocess.STDOUT, timeout=10)
        quads = out.decode()
        try:
            OUT_QUADS.write_text(quads, encoding='utf-8')
        except Exception:
            pass
        return jsonify({ 'ok': True, 'quads': quads, 'out_quads': str(OUT_QUADS) })
    except subprocess.CalledProcessError as e:
        return jsonify({ 'ok': False, 'error': e.output.decode() })
    except subprocess.TimeoutExpired:
        return jsonify({ 'ok': False, 'error': 'timeout' })


@app.route('/source')
def get_source():
    path = request.args.get('path')
    if not path:
        return jsonify({ 'ok': False, 'error': 'path required' }), 400
    infile = ROOT / path
    if not infile.exists():
        return jsonify({ 'ok': False, 'error': 'input not found' }), 404
    try:
        text = infile.read_text(encoding='utf-8')
        return jsonify({ 'ok': True, 'text': text })
    except Exception as e:
        return jsonify({ 'ok': False, 'error': str(e) }), 500


@app.route('/translate', methods=['POST'])
def translate():
    from quad_to_arm import parse_quads, quad_to_arm
    data = request.json or {}
    quads_text = data.get('quads','')
    quads = parse_quads(quads_text)
    asm = quad_to_arm(quads)
    try:
        OUT_ASM.write_text(asm, encoding='utf-8')
    except Exception:
        pass
    return jsonify({ 'ok': True, 'asm': asm, 'out_asm': str(OUT_ASM) })


if __name__ == '__main__':
    app.run(port=5000)
