from flask import Flask, request, jsonify, render_template
import sqlite3
import os
import threading
from datetime import datetime
import easyocr
import numpy as np
import cv2
import re
from werkzeug.utils import secure_filename
from PIL import Image
import io

try:
    import certifi
    os.environ.setdefault('SSL_CERT_FILE', certifi.where())
    os.environ.setdefault('REQUESTS_CA_BUNDLE', certifi.where())
except Exception:
    certifi = None

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif', 'webp'}

_reader = None
_reader_error = None
_reader_lock = threading.Lock()


def get_ocr_reader():
    global _reader, _reader_error
    if _reader is not None:
        return _reader

    if _reader_error is not None:
        raise RuntimeError(_reader_error)

    with _reader_lock:
        if _reader is not None:
            return _reader
        if _reader_error is not None:
            raise RuntimeError(_reader_error)

        try:
            _reader = easyocr.Reader(['ch_sim', 'en'])
            return _reader
        except Exception as exc:
            _reader_error = (
                f"OCR初始化失败: {exc}. 可能是证书链或模型下载失败。"
                "请检查网络、证书配置，或提前下载 easyocr 模型。"
            )
            raise RuntimeError(_reader_error) from exc

# 中国车牌规则: 省份简称 + 城市字母 + 5或6位字母数字
PROVINCE_CHARS = "京津沪渝冀晋辽吉黑苏浙皖闽赣鲁豫鄂湘粤琼川贵云陕甘青蒙桂宁新藏"
CITY_CHAR_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B"
}
PLATE_REGEX = re.compile(rf"^[{PROVINCE_CHARS}][A-Z][A-Z0-9]{{5,6}}$")
OCR_ALLOWLIST = PROVINCE_CHARS + "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
ALNUM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_HSV_RANGES = {
    "blue": (np.array([90, 60, 50]), np.array([140, 255, 255])),
    "green": (np.array([35, 40, 40]), np.array([95, 255, 255]))
}


def log_info(message):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"INFO {now_str} {message}", flush=True)


def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def normalize_ocr_text(text):
    # 保留中文、省份简称、英文字母和数字，去掉空格及其他符号
    cleaned = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", text)
    return cleaned.upper()


def preprocess_plate_image(img):
    resized = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.bilateralFilter(resized, 9, 75, 75)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def pil_decode_image(image_bytes):
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        arr = np.array(im)
        bgr = arr[:, :, ::-1].copy()
        return bgr
    except Exception as e:
        log_info(f"Pillow decode failed: {e}")
        return None


def generate_plate_variants(crop):
    variants = []
    try:
        base = preprocess_plate_image(crop)
        variants.append(("base", base))

        try:
            gaussian = cv2.GaussianBlur(base, (0, 0), 3)
            unsharp = cv2.addWeighted(base, 1.5, gaussian, -0.5, 0)
            variants.append(("sharpen", unsharp))
        except Exception:
            pass

        try:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            variants.append(("clahe_strong", cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)))
        except Exception:
            pass

        try:
            lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            merged = cv2.merge((l, a, b))
            variants.append(("lab_eq", cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)))
        except Exception:
            pass
    except Exception:
        pass

    seen = set()
    unique = []
    for name, v in variants:
        key = v.shape if hasattr(v, 'shape') else None
        if key not in seen:
            seen.add(key)
            unique.append((name, v))
    return unique


def detect_plate_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, w = crop.shape[:2]
    area = max(h * w, 1)

    scores = {}
    for color, (lower, upper) in PLATE_HSV_RANGES.items():
        mask = cv2.inRange(hsv, lower, upper)
        scores[color] = float(np.count_nonzero(mask)) / area

    best_color = max(scores, key=scores.get)
    if scores[best_color] < 0.08:
        return "unknown"
    return best_color


def find_plate_regions(img, max_regions=5):
    h, w = img.shape[:2]
    img_area = h * w
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    candidates = []

    for color_name, (lower, upper) in PLATE_HSV_RANGES.items():
        color_mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            area = cw * ch
            ratio = cw / max(ch, 1)
            if area < img_area * 0.002:
                continue
            if not 2.0 <= ratio <= 6.5:
                continue

            pad_x = int(cw * 0.1)
            pad_y = int(ch * 0.25)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + cw + pad_x)
            y2 = min(h, y + ch + pad_y)

            roi_mask = color_mask[y1:y2, x1:x2]
            color_ratio = float(np.count_nonzero(roi_mask)) / max((y2 - y1) * (x2 - x1), 1)
            score = color_ratio * 0.7 + min(area / img_area, 0.3)
            candidates.append((score, x1, y1, x2, y2, color_name))

    candidates.sort(key=lambda item: item[0], reverse=True)
    regions = []
    for _, x1, y1, x2, y2, color_name in candidates[:max_regions]:
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            regions.append({"image": crop, "color": color_name})

    # 回退: 即使没找到蓝/绿车牌候选，也尝试整图和中间区域
    if not regions:
        regions.append({"image": img, "color": "unknown"})
        cx1, cx2 = int(w * 0.2), int(w * 0.8)
        cy1, cy2 = int(h * 0.45), int(h * 0.85)
        center_crop = img[cy1:cy2, cx1:cx2]
        if center_crop.size > 0:
            regions.append({"image": center_crop, "color": detect_plate_color(center_crop)})

    return regions


def normalize_plate_candidate(candidate, expected_tail_lengths=(5, 6)):
    if len(candidate) < 7:
        return None

    candidate = candidate[0] + candidate[1].upper() + candidate[2:]
    if candidate[0] not in PROVINCE_CHARS:
        return None

    # 城市位必须是字母，若OCR识别成常见数字则纠正
    city = candidate[1]
    if city.isdigit() and city in CITY_CHAR_MAP:
        city = CITY_CHAR_MAP[city]
    candidate = candidate[0] + city + candidate[2:]

    tail_len = len(candidate) - 2
    if tail_len not in expected_tail_lengths:
        return None

    if PLATE_REGEX.match(candidate):
        return candidate
    return None


def extract_plate_from_ocr(results, tail_lengths=(6, 5)):
    if not results:
        return None

    normalized_items = [normalize_ocr_text(item) for item in results if item and normalize_ocr_text(item)]
    if not normalized_items:
        return None

    candidates = []

    # 候选1: 每段识别结果单独尝试
    candidates.extend(normalized_items)

    # 候选2: OCR分段时，尝试拼接整体结果
    merged = "".join(normalized_items)
    if merged:
        candidates.append(merged)

    # 候选3: 常见两段分割(如 川A + AA4444)
    for i in range(len(normalized_items) - 1):
        candidates.append(normalized_items[i] + normalized_items[i + 1])

    for raw in candidates:
        for tail_len in tail_lengths:
            if len(raw) >= tail_len + 2:
                plate_raw = raw[:2 + tail_len]
                plate = normalize_plate_candidate(plate_raw, expected_tail_lengths=tuple(tail_lengths))
                if plate:
                    return plate

    return None


def pick_province_char(text_items):
    merged = "".join(normalize_ocr_text(t) for t in text_items if t)
    for ch in merged:
        if ch in PROVINCE_CHARS:
            return ch
    return None


def pick_city_char(text_items):
    merged = "".join(normalize_ocr_text(t) for t in text_items if t)
    if not merged:
        return None

    ranked = []
    for idx, ch in enumerate(merged):
        if "A" <= ch <= "Z":
            score = 1.0
            if ch in {"I", "O"}:
                score -= 0.2
            ranked.append((score, idx, ch))
            continue

        if ch.isdigit() and ch in CITY_CHAR_MAP:
            mapped = CITY_CHAR_MAP[ch]
            score = 1.1
            if mapped in {"I", "O"}:
                score -= 0.25
            ranked.append((score, idx, mapped))

    if not ranked:
        return None

    # 分数优先；同分时更靠后的字符优先，常能规避左侧噪声
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]
    return None


def pick_tail_text(text_items, tail_lengths=(6, 5)):
    merged = "".join(normalize_ocr_text(t) for t in text_items if t)
    tail = re.sub(r"[^A-Z0-9]", "", merged)
    if not tail_lengths:
        return None
    if len(tail) < min(tail_lengths):
        return None
    for length in tail_lengths:
        if len(tail) >= length:
            return tail[:length]
    return None


def staged_plate_recognition(crop, plate_color="unknown"):
    h, w = crop.shape[:2]
    if h < 10 or w < 10:
        return None

    reader = get_ocr_reader()

    if plate_color == "blue":
        tail_lengths = (5,)
    elif plate_color == "green":
        tail_lengths = (6,)
    else:
        tail_lengths = (6, 5)

    # 尝试多个预处理变体以提高识别率
    variants = generate_plate_variants(crop)
    ph = pw = 0
    for vname, variant_img in variants:
        ph, pw = variant_img.shape[:2]
        try:
            result_allow = reader.readtext(variant_img, detail=0, allowlist=OCR_ALLOWLIST)
            result_open = reader.readtext(variant_img, detail=0)
            log_info(f"Plate-crop OCR ({vname}) allow={result_allow} open={result_open}")
        except Exception as exc:
            log_info(f"OCR on variant {vname} failed: {exc}")
            continue

        plate = extract_plate_from_ocr(result_allow + result_open, tail_lengths=tail_lengths)
        if plate:
            return plate

    # 路径2: 分段OCR(省份/城市/后缀)
    # 注意: 分段必须基于放大后的宽度 pw，而不是原图 w
    part1_end = max(int(pw * 0.20), 1)
    part2_end = max(int(pw * 0.32), part1_end + 1)

    roi_province = variants[0][1][:, :part1_end]
    roi_city = variants[0][1][:, part1_end:part2_end]
    roi_tail = variants[0][1][:, part2_end:]

    # 左侧扩展区域作为省份/城市识别回退，提升 "苏B" 一类识别率
    roi_left_wide = variants[0][1][:, :max(int(pw * 0.40), part2_end)]
    roi_city_wide = variants[0][1][:, part1_end:max(int(pw * 0.38), part2_end + 1)]

    province_raw = reader.readtext(roi_province, detail=0, allowlist=PROVINCE_CHARS)
    province_raw += reader.readtext(roi_left_wide, detail=0, allowlist=PROVINCE_CHARS)
    city_raw = reader.readtext(roi_city, detail=0, allowlist=ALNUM_CHARS)
    city_raw += reader.readtext(roi_city_wide, detail=0, allowlist=ALNUM_CHARS)
    tail_raw = reader.readtext(roi_tail, detail=0, allowlist=ALNUM_CHARS)

    log_info(f"ROI OCR province={province_raw}, city={city_raw}, tail={tail_raw}")

    province = pick_province_char(province_raw)
    city = pick_city_char(city_raw)
    tail = pick_tail_text(tail_raw, tail_lengths=tail_lengths)

    if province and city and tail:
        candidate = f"{province}{city}{tail}"
        normalized = normalize_plate_candidate(candidate, expected_tail_lengths=tail_lengths)
        if normalized:
            return normalized

    return None

# 初始化数据库
def init_db():
    log_info("Initializing database parking.db")
    conn = sqlite3.connect("parking.db")
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS cars(
            plate TEXT,
            enter_time TEXT,
            exit_time TEXT,
            fee REAL,
            status TEXT
        )
    ''')

    conn.commit()
    conn.close()
    log_info("Database initialized")

# 首页
@app.route('/')
def index():
    log_info(f"GET / from {request.remote_addr}")
    return render_template("index.html")

# 上传识别接口
@app.route('/upload', methods=['POST'])
def upload():
    client_ip = request.remote_addr or "unknown"
    log_info(f"POST /upload received from {client_ip}")

    # 支持指定通道: entry / exit / auto(默认)
    channel = (request.form.get('channel') or request.args.get('channel') or request.headers.get('X-Channel') or 'auto').lower()
    if channel not in {'entry', 'exit', 'auto'}:
        log_info(f"Invalid channel param: {channel}")
        return jsonify({"error": "invalid_channel", "msg": "通道参数无效，仅允许: entry, exit, auto"}), 400

    if 'image' not in request.files:
        log_info("Upload rejected: missing 'image' field")
        return jsonify({"error": "no_file", "msg": "没有上传文件"}), 400

    file = request.files['image']

    if not file or not file.filename:
        log_info("Upload rejected: empty file or filename")
        return jsonify({"msg": "没有文件"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        log_info("Upload rejected: invalid filename after secure_filename")
        return jsonify({"error": "invalid_filename", "msg": "文件名无效"}), 400

    # 检查文件扩展名是否被允许
    if not allowed_file(filename):
        log_info(f"Upload rejected: unsupported file extension for {filename}")
        return jsonify({"error": "unsupported_format", "msg": "图片格式不支持，仅支持 png/jpg/jpeg/bmp/gif"}), 415

    # 先在内存里解码，避免无效图片导致 OCR 内部崩溃
    image_bytes = file.read()
    if not image_bytes:
        log_info(f"Upload rejected: empty file content, filename={filename}")
        return jsonify({"error": "empty_file", "msg": "空文件"}), 400

    log_info(f"Upload file accepted: filename={filename}, bytes={len(image_bytes)}")

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        log_info(f"cv2 failed to decode image, trying Pillow for {filename}")
        img = pil_decode_image(image_bytes)
        if img is None or img.size == 0:
            log_info(f"Upload rejected: invalid image decoding, filename={filename}")
            return jsonify({"error": "invalid_image", "msg": "上传的不是有效图片或图片已损坏"}), 400

    log_info(f"Image decoded successfully: shape={img.shape}")

    path = os.path.join(UPLOAD_FOLDER, filename)
    file.stream.seek(0)
    file.save(path)
    log_info(f"Image saved to {path}")

    # OCR识别: 先自动定位车牌区域，再按规则识别
    try:
        regions = find_plate_regions(img)
        log_info(f"Detected {len(regions)} plate region candidate(s)")
    except Exception as exc:
        log_info(f"Plate region detection exception: {exc}")
        return jsonify({"error": "detection_error", "msg": "车牌定位出错，可能是图片过于模糊或格式异常"}), 502

    plate = None
    for idx, region_info in enumerate(regions, start=1):
        region = region_info["image"]
        detected_color = region_info["color"]
        log_info(f"Trying plate region #{idx}, shape={region.shape}, color={detected_color}")
        try:
            plate = staged_plate_recognition(region, plate_color=detected_color)
        except Exception as exc:
            log_info(f"Region #{idx} OCR exception: {exc}")
            return jsonify({"error": "ocr_error", "msg": "OCR识别过程出错，请稍后重试"}), 502
        if plate:
            log_info(f"Plate matched in region #{idx}: {plate}")
            break

    if not plate:
        log_info("OCR text does not match China plate rules or OCR returned no candidate")
        return jsonify({"error": "ocr_no_plate", "msg": "OCR识别失败或车牌不符合规则，请上传更清晰的正向车牌图片"}), 422

    log_info(f"Plate recognized after rule filter: {plate}")

    # 按通道强制入场或出场，'auto' 保持原行为
    now = datetime.now()
    try:
        conn = sqlite3.connect("parking.db", timeout=10)
        c = conn.cursor()
    except Exception as exc:
        log_info(f"DB connection error: {exc}")
        return jsonify({"error": "db_unavailable", "msg": "数据库不可用，请稍后重试"}), 503

    try:
        c.execute("SELECT * FROM cars WHERE plate=? AND status='在场'", (plate,))
        car = c.fetchone()

        # 强制入场通道
        if channel == 'entry':
            if car is not None:
                log_info(f"Duplicate entry attempt: plate={plate}")
                return jsonify({"error": "duplicate_entry", "msg": "重复入场：车辆已在场"}), 409

            c.execute("INSERT INTO cars VALUES(?,?,?,?,?)", (plate, str(now), "", 0, "在场"))
            conn.commit()
            log_info(f"Entry recorded (entry channel): plate={plate}, time={now}")
            return jsonify({"plate": plate, "type": "入场成功", "time": str(now)})

        # 强制出场通道
        if channel == 'exit':
            if car is None:
                log_info(f"Exit-before-entry attempt: plate={plate}")
                return jsonify({"error": "exit_before_entry", "msg": "未入场先出场：找不到在场记录"}), 409

            enter_time = datetime.strptime(car[1], "%Y-%m-%d %H:%M:%S.%f")
            hours = (now - enter_time).total_seconds() / 3600
            fee = round(hours * 5, 2)

            c.execute("""
                UPDATE cars
                SET exit_time=?, fee=?, status='离场'
                WHERE plate=? AND status='在场'
            """, (str(now), fee, plate))
            conn.commit()
            log_info(f"Exit recorded (exit channel): plate={plate}, enter={enter_time}, exit={now}, fee={fee}")
            return jsonify({"plate": plate, "type": "出场成功", "fee": fee, "time": str(now)})

        # 自动模式：保持原有语义（在场->出场，否则入场）
        if channel == 'auto':
            if car is None:
                c.execute("INSERT INTO cars VALUES(?,?,?,?,?)", (plate, str(now), "", 0, "在场"))
                conn.commit()
                log_info(f"Entry recorded (auto): plate={plate}, time={now}")
                return jsonify({"plate": plate, "type": "入场成功", "time": str(now)})

            else:
                enter_time = datetime.strptime(car[1], "%Y-%m-%d %H:%M:%S.%f")
                hours = (now - enter_time).total_seconds() / 3600
                fee = round(hours * 5, 2)

                c.execute("""
                    UPDATE cars
                    SET exit_time=?, fee=?, status='离场'
                    WHERE plate=? AND status='在场'
                """, (str(now), fee, plate))
                conn.commit()
                log_info(f"Exit recorded (auto): plate={plate}, enter={enter_time}, exit={now}, fee={fee}")
                return jsonify({"plate": plate, "type": "出场成功", "fee": fee, "time": str(now)})

    except sqlite3.OperationalError as exc:
        log_info(f"SQLite operational error: {exc}")
        return jsonify({"error": "db_error", "msg": "数据库操作失败，请稍后重试"}), 503
    except Exception as exc:
        log_info(f"Unhandled processing error: {exc}")
        return jsonify({"error": "internal_error", "msg": "处理请求时发生未预期错误"}), 500
    finally:
        try:
            conn.close()
        except Exception:
            pass

if __name__ == '__main__':
    init_db()
    log_info("Server starting at 0.0.0.0:1145")
    app.run(host='0.0.0.0', port=1145)

