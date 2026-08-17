# Tài liệu ghép Raspberry Pi — WebSocket realtime

Gửi team Pi. Client **chỉ cần WebSocket**. Server nhận camera + mic liên tục, trả text hội thoại đã xử lý (tách giọng người đối diện + ASR tiếng Việt).

Không dùng REST upload từng đoạn cho luồng live.

Có 2 việc:
1. **Chạy server + model ở máy dev** (Mac/PC) — mục 1
2. **Ghép WebSocket trên Pi** — mục 2 trở đi

---

## 1. Chạy server + model ở local (máy dev)

Làm trên **Mac / Linux x86_64 hoặc Apple Silicon**. Không khuyến nghị chạy model nặng trên chính Raspberry Pi.

### 1.1. Cần có sẵn

- Python **3.12** (không dùng 3.14)
- `ffmpeg` (`brew install ffmpeg` hoặc `sudo apt install ffmpeg`)
- RAM ≥ 8 GB, disk trống ≥ 8 GB (PhoWhisper + AV-TSE)
- Camera + mic nếu test client trên cùng máy

### 1.2. Clone và tạo venv

```bash
git clone <REPO_URL> detect_giong_noi
cd detect_giong_noi

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
```

### 1.3. Cài dependency + PyTorch

**Mac (Apple Silicon / MPS):**

```bash
pip install torch torchaudio
pip install -r requirements.txt
```

**Linux + NVIDIA GPU:**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**Linux CPU-only (chậm hơn):**

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 1.4. Tải model

Lần đầu sẽ tải từ Hugging Face (cần mạng).

```bash
export PYTHONPATH=.
python download_model.py
```

Tải:
- ASR: `vinai/PhoWhisper-small` (cache Hugging Face)
- AV-TSE: `AV_MossFormer2_TSE_16K` tự tải vào `checkpoints/` khi server start lần đầu
- Face: nếu thiếu `models/face_landmarker.task`:

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

### 1.5. Bật server

```bash
source .venv/bin/activate
export PYTHONPATH=.
./run.sh
```

Hoặc:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` để Pi trong LAN gọi được.

Đợi ~10–30 giây. Kiểm tra:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/pi/info
```

Phải thấy `"ready": true`. Nếu `"ready": false` thì đọc `"stage"` (đang load model).

Web demo (không bắt buộc): mở http://127.0.0.1:8000

### 1.6. Chạy client mẫu trên cùng máy (test socket)

Terminal khác:

```bash
source .venv/bin/activate
pip install -r clients/requirements-pi.txt
python clients/pi_client.py --url http://127.0.0.1:8000 --device-id local-dev
```

Nói tiếng Việt trước camera. Terminal in `>> câu mới` và block `hội thoại`. Im lặng thì không ra chữ (chỉ `silence`).

### 1.7. Cho Pi trong LAN gọi máy dev

1. Máy dev và Pi cùng Wi-Fi
2. Lấy IP máy dev: `ipconfig getifaddr en0` (Mac) hoặc `hostname -I`
3. Tắt firewall chặn port **8000** (hoặc mở inbound 8000)
4. Trên Pi:

```bash
pip install -r clients/requirements-pi.txt
python clients/pi_client.py --url http://IP_MAY_DEV:8000 --device-id pi-01
```

WebSocket: `ws://IP_MAY_DEV:8000/api/pi/ws?device_id=pi-01`

### 1.8. (Tuỳ chọn) Public HTTPS bằng Cloudflare Tunnel

Cam/mic trên trình duyệt cần HTTPS. Pi dùng `ws://` LAN thì **không cần** tunnel.

```bash
brew install cloudflared
cloudflared tunnel --protocol http2 --url http://127.0.0.1:8000
```

Lấy URL `https://xxxx.trycloudflare.com` — Pi dùng:

```
wss://xxxx.trycloudflare.com/api/pi/ws?device_id=pi-01
```

### 1.9. Lỗi thường gặp khi run local

| Hiện tượng | Cách xử |
|---|---|
| `/health` 404 hoặc app khác | Port 8000 bị chiếm — đổi `--port 8001` |
| `ready: false` mãi | Đợi load; xem log terminal server |
| OOM / killed | Máy thiếu RAM — đóng app khác |
| Không mở cam/mic client | Cấp quyền OS; không dùng IDE Simple Browser |
| Pi không connect | Sai IP, server bind `127.0.0.1` thay vì `0.0.0.0`, hoặc firewall |
| Chữ nhảy lúc im | Đã lọc phía server; restart server bản mới |

---

## 2. Kết nối

```
WS  {HOST}/api/pi/ws?device_id={ID}&session_id={SID}&token={TOKEN}
```

| Query | Bắt buộc | Ý nghĩa |
|---|---|---|
| `device_id` | Không | Tên máy, vd. `pi-01` |
| `session_id` | Không | Lần đầu bỏ trống. Lần sau gửi lại để **giữ hội thoại** khi reconnect |
| `token` | Chỉ khi server bật `PI_API_TOKEN` | Auth |

Ví dụ:

```
ws://192.168.1.10:8000/api/pi/ws?device_id=pi-01
wss://your-tunnel.trycloudflare.com/api/pi/ws?device_id=pi-01&session_id=a1b2c3d4e5f6
```

- Local / LAN: `ws://`
- Cloudflare / HTTPS: `wss://`

Sau khi connect, **đợi JSON `type=ready`** rồi mới stream.

```json
{
  "type": "ready",
  "session_id": "a1b2c3d4e5f6",
  "sample_rate": 16000,
  "protocol": "websocket"
}
```

**Lưu `session_id`.** Mất mạng thì mở lại socket với cùng `session_id`.

---

## 3. Pi → Server (binary)

Mỗi message binary: **1 byte tag + payload**.

| Tag | Hex | Payload | Ghi chú |
|---|---|---|---|
| JPEG | `0x01` | 1 frame JPEG | ~5–8 fps, 640×480, quality ~70 |
| PCM | `0x02` | audio thô | **PCM16 LE, mono, 16 kHz** |

### Audio (bắt buộc đúng format)

- 1 kênh (mono)
- 16_000 Hz
- `int16` little-endian
- Gửi liên tục, block ~2048–4096 sample (~128–256 ms)

Ví dụ Python:

```python
ws.send(b"\x02" + pcm16_bytes, opcode=2)  # binary frame
```

### Video

- JPEG, mặt người nói càng rõ càng tốt (AV-TSE dùng face/lip)
- **~12 fps** để bắt môi đang nói (6 fps dễ nhầm jitter với nói)
- Gửi binary:

```python
ws.send(b"\x01" + jpeg_bytes, opcode=2)
```

**Không** gửi MP4 / H264 trên socket này.

---

## 4. Pi → Server (JSON text)

| Message | Khi nào |
|---|---|
| `{"type":"flush"}` | Chốt câu (dừng nói / hết lượt) |
| `{"type":"config","require_speaking":true}` | Chỉ nhận khi môi cử động (mặc định `true`) |
| `{"type":"get_conversation"}` | Xin lại toàn bộ hội thoại |
| `{"type":"pong"}` | Trả lời `ping` của server |

---

## 5. Server → Pi (JSON text)

Chỉ quan tâm `transcript` để hiện chữ. Các type khác để debug / giữ kết nối.

### `ready`

Socket OK, bắt đầu gửi cam/mic.

### `transcript` — **text hội thoại mới**

```json
{
  "type": "transcript",
  "text": "cho mình xem menu",
  "final": false,
  "used_tse": true,
  "session_id": "a1b2c3d4e5f6",
  "conversation": "xin chào quý khách\ncho mình xem menu",
  "turns": [
    {"index": 1, "ts": 1710000000.1, "text": "xin chào quý khách", "used_tse": true},
    {"index": 2, "ts": 1710000003.4, "text": "cho mình xem menu", "used_tse": true}
  ]
}
```

| Field | Dùng làm gì |
|---|---|
| `text` | Câu vừa nhận (một lượt) |
| `conversation` | **Toàn bộ hội thoại**, mỗi câu một dòng — ưu tiên hiển thị cái này |
| `turns` | Mảng câu có timestamp |
| `final` | `false` = chữ tạm (đang nói, **thay dòng hiện tại**). `true` = chốt câu (ngừng nói / `flush` / đủ ~6s) |

Server chạy **live cửa sổ trượt**: partial ~0.6–1s/lần (ASR nhanh, chưa TSE), `final` mới chạy AV-TSE. Chỉ `append` hội thoại khi `final=true`.

Im lặng **không** gửi `transcript` mới (trừ khi chốt câu đã có chữ tạm).

### `status`

```json
{"type":"status","text":"silence"}
```

`silence` = không nói / tạp âm, **không append chữ**. Có thể bỏ qua trên UI.

### `face` (debug)

```json
{"type":"face","found":true,"speaking":true,"x":0.2,"y":0.1,"w":0.4,"h":0.5}
```

### `ping`

Server gửi ~15 giây/lần. Pi trả:

```json
{"type":"pong"}
```

### `error`

```json
{"type":"error","text":"Invalid or missing token"}
```

Đóng socket / hiện lỗi.

### `conversation`

Trả lời `get_conversation`: cùng shape session (`session_id`, `turns`, `conversation`).

---

## 6. Luồng khuyến nghị trên Pi

```
1. Open WS
2. Đợi type=ready, lưu session_id
3. Thread mic  → gửi 0x02 liên tục
4. Thread cam  → gửi 0x01 ~6 fps
5. Thread recv → nếu type=transcript: cập nhật UI bằng field conversation
6. Hết câu / user ngừng nói ~0.6s → gửi {"type":"flush"}
7. Mất WS → sleep 2s → connect lại kèm session_id
```

Pseudo:

```
on_message(json):
  if type == "ready":
      save(session_id)
  if type == "transcript":
      ui.set_text(conversation)   # full hội thoại
      ui.append_last(text)        # câu mới
  if type == "ping":
      send({"type":"pong"})
  if type == "error":
      reconnect()
```

---

## 7. Ràng buộc / lưu ý

1. **Sample rate phải 16 kHz.** Sai rate → chữ sai hoặc không ra chữ.
2. Camera nhìn **mặt + miệng** người đối diện. Nhiều người nói chồng: server tách theo mặt trong frame.
3. Im lặng đã được server lọc. Đừng tự invent text phía Pi.
4. `require_speaking=true`: có mặt mà môi không cử động thì audio bị bỏ. Test trong tối / không thấy mặt thì gửi `{"type":"config","require_speaking":false}`.
5. Mỗi ~3 giây server mới chạy 1 lần ASR nếu đang nói. Không expect chữ từng từ realtime tuyệt đối.
6. Giữ **một** WebSocket. Đừng mở nhiều socket cùng lúc từ một Pi.

---

## 8. Client mẫu (repo)

```bash
pip install -r clients/requirements-pi.txt
python clients/pi_client.py --url http://IP_SERVER:8000 --device-id pi-01
```

Code mẫu: `clients/pi_client.py`.

---

## 9. Kiểm tra nhanh

1. Connect → nhận `ready` + `session_id`
2. Không nói 5–10 giây → chỉ thấy `silence` hoặc không có `transcript`
3. Nói tiếng Việt, mặt trong camera → nhận `transcript` + `conversation` dài dần
4. Ngắt mạng, connect lại `?session_id=...` → hội thoại cũ còn

Hỏi server còn sống:

```
GET {HOST}/api/pi/info
GET {HOST}/health
```

`ready: true` mới stream.
