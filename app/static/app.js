(() => {
  const video = document.getElementById("video");
  const overlay = document.getElementById("overlay");
  const ctx = overlay.getContext("2d");
  const startBtn = document.getElementById("startBtn");
  const listenBtn = document.getElementById("listenBtn");
  const stopBtn = document.getElementById("stopBtn");
  const flushBtn = document.getElementById("flushBtn");
  const meta = document.getElementById("meta");
  const banner = document.getElementById("banner");
  const transcriptEl = document.getElementById("transcript");
  const faceStatus = document.getElementById("faceStatus");
  const speakStatus = document.getElementById("speakStatus");
  const tseStatus = document.getElementById("tseStatus");
  const levelBar = document.getElementById("levelBar");
  const requireSpeaking = document.getElementById("requireSpeaking");
  const placeholder = document.getElementById("videoPlaceholder");

  let ws = null;
  let stream = null;
  let audioCtx = null;
  let processor = null;
  let analyser = null;
  let source = null;
  let mute = null;
  let rafId = null;
  let frameTimer = null;
  let face = null;
  let lines = [];
  let mediaOpen = false;
  let listening = false;

  const TARGET_SR = 16000;

  function setMeta(text) {
    meta.textContent = text;
  }

  function setBanner(text, kind = "") {
    banner.innerHTML = text;
    banner.className = "banner" + (kind ? " " + kind : "");
  }

  function pushTranscript(text, final, usedTse, speaker) {
    if (!text || !String(text).trim()) return;
    const who = speaker ? speaker + ": " : "";
    const body = String(text).trim();
    const shown = who + body;
    const pending = "… " + who;
    if (!final) {
      const i = lines.findIndex((line) => line.startsWith(pending));
      if (i >= 0) lines[i] = pending + body;
      else lines.push(pending + body);
    } else {
      const i = lines.findIndex((line) => line.startsWith(pending));
      if (i >= 0) lines[i] = shown;
      else lines.push(shown);
    }
    transcriptEl.textContent = lines.join("\n");
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
    setBanner((final ? "" : "… ") + "<strong>" + shown.replace(/</g, "") + "</strong>", "ok");
  }

  function resizeOverlay() {
    const w = video.clientWidth || overlay.clientWidth;
    const h = video.clientHeight || overlay.clientHeight;
    if (!w || !h) return;
    overlay.width = Math.floor(w * devicePixelRatio);
    overlay.height = Math.floor(h * devicePixelRatio);
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, w, h);
  }

  function drawFace() {
    const w = video.clientWidth;
    const h = video.clientHeight;
    ctx.clearRect(0, 0, w, h);
    const people = face && face.people && face.people.length
      ? face.people
      : face && face.found ? [face] : [];
    if (!w || !people.length) return;
    const palette = { A: "#1f8f4e", B: "#2f6fed" };
    for (const person of people) {
      const color = palette[person.speaker] || "#f0c14b";
      const x = person.x * w;
      const y = person.y * h;
      const bw = person.w * w;
      const bh = person.h * h;
      ctx.strokeStyle = person.speaking ? color : "#f0c14b";
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, bw, bh);
      ctx.fillStyle = person.speaking ? color + "22" : "rgba(240,193,75,0.12)";
      ctx.fillRect(x, y, bw, bh);
      if (person.speaker) {
        ctx.fillStyle = color;
        ctx.font = "700 16px sans-serif";
        ctx.fillText(person.speaker, x + 6, Math.max(18, y - 6));
      }
    }
  }

  function downsampleTo16k(float32, inputRate) {
    if (Math.abs(inputRate - TARGET_SR) < 1) return float32;
    const ratio = inputRate / TARGET_SR;
    const newLen = Math.floor(float32.length / ratio);
    const out = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const start = i * ratio;
      const end = Math.min(float32.length, (i + 1) * ratio);
      const i0 = Math.floor(start);
      const i1 = Math.max(i0 + 1, Math.min(float32.length, Math.ceil(end)));
      let sum = 0;
      for (let j = i0; j < i1; j++) sum += float32[j];
      out[i] = sum / (i1 - i0);
    }
    return out;
  }

  function floatTo16BitPCM(float32) {
    const buf = new ArrayBuffer(float32.length * 2);
    const view = new DataView(buf);
    for (let i = 0; i < float32.length; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Uint8Array(buf);
  }

  function sendBinary(tag, payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const out = new Uint8Array(1 + payload.byteLength);
    out[0] = tag;
    out.set(payload, 1);
    ws.send(out);
  }

  async function sendVideoFrame() {
    if (!listening || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    const scale = 400 / Math.max(video.videoWidth, video.videoHeight);
    canvas.width = Math.max(2, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(2, Math.round(video.videoHeight * scale));
    const c = canvas.getContext("2d", { willReadFrequently: false });
    c.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.62));
    if (!blob) return;
    sendBinary(1, new Uint8Array(await blob.arrayBuffer()));
  }

  function connectWs() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return Promise.resolve();
    }
    if (ws) {
      try { ws.close(); } catch (_) {}
      ws = null;
    }
    return new Promise((resolve, reject) => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      ws = new WebSocket(`${proto}://${location.host}/ws`);
      ws.binaryType = "arraybuffer";
      const t = setTimeout(() => reject(new Error("WebSocket timeout")), 15000);

      ws.onopen = () => {
        clearTimeout(t);
        ws.send(
          JSON.stringify({
            type: "config",
            require_speaking: requireSpeaking.checked,
          })
        );
        resolve();
      };

      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch (_) {
          return;
        }
        if (msg.type === "ready") {
          setMeta(`${msg.av_tse_model} + ${msg.asr_model} · ${msg.device} · live`);
        } else if (msg.type === "face") {
          face = msg;
          const people = msg.people || [];
          const labels = people.map((p) => p.speaker).filter(Boolean);
          const talking = people.filter((p) => p.speaking).map((p) => p.speaker);
          if (!msg.found && !people.length) {
            faceStatus.textContent = "Chưa thấy mặt";
          } else if (labels.length) {
            faceStatus.textContent = "Trong khung: " + labels.join(", ");
          } else if (msg.frontal === false) {
            faceStatus.textContent = "Mặt lệch — hãy nhìn camera";
          } else {
            faceStatus.textContent = "Đã thấy mặt";
          }
          const voices = msg.voices || [];
          if (talking.length) {
            speakStatus.textContent = "Đang nói: " + talking.join(", ");
          } else if (voices.length >= 2) {
            speakStatus.textContent = "Đã tách giọng " + voices.join(" / ");
          } else if (voices.length === 1) {
            speakStatus.textContent = "Đã nhớ giọng " + voices[0];
          } else if (msg.lip_active) {
            speakStatus.textContent = "Môi nhúc";
          } else {
            speakStatus.textContent = "Không nói";
          }
          speakStatus.classList.toggle("hot", !!msg.speaking);
          speakStatus.classList.toggle("muted", !msg.speaking);
          drawFace();
        } else if (msg.type === "status") {
          tseStatus.textContent = msg.text || "AV-TSE";
          tseStatus.classList.toggle("hot", /tách|TSE|load/i.test(msg.text || ""));
        } else if (msg.type === "transcript") {
          pushTranscript(msg.text, !!msg.final, !!msg.used_tse, msg.speaker || "");
          if (msg.used_tse) {
            tseStatus.textContent = "AV-TSE OK";
            tseStatus.classList.add("hot");
          }
        } else if (msg.type === "error") {
          setBanner("Server: " + msg.text, "error");
        }
      };

      ws.onerror = () => {
        clearTimeout(t);
        reject(new Error("WebSocket error"));
      };
      ws.onclose = () => {
        if (listening) setBanner("Mất kết nối WebSocket. Bấm mở lại.", "error");
      };
    });
  }

  function startLevelMeter() {
    if (!analyser) return;
    const data = new Uint8Array(analyser.fftSize);
    const tick = () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (let i = 0; i < data.length; i++) {
        peak = Math.max(peak, Math.abs(data[i] - 128));
      }
      levelBar.style.width = `${Math.min(100, (peak / 128) * 160)}%`;
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  async function waitForVideoFrames(timeoutMs = 2500) {
    const start = performance.now();
    while (performance.now() - start < timeoutMs) {
      if (video.videoWidth > 0 && video.readyState >= 2) return true;
      await new Promise((r) => setTimeout(r, 100));
    }
    return video.videoWidth > 0;
  }

  async function getMediaStream() {
    // Try full constraints, then relax, then split audio/video
    const attempts = [
      {
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: false,
          channelCount: 1,
        },
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      },
      { audio: true, video: { facingMode: "user" } },
      { audio: true, video: true },
    ];

    let lastErr = null;
    for (const constraints of attempts) {
      try {
        return await navigator.mediaDevices.getUserMedia(constraints);
      } catch (err) {
        lastErr = err;
      }
    }

    // Last resort: open separately
    try {
      const v = await navigator.mediaDevices.getUserMedia({ video: true });
      try {
        const a = await navigator.mediaDevices.getUserMedia({ audio: true });
        a.getAudioTracks().forEach((t) => v.addTrack(t));
      } catch (_) {
        /* video-only fallback */
      }
      return v;
    } catch (err) {
      throw lastErr || err;
    }
  }

  async function openMedia() {
    if (mediaOpen && stream && stream.active) return stream;

    if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      throw new Error("Camera/mic cần http://127.0.0.1 hoặc https");
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("Trình duyệt không hỗ trợ camera/mic. Hãy mở bằng Chrome/Safari (không dùng Simple Browser).");
    }

    setBanner("Đang xin quyền Camera + Microphone…");
    stream = await getMediaStream();

    const vTracks = stream.getVideoTracks();
    const aTracks = stream.getAudioTracks();
    if (!vTracks.length && !aTracks.length) {
      throw new Error("Không nhận được track camera/mic nào");
    }

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("playsinline", "true");
    video.setAttribute("webkit-playsinline", "true");
    video.style.display = "block";
    video.style.opacity = "1";
    placeholder.style.display = "none";

    try {
      await video.play();
    } catch (err) {
      console.warn("video.play failed", err);
    }

    const okFrames = await waitForVideoFrames(3000);
    resizeOverlay();
    window.addEventListener("resize", resizeOverlay);

    // Audio graph: analyser for meter + script processor for streaming PCM
    if (aTracks.length) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      if (audioCtx.state === "suspended") await audioCtx.resume();
      source = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      processor = audioCtx.createScriptProcessor(4096, 1, 1);
      mute = audioCtx.createGain();
      mute.gain.value = 0;

      processor.onaudioprocess = (e) => {
        if (!listening) return;
        const input = e.inputBuffer.getChannelData(0);
        const down = downsampleTo16k(input, audioCtx.sampleRate);
        sendBinary(2, floatTo16BitPCM(down));
      };

      source.connect(analyser);
      source.connect(processor);
      processor.connect(mute);
      mute.connect(audioCtx.destination);
      startLevelMeter();
    }

    mediaOpen = true;
    startBtn.disabled = true;
    listenBtn.disabled = false;
    stopBtn.disabled = false;

    const vLabel = vTracks[0] ? vTracks[0].label || "camera" : "no-cam";
    const aLabel = aTracks[0] ? aTracks[0].label || "mic" : "no-mic";
    if (!okFrames && vTracks.length) {
      setBanner(
        "Đã cấp quyền nhưng <strong>không có khung hình camera</strong>. " +
          "Hãy mở bằng <strong>Chrome/Safari thật</strong> tại <code>http://127.0.0.1:8000</code> " +
          "(Simple Browser/IDE preview thường không chạy được webcam).<br/>" +
          `Tracks: video=${vLabel} · audio=${aLabel}`,
        "error"
      );
    } else {
      setBanner(
        `Camera/mic OK (${video.videoWidth || 0}×${video.videoHeight || 0}). ` +
          `video=<code>${vLabel}</code> · mic=<code>${aLabel}</code>. ` +
          `Bấm <strong>Bắt đầu nghe</strong> (hoặc đợi tự chạy).`,
        "ok"
      );
    }
    return stream;
  }

  async function startListening() {
    if (listening) return;
    await openMedia();
    if (audioCtx && audioCtx.state === "suspended") await audioCtx.resume();
    setBanner("Đang kết nối WebSocket…");
    await connectWs();
    if (!lines.length) transcriptEl.textContent = "(nói đi, chữ sẽ hiện ở đây)";
    listening = true;
    listenBtn.disabled = true;
    flushBtn.disabled = false;
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = setInterval(sendVideoFrame, 40);
    setBanner("Đang nghe người đối diện → Zipformer tiếng Việt.", "ok");
  }

  function stopAll() {
    listening = false;
    mediaOpen = false;
    if (frameTimer) clearInterval(frameTimer);
    frameTimer = null;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    try { if (processor) processor.disconnect(); } catch (_) {}
    try { if (analyser) analyser.disconnect(); } catch (_) {}
    try { if (source) source.disconnect(); } catch (_) {}
    try { if (mute) mute.disconnect(); } catch (_) {}
    try { if (audioCtx) audioCtx.close(); } catch (_) {}
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (ws) try { ws.close(); } catch (_) {}
    processor = analyser = source = mute = audioCtx = stream = ws = null;
    video.srcObject = null;
    placeholder.style.display = "grid";
    placeholder.textContent = "Camera chưa mở";
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    startBtn.disabled = !window.__modelsReady;
    listenBtn.disabled = true;
    stopBtn.disabled = true;
    flushBtn.disabled = true;
    setMeta(window.__modelsReady ? "Đã dừng" : "Đang load model…");
    setBanner("Đã dừng. Bấm <strong>Mở camera & mic</strong> để chạy lại.");
    faceStatus.textContent = "Chưa thấy mặt";
    speakStatus.textContent = "Miệng im";
    speakStatus.classList.add("muted");
    speakStatus.classList.remove("hot");
    tseStatus.textContent = "AV-TSE idle";
    tseStatus.classList.remove("hot");
    levelBar.style.width = "0%";
  }

  startBtn.addEventListener("click", () => {
    openMedia()
      .then(() => {
        // Auto start listening after successful media open
        return startListening();
      })
      .catch((err) => {
        console.error(err);
        setBanner(
          "Không mở được camera/mic: <code>" +
            (err && err.name ? err.name + ": " : "") +
            (err && err.message ? err.message : err) +
            "</code><br/>Mở bằng Chrome/Safari: <code>http://127.0.0.1:8000</code> và Allow Camera + Microphone.",
          "error"
        );
      });
  });

  listenBtn.addEventListener("click", () => {
    startListening().catch((err) => {
      console.error(err);
      setBanner("Không bắt đầu nghe được: " + (err.message || err), "error");
    });
  });

  stopBtn.addEventListener("click", stopAll);
  flushBtn.addEventListener("click", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "flush" }));
    }
  });
  requireSpeaking.addEventListener("change", () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "config",
          require_speaking: requireSpeaking.checked,
        })
      );
    }
  });

  if (window.__modelsReady) {
    startBtn.disabled = false;
    const pbtn = document.getElementById("processVideoBtn");
    if (pbtn) pbtn.disabled = false;
  }

  // ---- MP4 upload demo ----
  const videoFile = document.getElementById("videoFile");
  const processVideoBtn = document.getElementById("processVideoBtn");
  const tseAudio = document.getElementById("tseAudio");
  let selectedFile = null;

  if (videoFile) {
    videoFile.addEventListener("change", () => {
      selectedFile = videoFile.files && videoFile.files[0] ? videoFile.files[0] : null;
      if (!selectedFile) return;
      const url = URL.createObjectURL(selectedFile);
      video.srcObject = null;
      video.muted = false;
      video.controls = true;
      video.src = url;
      video.style.display = "block";
      placeholder.style.display = "none";
      video.play().catch(() => {});
      setBanner("Đã chọn <code>" + selectedFile.name + "</code>. Bấm <strong>Chạy AV-TSE + ASR</strong>.", "ok");
    });
  }

  if (processVideoBtn) {
    processVideoBtn.addEventListener("click", async () => {
      if (!selectedFile) {
        setBanner("Chưa chọn file MP4.", "error");
        return;
      }
      if (!window.__modelsReady) {
        setBanner("Model chưa sẵn sàng.", "error");
        return;
      }
      processVideoBtn.disabled = true;
      setBanner("Đang xử lý video (face → AV-TSE → PhoWhisper)… có thể mất vài chục giây.");
      tseStatus.textContent = "Processing MP4…";
      tseStatus.classList.add("hot");
      transcriptEl.textContent = "Đang chạy pipeline…";

      try {
        const fd = new FormData();
        fd.append("file", selectedFile, selectedFile.name);
        const resp = await fetch("/api/process-video", { method: "POST", body: fd });
        const data = await resp.json().catch(() => ({}));
        if (!resp.ok) {
          throw new Error(data.detail || ("HTTP " + resp.status));
        }
        transcriptEl.textContent = data.text || "(empty)";
        faceStatus.textContent = data.face_frames > 0 ? ("Face frames: " + data.face_frames) : "Không thấy mặt";
        tseStatus.textContent = data.used_tse ? "AV-TSE OK" : "ASR raw (no TSE)";
        tseStatus.classList.toggle("hot", !!data.used_tse);
        if (data.audio_url) {
          tseAudio.style.display = "block";
          tseAudio.src = data.audio_url + "?t=" + Date.now();
        }
        setBanner(
          "Xong · " +
            (data.duration_sec || "?") +
            "s · face_frames=" +
            (data.face_frames || 0) +
            " · TSE=" +
            (data.used_tse ? "yes" : "no"),
          "ok"
        );
      } catch (err) {
        console.error(err);
        setBanner("Lỗi xử lý video: <code>" + (err.message || err) + "</code>", "error");
        transcriptEl.textContent = "Lỗi: " + (err.message || err);
      } finally {
        processVideoBtn.disabled = !window.__modelsReady;
        tseStatus.classList.remove("hot");
      }
    });
  }
})();
