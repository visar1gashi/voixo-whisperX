import os, io, tempfile
import json
import yt_dlp
import torch
import runpod

# Optional: relax PyTorch safe-load for older checkpoints that use pickle globals
try:
    import torch.serialization as ts
    try:
        from omegaconf.listconfig import ListConfig as OmegaListConfig  # type: ignore
        ts.add_safe_globals([OmegaListConfig])  # allow-list for some templates
    except Exception:
        pass
    _orig_load = torch.load
    def _load_compat(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False  # fallback for templates expecting pickle
        return _orig_load(*args, **kwargs)
    torch.load = _load_compat  # type: ignore
except Exception:
    pass

import whisperx


def _get_video_url(inp: dict) -> str:
    for k in ("videoUrl", "url", "audio_url", "youtube_url"):
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _get_language(inp: dict, fallback=None):
    lang = inp.get("language") or inp.get("lang")
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return fallback


def _dl_audio_to_wav(video_url: str, out_dir: str) -> str:
    # Download best audio and convert to wav via ffmpeg
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        base = info.get("id")
        for fn in os.listdir(out_dir):
            if fn.startswith(str(base)) and fn.lower().endswith(".wav"):
                return os.path.join(out_dir, fn)
    raise RuntimeError("Failed to download/convert audio")


def _simplify_segments(segments):
    out = []
    for s in segments or []:
        try:
            out.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": str(s.get("text") or "").strip()
            })
        except Exception:
            pass
    return out


def _flatten_words(segments):
    words = []
    for s in segments or []:
        for w in (s.get("words") or []):
            words.append({
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "word": str(w.get("word") or w.get("text") or "")
            })
    return words


def handler(event):
    """
    Input (event['input']):
      {
        "videoUrl" | "url" | "audio_url": "https://www.youtube.com/watch?v=...",
        "language": "en"  # optional
      }
    Returns:
      { "segments": [{start,end,text}], "words": [{start,end,word}], "language": str }
    """
    inp = event.get("input") or {}
    video_url = _get_video_url(inp)
    if not video_url:
        return {"error": "videoUrl required"}

    lang = _get_language(inp, None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    with tempfile.TemporaryDirectory() as td:
        # 1) Download audio to WAV
        wav_path = _dl_audio_to_wav(video_url, td)

        # 2) Load WhisperX model (no VAD) and transcribe
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        audio = whisperx.load_audio(wav_path)
        result = model.transcribe(audio, batch_size=16, language=lang)

        # 3) Alignment (word-level)
        use_lang = lang or result.get("language") or "en"
        model_a, metadata = whisperx.load_align_model(language_code=use_lang, device=device)
        aligned = whisperx.align(result["segments"], model_a, metadata, audio, device,
                                 return_char_alignments=False)

        segs_simple = _simplify_segments(aligned.get("segments") or result.get("segments") or [])
        words = _flatten_words(aligned.get("segments") or [])

        return {"segments": segs_simple, "words": words, "language": use_lang}


runpod.serverless.start({"handler": handler})
