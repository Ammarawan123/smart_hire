"""
Real-time speech-to-text using RealtimeSTT (faster-whisper + VAD), CPU-only.

CHANGES IN THIS VERSION
  1. VAD loosened so speech is actually picked up while the user is talking
     (quiet starts, short answers like "yes"/"no", speech right after a cut).
     Main causes of missed voice before:
       - silero_sensitivity=0.3 / webrtc_sensitivity=3 were too strict
       - min_length_of_recording=0.9 silently dropped short answers
       - RealtimeSTT's default min_gap_between_recordings=1.0 swallowed
         speech that started within 1s after a chunk was cut
       - pre_recording_buffer_duration=0.6 clipped the first word
  2. Pause stitching fixed. A pause shorter than NEW_ANSWER_GAP_SECONDS
     (3s, 5s, ... up to 10s) stays in the SAME transcript line. The gap is
     now tracked PER CHUNK. Before, a single global value was read after the
     (slow) transcription finished, so it often belonged to a different chunk.
  3. Removed: _LEAK_PHRASES / strip_prompt_leak and HALLUCINATION_FILLERS /
     is_hallucinated_filler, as requested.
  4. The WAV is written by a background thread, so it is never blocked while
     the main loop waits on the model.
  5. Latency print + summary kept (total, model-only, RTF).

FIXES AFTER FIRST REAL RUN
  6. CRASH FIX: "No clip timestamps found" killed the whole session after a
     few chunks. Now (a) batch_size=0 avoids faster-whisper's batched
     pipeline, which raises that error when it finds no speech in a chunk,
     (b) faster_whisper_vad_filter is back ON (it was a mistake to turn it
     off: it also strips silence/noise, which reduces hallucination), and
     (c) one failed chunk is logged and skipped instead of ending the run.
  7. ENERGY GATE: chunks whose audio is too quiet (rms < MIN_CHUNK_RMS) are
     not sent to the model at all. Saves ~15s of CPU per junk chunk and
     removes noise-triggered garbage text. RMS is printed per chunk so the
     threshold can be calibrated.
  8. Grammar pass fixed for new `transformers` versions (the old
     pipeline("text2text-generation") task no longer exists there).
"""

import os
import re
import json
import wave
import queue
import time
import threading
import datetime
import inspect
import numpy as np
import sounddevice as sd
from rapidfuzz import fuzz
from RealtimeSTT import AudioToTextRecorder


def _local_model(name):
    try:
        import config as _r
        return _r.local_or({'distil-large-v3': _r.WHISPER_DISTIL_DIR, 'base.en': _r.WHISPER_BASE_DIR}.get(name, _r.WHISPER_BASE_DIR), name)
    except Exception:
        return name


def _local_grammar(name):
    try:
        import config as _r
        return _r.local_or(_r.GRAMMAR_T5_DIR, name)
    except Exception:
        return name

# ============================================================
# TUNABLE KNOBS
# ============================================================
MODEL_NAME = "distil-large-v3"
BEAM_SIZE = 3

ENABLE_LIVE_PREVIEW = False
REALTIME_PROCESSING_PAUSE = 0.7  # only used if ENABLE_LIVE_PREVIEW is True

# --- Voice detection (VAD) ---
# Silence needed AFTER speech before a chunk is cut. A bit longer than before
# so normal thinking pauses don't cut a sentence in half. Each cut costs a
# full model pass, so fewer cuts = less total CPU work too.
POST_SPEECH_SILENCE = 1.0

# Higher = MORE sensitive (picks up quieter voice). 0.3 was too strict.
SILERO_SENSITIVITY = 0.5

# 0-3. 3 = most aggressive at rejecting non-speech (can clip quiet speech).
WEBRTC_SENSITIVITY = 2

# Audio kept from BEFORE speech is detected, so the first word isn't cut.
PRE_RECORDING_BUFFER = 1.0

# Shortest recording that is still transcribed. Low so "yes"/"no" survive.
MIN_RECORDING_LENGTH = 0.4

# Minimum gap between two recordings. 0 = speech right after a cut is
# still captured (RealtimeSTT's default is 1.0, which drops it).
MIN_GAP_BETWEEN_RECORDINGS = 0.0

# Chunks quieter than this (RMS, 0-1 scale) are NOT sent to the model.
# Calibrate from the `rms` value printed for each chunk: real speech chunks
# should be well above it, noise-only chunks below. 0 disables the gate.
MIN_CHUNK_RMS = 0.006

# Stop the program only if transcription fails this many times in a row.
MAX_CONSECUTIVE_ERRORS = 5

# --- Stitching ---
# If the user pauses for less than this and then keeps talking, the new text
# is appended to the SAME line. Longer than this = new line (new answer).
NEW_ANSWER_GAP_SECONDS = 10.0

# ----------------------------
# Setup output paths
# ----------------------------
os.makedirs("recordings", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
txt_path = f"recordings/transcript_{timestamp}.txt"
wav_path = f"recordings/recording_{timestamp}.wav"

SAMPLE_RATE = 16000
transcript_lines = []

# ----------------------------
# Microphone selection (headset / close-talk mic strongly recommended)
# ----------------------------
MIC_DEVICE_INDEX = None  # set to your headset's index from the list printed at start

def list_input_devices():
    print("\nAvailable input devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  [{i}] {dev['name']}")
    print("\nIf voice is missed or background noise is picked up, set MIC_DEVICE_INDEX")
    print("near the top of this file to your headset's index from the list above.\n")

# ----------------------------
# Per-chunk timing state (set by RealtimeSTT callbacks)
# ----------------------------
CURRENT_START = None      # when the recording currently in progress began
LAST_SEGMENT = None       # (start, end) of the most recently finished recording
CHUNK_SEGMENT = None      # (start, end) of the chunk being transcribed right now
TRANSCRIBE_START = None   # when the model started on the current chunk
CHUNK_RMS = None          # loudness (RMS) of the current chunk
LAST_REAL_END = None      # end time of the last ACCEPTED chunk (for gap measuring)

LATENCIES = []
MODEL_TIMES = []

def _on_recording_start():
    global CURRENT_START
    CURRENT_START = time.time()

def _on_recording_stop():
    global LAST_SEGMENT
    if CURRENT_START is not None:
        LAST_SEGMENT = (CURRENT_START, time.time())

def _rms(audio) -> float:
    """Loudness of a chunk on a 0-1 scale (handles int16 or float input)."""
    a = np.asarray(audio, dtype=np.float32)
    if a.size == 0:
        return 0.0
    if float(np.max(np.abs(a))) > 1.5:   # int16 range -> normalize
        a = a / 32768.0
    return float(np.sqrt(np.mean(a * a)))

def _on_transcription_start(*args, **kwargs):
    """Fires right before the model starts on a chunk. Snapshots which
    recording is being transcribed (so timing stays correct even if the user
    already started the next chunk) and applies the energy gate: returning
    True tells RealtimeSTT to skip transcription of this chunk."""
    global CHUNK_SEGMENT, TRANSCRIBE_START, CHUNK_RMS
    CHUNK_SEGMENT = LAST_SEGMENT
    TRANSCRIBE_START = time.time()
    CHUNK_RMS = None
    try:
        if args and args[0] is not None:
            CHUNK_RMS = _rms(args[0])
            if MIN_CHUNK_RMS > 0 and CHUNK_RMS < MIN_CHUNK_RMS:
                log_filtered(f"LOW ENERGY rms={CHUNK_RMS:.4f} < {MIN_CHUNK_RMS}",
                             "<chunk not sent to model>")
                return True
    except Exception:
        pass
    return False  # transcribe normally

# ----------------------------
# Vocabulary hints — fixes misheard names/jargon
# ----------------------------
INITIAL_PROMPT = (
    "Faaiz, Ammar, SmartHire, Muhammad Ammar, Dawood University of "
    "Engineering and Technology, 360 Expert Solutions, domain of AI, "
    "AI backend, multi-agent system, multi-agent workflow."
)

# ============================================================
# VOCABULARY CORRECTION (loaded from candidate_profile.json)
# ============================================================
# candidate_profile.json format:
# { "vocabulary": ["Muhammad Ammar", "Dawood University of Engineering and Technology", "SmartHire"] }
_PROFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "candidate_profile.json")

def _load_vocabulary():
    default_vocab = [
        "Faaiz",
        "SmartHire",
        "domain of AI",
        "AI backend",
        "multi-agent system",
        "multi-agent workflow",
    ]
    if os.path.isfile(_PROFILE_PATH):
        try:
            with open(_PROFILE_PATH, "r", encoding="utf-8") as f:
                profile = json.load(f)
            return profile.get("vocabulary", default_vocab)
        except Exception as e:
            print(f"Could not read candidate_profile.json ({e}), using defaults.")
    else:
        print(f"No candidate_profile.json found at {_PROFILE_PATH} — using default vocabulary.")
        print("Create it before the interview to auto-correct this candidate's specific name/university/company.\n")
    return default_vocab

DOMAIN_VOCABULARY = _load_vocabulary()

# 0-100. Lower = catches more variants but risks wrong replacements.
FUZZY_THRESHOLD = 72

def _best_vocab_match(phrase: str):
    best_entry = None
    best_score = 0
    for entry in DOMAIN_VOCABULARY:
        # Skip wildly different lengths (a short word vs a long phrase).
        len_a, len_b = len(phrase), len(entry)
        shorter, longer = min(len_a, len_b), max(len_a, len_b)
        if longer == 0 or shorter / longer < 0.4:
            continue
        # fuzz.ratio (not partial_ratio): partial_ratio ignores length
        # differences and caused random words to be replaced.
        score = fuzz.ratio(phrase.lower(), entry.lower())
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry, best_score

def apply_corrections(text: str) -> str:
    """Replaces 1-6 word phrases that closely resemble a vocabulary entry."""
    words = text.split()
    n = len(words)
    result = []
    i = 0
    while i < n:
        matched = False
        max_len = min(6, n - i)
        for phrase_len in range(max_len, 0, -1):
            raw_phrase = " ".join(words[i:i + phrase_len])
            clean_phrase = raw_phrase.strip(".,!?")
            best_entry, score = _best_vocab_match(clean_phrase)
            if score >= FUZZY_THRESHOLD:
                result.append(best_entry)
                i += phrase_len
                matched = True
                break
        if not matched:
            result.append(words[i])
            i += 1
    return " ".join(result)

# ----------------------------
# Repetition-loop filters (hallucination safety net)
# ----------------------------
def is_degenerate(text: str, max_repeat: int = 3) -> bool:
    """A run of the same word repeated more than max_repeat times."""
    words = text.lower().split()
    run = 1
    for i in range(1, len(words)):
        if words[i].strip(".,!?") == words[i - 1].strip(".,!?"):
            run += 1
            if run > max_repeat:
                return True
        else:
            run = 1
    return False


def trim_repeated_phrase(text: str, min_repeats: int = 3) -> str:
    """Cuts phrase-level loops ('in future, we can, in future, we can...')
    at the point the loop starts, keeping the real text before it."""
    words = text.split()
    lowered = [w.lower().strip(".,!?") for w in words]
    n = len(words)
    for phrase_len in range(2, 7):
        i = 0
        while i + phrase_len * min_repeats <= n:
            phrase = lowered[i:i + phrase_len]
            repeats = 1
            j = i + phrase_len
            while lowered[j:j + phrase_len] == phrase:
                repeats += 1
                j += phrase_len
            if repeats >= min_repeats:
                return " ".join(words[:i]).strip(" ,.")
            i += 1
    return text

# ----------------------------
# Parallel raw audio recorder -> .wav (written by a background thread)
# ----------------------------
audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()
_stop_writer = threading.Event()

def _audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def _float_to_int16(audio_float: np.ndarray) -> np.ndarray:
    return (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)

wav_file = wave.open(wav_path, "wb")
wav_file.setnchannels(1)
wav_file.setsampwidth(2)
wav_file.setframerate(SAMPLE_RATE)

def _wav_writer():
    while not (_stop_writer.is_set() and audio_queue.empty()):
        try:
            chunk = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        wav_file.writeframes(_float_to_int16(chunk).tobytes())

raw_stream = sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
    callback=_audio_callback,
    device=MIC_DEVICE_INDEX,
)

# ----------------------------
# Callbacks / handlers
# ----------------------------
def on_realtime_update(text):
    """Live partial preview. Only used when ENABLE_LIVE_PREVIEW = True."""
    print(f"\r... {text}", end="", flush=True)

FILTER_LOG_PATH = os.path.join("recordings", "filtered_log.txt")

def log_filtered(reason: str, original_text: str):
    """Every time a filter removes or changes text, it is shown here and
    written to a log file, so nothing vanishes silently."""
    print(f"\n[FILTERED - {reason}] \"{original_text}\"")
    try:
        with open(FILTER_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{reason}] {original_text}\n")
    except Exception:
        pass


def handle_final_text(raw_text: str, finished_at: float):
    """Processes one transcribed chunk. Runs in the main thread right after
    recorder.text() returns, so the timing state above belongs to THIS chunk.

    A pause shorter than NEW_ANSWER_GAP_SECONDS keeps the new text on the
    same line; a longer pause starts a new line.
    """
    global LAST_REAL_END

    text = (raw_text or "").strip()
    if not text:
        return

    seg = CHUNK_SEGMENT or LAST_SEGMENT
    seg_start, seg_end = seg if seg else (None, None)

    # ---- latency report (speech stopped -> text ready) ----
    if seg_end is not None:
        latency = finished_at - seg_end
        audio_len = (seg_end - seg_start) if seg_start else 0
        LATENCIES.append(latency)
        if TRANSCRIBE_START is not None:
            model_time = finished_at - TRANSCRIBE_START
            MODEL_TIMES.append(model_time)
            model_rtf = model_time / audio_len if audio_len > 0 else 0
            rms_txt = f" | rms {CHUNK_RMS:.4f}" if CHUNK_RMS is not None else ""
            print(f"\n[LATENCY] total {latency:.2f}s | model-only {model_time:.2f}s | "
                  f"audio {audio_len:.1f}s | model RTF {model_rtf:.2f}{rms_txt}", end="", flush=True)
        else:
            print(f"\n[LATENCY] total {latency:.2f}s | audio {audio_len:.1f}s", end="", flush=True)

    # ---- repetition-loop filters ----
    trimmed = trim_repeated_phrase(text)
    if trimmed != text:
        log_filtered("REPEATED PHRASE TRIMMED", text)
    text = trimmed
    if not text:
        return

    if is_degenerate(text):
        log_filtered("DEGENERATE (repeated word)", text)
        return

    text = apply_corrections(text)

    # ---- pause gap before this chunk (measured from last ACCEPTED chunk) ----
    # The recorder cuts a chunk only after POST_SPEECH_SILENCE of quiet, so
    # that time is added back to approximate the real pause length.
    gap = None
    if LAST_REAL_END is not None and seg_start is not None:
        gap = max(0.0, seg_start - LAST_REAL_END) + POST_SPEECH_SILENCE
    if seg_end is not None:
        LAST_REAL_END = seg_end

    is_continuation = (
        gap is not None
        and gap < NEW_ANSWER_GAP_SECONDS
        and bool(transcript_lines)
    )

    if is_continuation:
        transcript_lines[-1] = (transcript_lines[-1] + " " + text).strip()
        print(f"\n   + {text}", end="", flush=True)
    else:
        transcript_lines.append(text)
        print(f"\n>> {text}", end="", flush=True)

    # Rewrite the file from memory so stitched text updates the last line.
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(transcript_lines) + "\n")


def print_latency_summary():
    if not LATENCIES:
        return
    avg = sum(LATENCIES) / len(LATENCIES)
    print("\n--- Latency summary (speech stop -> final text) ---")
    print(f"  chunks : {len(LATENCIES)}")
    print(f"  average: {avg:.2f}s")
    print(f"  best   : {min(LATENCIES):.2f}s")
    print(f"  worst  : {max(LATENCIES):.2f}s")
    if MODEL_TIMES:
        print(f"  model-only avg: {sum(MODEL_TIMES) / len(MODEL_TIMES):.2f}s "
              f"(min {min(MODEL_TIMES):.2f}s, max {max(MODEL_TIMES):.2f}s)")
    print(f"  config : model={MODEL_NAME}, beam_size={BEAM_SIZE}, "
          f"live_preview={ENABLE_LIVE_PREVIEW}, silence={POST_SPEECH_SILENCE}s")


def run_grammar_correction(input_path: str, output_path: str):
    """Generic English grammar pass. Runs AFTER the session ends, so it never
    slows down live capture. Cannot fix names/unknown company words (that is
    what the vocabulary correction above is for).

    Uses AutoModelForSeq2SeqLM directly: newer `transformers` versions no
    longer ship the "text2text-generation" pipeline task."""
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    except ImportError:
        print("\nSkipping grammar correction: run `pip install transformers torch sentencepiece` to enable it.")
        return

    print("\nRunning grammar correction pass on the saved transcript...")
    model_name = _local_grammar("vennify/t5-base-grammar-correction")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        gmodel = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        gmodel.eval()
    except Exception as e:
        print(f"Could not load grammar correction model ({e}). Skipping.")
        return

    def _correct(sentence: str) -> str:
        inputs = tokenizer(f"grammar: {sentence}", return_tensors="pt",
                           truncation=True, max_length=128)
        with torch.no_grad():
            out = gmodel.generate(**inputs, max_length=128, num_beams=4)
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    corrected_lines = []
    for line in lines:
        # Split into sentences so each piece stays short for this model.
        sentences = re.split(r"(?<=[.!?]) +", line)
        corrected_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                candidate = _correct(sentence)
                # If the output length diverges too much, keep the original.
                orig_words = len(sentence.split())
                new_words = len(candidate.split())
                if orig_words > 0 and (new_words < orig_words * 0.5 or new_words > orig_words * 1.8):
                    corrected_sentences.append(sentence)
                else:
                    corrected_sentences.append(candidate)
            except Exception:
                corrected_sentences.append(sentence)
        corrected_lines.append(" ".join(corrected_sentences))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(corrected_lines) + "\n")

    print(f"Grammar-corrected transcript saved to: {output_path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    print(f"Loading models... ({MODEL_NAME} downloads once if not cached yet)")

    list_input_devices()

    desired_kwargs = dict(
        model=_local_model(MODEL_NAME),
        language="en",
        spinner=False,
        on_recording_start=_on_recording_start,
        on_recording_stop=_on_recording_stop,
        on_transcription_start=_on_transcription_start,
        input_device_index=MIC_DEVICE_INDEX,

        # VAD
        post_speech_silence_duration=POST_SPEECH_SILENCE,
        silero_sensitivity=SILERO_SENSITIVITY,
        webrtc_sensitivity=WEBRTC_SENSITIVITY,
        pre_recording_buffer_duration=PRE_RECORDING_BUFFER,
        min_length_of_recording=MIN_RECORDING_LENGTH,
        min_gap_between_recordings=MIN_GAP_BETWEEN_RECORDINGS,

        initial_prompt=INITIAL_PROMPT,
        compute_type="int8",
        beam_size=BEAM_SIZE,
        condition_on_previous_text=False,
        # Keep faster-whisper's own VAD ON: it strips silence/noise inside the
        # chunk (less hallucination) and avoids the "No clip timestamps found"
        # crash in the batched pipeline.
        faster_whisper_vad_filter=True,
        # 0 = use the normal (non-batched) faster-whisper path. The batched
        # pipeline raised RuntimeError("No clip timestamps found") on a chunk
        # with no detectable speech and killed the session.
        batch_size=0,

        # Model-side hallucination detection
        no_speech_threshold=0.7,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
    )

    if ENABLE_LIVE_PREVIEW:
        desired_kwargs.update(
            enable_realtime_transcription=True,
            realtime_model_type="tiny",
            on_realtime_transcription_update=on_realtime_update,
            realtime_processing_pause=REALTIME_PROCESSING_PAUSE,
        )
    else:
        desired_kwargs.update(enable_realtime_transcription=False)

    supported_params = set(inspect.signature(AudioToTextRecorder.__init__).parameters.keys())
    safe_kwargs = {}
    skipped = []
    for key, value in desired_kwargs.items():
        if key in supported_params:
            safe_kwargs[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"Note: your installed RealtimeSTT version doesn't support: {', '.join(skipped)}")
        print("(These were skipped automatically — everything else still applies.)\n")
        if "on_recording_start" in skipped or "on_recording_stop" in skipped:
            print("WARNING: pause stitching needs on_recording_start/on_recording_stop.")
            print("Run `pip install -U RealtimeSTT`, or every pause will start a new line.\n")
        if "min_gap_between_recordings" in skipped:
            print("WARNING: min_gap_between_recordings is not supported, so speech starting")
            print("right after a chunk cut may still be missed. Upgrade RealtimeSTT.\n")

    recorder = AudioToTextRecorder(**safe_kwargs)

    print("Model loaded.\n")
    print("Listening... (Ctrl+C to stop)")
    print(f"Recording will be saved to:  {wav_path}")
    print(f"Transcript will be saved to: {txt_path}\n")

    raw_stream.start()
    writer_thread = threading.Thread(target=_wav_writer, daemon=True)
    writer_thread.start()

    consecutive_errors = 0
    try:
        while True:
            CHUNK_SEGMENT = None
            try:
                raw_text = recorder.text()      # blocks until a chunk is transcribed
                consecutive_errors = 0
            except Exception as e:
                # One bad chunk must never end an interview session.
                consecutive_errors += 1
                print(f"\n[ERROR] chunk skipped ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("Too many errors in a row — stopping.")
                    break
                continue
            handle_final_text(raw_text, time.time())
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        recorder.shutdown()
        raw_stream.stop()
        raw_stream.close()
        _stop_writer.set()
        writer_thread.join(timeout=5)
        wav_file.close()
        print(f"Saved recording:  {wav_path}")
        print(f"Saved transcript: {txt_path}")

        print_latency_summary()

        corrected_path = txt_path.replace(".txt", "_corrected.txt")
        run_grammar_correction(txt_path, corrected_path)