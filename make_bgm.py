"""情缘签背景乐渲染：D调五声拨弦 + 低音暖垫 + 回声，72秒无缝循环"""
import numpy as np, wave, subprocess, sys

SR = 44100
DUR = 72.0
N = int(SR * DUR)
t = np.arange(N) / SR
rng = np.random.default_rng(1129)  # 阿晏生日做种子

def loop_freq(f):
    """把频率吸附到整数周期上，保证首尾波形无缝衔接"""
    return round(f * DUR) / DUR

# ── 低音暖垫：D2 + A2，被极慢的呼吸推着起伏 ──
pad = np.zeros(N)
for f, amp, breath_cycles in [(73.42, .050, 3), (110.0, .042, 4)]:
    fq = loop_freq(f)
    lfo = .62 + .38 * np.sin(2 * np.pi * breath_cycles * t / DUR)
    pad += amp * lfo * np.sin(2 * np.pi * fq * t)
    # 加一点点高八度泛音让垫子不闷
    pad += amp * .25 * lfo * np.sin(2 * np.pi * fq * 2 * t)

# ── 拨弦：古筝质感 = 基音 + 衰减更快的谐波 ──
PENTA = [293.66, 329.63, 369.99, 440.0, 493.88, 587.33, 659.26]

def pluck_wave(freq, dur=2.6):
    n = int(SR * dur)
    tt = np.arange(n) / SR
    env = np.minimum(tt / .012, 1) * np.exp(-tt * 2.1)
    w = (np.sin(2 * np.pi * freq * tt) * 1.0 +
         np.sin(2 * np.pi * freq * 2 * tt) * .45 * np.exp(-tt * 4) +
         np.sin(2 * np.pi * freq * 3 * tt) * .18 * np.exp(-tt * 7))
    return w * env

L = np.zeros(N); R = np.zeros(N)

def add_wrap(buf, wav_, start_idx, gain):
    """音符尾巴越过曲尾就绕回开头，循环永远接得上"""
    n = len(wav_)
    idx = (np.arange(n) + start_idx) % N
    np.add.at(buf, idx, wav_ * gain)

pos = 0.0
while pos < DUR:
    f = PENTA[rng.integers(len(PENTA))]
    g = .10 + rng.random() * .05
    pan = .30 + rng.random() * .40          # 每声落点略偏左右
    w = pluck_wave(f)
    i0 = int(pos * SR)
    add_wrap(L, w, i0, g * (1 - pan))
    add_wrap(R, w, i0, g * pan)
    if rng.random() < .30:                   # 偶尔补一声纯五度
        f2 = f * 1.5 if f * 1.5 < 700 else f / 2
        i1 = i0 + int((.32 + rng.random() * .3) * SR)
        w2 = pluck_wave(f2)
        add_wrap(L, w2, i1, g * .7 * pan)
        add_wrap(R, w2, i1, g * .7 * (1 - pan))
    pos += 1.9 + rng.random() * 2.6

# ── 回声：0.42s 反馈延迟，尾巴同样绕回 ──
d = int(.42 * SR)
for buf in (L, R):
    echo = np.copy(buf)
    for k in range(1, 4):
        echo = np.roll(echo, d) * .34
        buf += echo

L += pad; R += pad

peak = max(np.abs(L).max(), np.abs(R).max())
L *= .82 / peak; R *= .82 / peak

stereo = np.empty(N * 2, dtype=np.int16)
stereo[0::2] = (L * 32767).astype(np.int16)
stereo[1::2] = (R * 32767).astype(np.int16)

wav_path = sys.argv[1] + ".wav"
with wave.open(wav_path, "wb") as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes(stereo.tobytes())

subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "96k", sys.argv[1]],
               check=True, capture_output=True)
print("done:", sys.argv[1])
