from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.stats import norm
import pandas as pd
import os
import cv2

def gaussian_psf(size, sigma):
    r = (size - 1) / 2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    g = np.exp(-(x**2 + y**2)/(2*sigma**2))
    g /= g.sum()
    return g

def pick_sigma(Y, sigmas=np.linspace(0.6, 2.0, 15)):
    best_sigma, best_score = sigmas[0], -np.inf
    for s in sigmas:
        k = int(np.ceil(s*6)) | 1
        psf = gaussian_psf(k, s)
        R = fftconvolve(Y, psf[::-1, ::-1], mode="same")
        score = R.max() / (np.mean(np.abs(R)) + 1e-6)
        if score > best_score:
            best_sigma, best_score = s, score
    return float(best_sigma)

def place_psf(h, w, yc, xc, amp, psf):
    out = np.zeros((h,w), dtype=np.float32)
    kh, kw = psf.shape
    ry = (kh-1)/2; rx = (kw-1)/2
    y0 = int(np.floor(yc - ry)); x0 = int(np.floor(xc - rx))
    dy = (yc - ry) - y0; dx = (xc - rx) - x0
    w00 = (1-dy)*(1-dx); w01 = (1-dy)*dx; w10 = dy*(1-dx); w11 = dy*dx
    def add_block(y, x, weight):
        y1 = max(0, y); x1 = max(0, x)
        y2 = min(h, y + kh); x2 = min(w, x + kw)
        if y2 <= y1 or x2 <= x1: return
        py1 = y1 - y; px1 = x1 - x
        py2 = py1 + (y2 - y1); px2 = px1 + (x2 - x1)
        out[y1:y2, x1:x2] += amp * weight * psf[py1:py2, px1:px2]
    add_block(y0, x0, w00); add_block(y0, x0+1, w01)
    add_block(y0+1, x0, w10); add_block(y0+1, x0+1, w11)
    return out

def refine_subpixel(R, y, x):
    h, w = R.shape
    y = int(np.clip(y, 1, h-2)); x = int(np.clip(x, 1, w-2))
    fxm, fx0, fxp = R[y, x-1], R[y, x], R[y, x+1]
    fym, fy0, fyp = R[y-1, x], R[y, x], R[y+1, x]
    denx = (fxm - 2*fx0 + fxp); deny = (fym - 2*fy0 + fyp)
    ox = 0.0 if denx == 0 else 0.5*(fxm - fxp)/denx
    oy = 0.0 if deny == 0 else 0.5*(fym - fyp)/deny
    ox = float(np.clip(ox, -0.5, 0.5)); oy = float(np.clip(oy, -0.5, 0.5))
    return y + oy, x + ox

def robust_noise_sigma(Y, psf):
    blur = fftconvolve(Y, psf, mode="same")
    high = Y - blur
    return float(np.median(np.abs(high - np.median(high))) / 0.6745)

def detect_leds_autoK(Y, Kmax=50, alpha=0.01):
    h, w = Y.shape; n = h*w
    sigma = pick_sigma(Y)
    psf = gaussian_psf(int(np.ceil(sigma*6)) | 1, sigma)
    sigma_noise = robust_noise_sigma(Y, psf)
    sigma_mf = sigma_noise * np.sqrt(np.sum(psf**2))
    from math import isfinite
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / n)
    peak_thresh = z * sigma_mf

    residual = Y.copy()
    steps = []
    best_aicc = float("inf"); best_k = 0; aicc_worse_streak = 0

    for k in range(1, Kmax+1):
        R = fftconvolve(residual, psf[::-1, ::-1], mode="same")
        peak = float(R.max())
        if peak < peak_thresh:
            break
        y0, x0 = np.unravel_index(np.argmax(R), R.shape)
        y_ref, x_ref = refine_subpixel(R, y0, x0)
        model_unit = place_psf(h, w, y_ref, x_ref, 1.0, psf)
        amp = float(np.sum(model_unit * residual) / (np.sum(model_unit**2) + 1e-8))
        amp = max(0.0, amp)
        residual = residual - place_psf(h, w, y_ref, x_ref, amp, psf)

        sse = float(np.sum(residual**2))
        p = 3*k + 1
        aic = n*np.log(sse/n + 1e-12) + 2*p
        aicc = aic + (2*p*(p+1))/(n - p - 1) if p < n-1 else float("inf")
        bic = n*np.log(sse/n + 1e-12) + p*np.log(n)
        snr = peak / (sigma_mf + 1e-12)

        steps.append({"k":k, "y":y_ref, "x":x_ref, "amp":amp, "sse":sse,
                      "aicc":aicc, "bic":bic, "snr":snr})
        if aicc < best_aicc:
            best_aicc = aicc; best_k = k; aicc_worse_streak = 0
        else:
            aicc_worse_streak += 1
            if aicc_worse_streak >= 2:
                break

    sources = steps[:best_k]
    return {"sigma": sigma, "psf": psf, "steps": steps, "best_k": best_k,
            "sources": sources, "residual": residual, "peak_thresh": peak_thresh}

def run_on_image(img_path, tag):
    Y = np.asarray(Image.open(img_path).convert("L"), dtype=np.float32)/255.0
    h, w = Y.shape
    res = detect_leds_autoK(Y, Kmax=50, alpha=0.01)
    K = res["best_k"]; steps = res["steps"]; sources = res["sources"]

    plt.figure()
    plt.title(f"SSE vs k — {tag}")
    plt.plot([s["k"] for s in steps], [s["sse"] for s in steps], marker="o")
    plt.xlabel("k"); plt.ylabel("SSE"); plt.grid(True); plt.show()

    plt.figure()
    plt.title(f"Information Criteria — {tag}")
    plt.plot([s["k"] for s in steps], [s["aicc"] for s in steps], marker="o", label="AICc")
    plt.plot([s["k"] for s in steps], [s["bic"] for s in steps], marker="o", label="BIC")
    plt.xlabel("k"); plt.ylabel("Score (lower is better)"); plt.legend(); plt.grid(True); plt.show()

    xs = [s["x"] for s in sources]; ys = [s["y"] for s in sources]
    plt.figure(figsize=(4,4))
    plt.title(f"Detections (k={K}) on original — {tag}")
    plt.imshow(Y, origin="upper", extent=(-0.5, w-0.5, h-0.5, -0.5), interpolation="nearest")
    plt.scatter(xs, ys, marker='x', s=300, linewidths=2.5)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    run_on_image("0dd4206a-d9c3-4640-98b8-ad0b6c1df4c2.png", "img1")