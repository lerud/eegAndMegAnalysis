# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: neuroTools
#     language: python
#     name: neurotools
# ---

# %%
import parselmouth as pm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import pandas as pd
from IPython.display import Audio

sns.set()
plt.rcParams["figure.dpi"] = 100
# matplotlib.use('TkAgg')
matplotlib.use("QtAgg")
plt.ion()

# %%
parentDir = "/Users/karl/Dropbox/UMD/"
# parentDir='/media/karl/Data/Dropbox/UMD/'

snd = pm.Sound(parentDir + "audioBehavMixes/3_mix6.wav")
# snd = pm.Sound(parentDir+'testf0.wav')

# %%
# snd_part = snd.extract_part(from_time=1.6, to_time=2.4, preserve_times=True)
snd_part = snd.extract_part(from_time=1.6, to_time=2.4, preserve_times=False)

plt.figure()
plt.plot(snd_part.xs(), snd_part.values.T, linewidth=0.5)
plt.xlim([snd_part.xmin, snd_part.xmax])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.show()


# %%
def draw_spectrogram(spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    sg_db = 10 * np.log10(spectrogram.values)
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max() - dynamic_range, cmap="afmhot")
    plt.ylim([spectrogram.ymin, spectrogram.ymax])
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    plt.colorbar(pad=0.12)


def draw_intensity(intensity):
    plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color="w")
    plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
    plt.grid(False)
    plt.ylim(0)
    plt.ylabel("intensity [dB]")


# %%
# snd=snd_part

# %%
intensity = snd.to_intensity()
spectrogram = snd.to_spectrogram()
plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_intensity(intensity)
plt.xlim([snd.xmin, snd.xmax])
plt.show()


# %%
def draw_pitch(pitch):
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot
    pitch_values = pitch.selected_array["frequency"]
    pitch_values[pitch_values == 0] = np.nan
    plt.plot(pitch.xs(), pitch_values, "o", markersize=5, color="w")
    plt.plot(pitch.xs(), pitch_values, "o", markersize=2)
    plt.grid(False)
    plt.ylim(0, pitch.ceiling)
    plt.ylabel("fundamental frequency [Hz]")


# %%
pitch = snd.to_pitch()

# %%
spectrogram = snd.to_spectrogram(window_length=0.03, maximum_frequency=4000)

plt.figure()
draw_spectrogram(spectrogram)
plt.twinx()
draw_pitch(pitch)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

# %%
# pre_emphasized_snd = snd.copy()
# pre_emphasized_snd.pre_emphasize()
# spectrogram = pre_emphasized_snd.to_spectrogram(window_length=0.03, maximum_frequency=4000)

# plt.figure()
# draw_spectrogram(spectrogram)
# plt.twinx()
# draw_pitch(pitch)
# plt.xlim([snd.xmin, snd.xmax])
# plt.show()

# %%
# pitch.selected[

# %%
dispAttribute = "frequency"

pitchInterped = pitch.interpolate()
# manipulation=pm.praat.call(snd, "To Manipulation", 0.01, 75, 600)
# pitch_tier=pm.praat.call(manipulation, "Extract pitch tier")

plt.figure()
plt.plot(pitch.selected_array[dispAttribute])
plt.show()
plt.figure()
plt.plot(pitch.selected_array[dispAttribute])
plt.plot(pitchInterped.selected_array[dispAttribute])
plt.show()

# %%
t = snd.ts()
x = snd.values.T

# %%

# %%

# %%

# %%

# %%
contour = pitchInterped.selected_array["frequency"]

beginCount = 0
if contour[beginCount] == 0:
    while contour[beginCount] == 0:
        beginCount += 1
        beginFreq = contour[beginCount]
    contour[0:beginCount] = beginFreq

endCount = len(contour) - 1
if contour[endCount] == 0:
    while contour[endCount] == 0:
        endCount -= 1
        endFreq = contour[endCount]
    contour[endCount + 1 :] = endFreq

tShort = np.linspace(t[0], t[-1], len(contour))
createInterpedContour = sp.interpolate.CubicSpline(tShort, contour)
interpedContour = createInterpedContour(t)

plt.figure()
plt.plot(
    t,
    sp.interpolate.interp1d(tShort, pitch.selected_array[dispAttribute])(t),
    t,
    sp.interpolate.interp1d(tShort, pitchInterped.selected_array[dispAttribute])(t),
    t,
    interpedContour,
)
plt.show()


# This works pretty well

invRelPitch = (
    interpedContour[0] / interpedContour
)  # This is the right shape, but not the right calculation, or units


# This works pretty well too

# invRelPitch=(interpedContour*-1+2*interpedContour[0])
# invRelPitch=invRelPitch/invRelPitch[0]
# factor=np.sqrt(2)/2
# invRelPitch=invRelPitch*factor+1-factor


# interpedPeriods=1/interpedContour
# invRelPeriods=interpedPeriods/interpedPeriods[0]

# relPitchIntegral=np.cumsum(invRelPitch-1)/snd.sampling_frequency

fig, ax1 = plt.subplots()
ax1.plot(t, interpedContour, color="tab:blue")
ax1.set_xlabel("Time (Sec)")
ax1.set_ylabel("Frequency (Hz)", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(t, invRelPitch, color="tab:red")
# ax2.plot(t,invPitch,color='tab:red')
ax2.set_ylabel("Frequency (Hz)", color="tab:red")
# ax2.plot(t,invRelPeriods,color='tab:red')
# ax2.set_ylabel('Relative periods (Sec/Sec)', color='tab:red')
plt.show()

# %%
plt.figure()
plt.plot(t, x)
plt.show()
# plt.close('all')
createNewX = sp.interpolate.CubicSpline(t, x)
# warper=-.05*np.sin(2*np.pi*(snd.sampling_frequency/len(x))*t)
# warper=warper+1-warper[0]
# warper=np.linspace(.9,.9,len(x))
# warper=relPitchIntegral
# warper=np.cumsum(invPitch)-invPitch[0]
# warper=np.cumsum(invRelPitch)-invRelPitch[0]
warper = np.cumsum(invRelPitch) - invRelPitch[0]
createNewRelPitch = sp.interpolate.CubicSpline(np.arange(len(invRelPitch)), invRelPitch)
newRelPitch = createNewRelPitch(warper)
newWarper = np.cumsum(newRelPitch) - newRelPitch[0]

# newT=warper/(snd.sampling_frequency)/120
# newT=warper/(snd.sampling_frequency)
newT = newWarper / (snd.sampling_frequency)

# warper=invRelPitch*1.5

# newT=t*warper
plt.figure()
plt.plot(t, warper, t, newWarper)
plt.show()
plt.figure()
plt.plot(t, invRelPitch, t, newRelPitch)
plt.show()
newX = createNewX(newT)
# plt.figure()
# plt.plot(t,x,t,newX)
# plt.show()

plt.figure()
plt.plot(t)
plt.plot(newT)
plt.show()

# %%
Audio(data=x.T, rate=snd.sampling_frequency)

# %%
Audio(data=newX.T, rate=snd.sampling_frequency)

# %%
# plt.close('all')
snd.sampling_frequency

# %%
sp.io.wavfile.write("3_mix6_warped.wav", int(snd.sampling_frequency), newX)

# %%
sndTest = pm.Sound("3_mix6_warped.wav")

# %%
pitchTest = sndTest.to_pitch()
spectrogramTest = sndTest.to_spectrogram(window_length=0.03, maximum_frequency=4000)

plt.figure()
draw_spectrogram(spectrogramTest)
plt.twinx()
draw_pitch(pitchTest)
plt.xlim([snd.xmin, snd.xmax])
plt.show()

# %%
# plt.close('all')

# %%

# %%

# %%

# %%

# %%
len(interpedContour) / 24000

# %%
t[-1]

# %%
# mdict={'interpedContour':interpedContour}
# sp.io.savemat("interpedContour.mat", mdict)

# %%
