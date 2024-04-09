from tuning import Tuning
import usb.core
import usb.util
import time
import numpy as np
from scipy.io import wavfile

def calculate_spl_from_wav(wav_file):
    # Read the WAV file
    sample_rate, data = wavfile.read(wav_file)

    # Convert audio data to floating point values (-1 to 1 range)
    data = data / np.iinfo(data.dtype).max

    # Calculate the root mean square (RMS) value
    rms = np.sqrt(np.mean(data**2))

    # Reference voltage level in pascals for 0 dB SPL
    ref_voltage = 1e-6 #for underwater

    # Convert RMS to sound pressure level (SPL) in dB
    spl_db = 20 * np.log10(rms / ref_voltage)

    return spl_db

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
doa = []

if dev:
    Mic_tuning = Tuning(dev)
    print (Mic_tuning.direction)
    for i in range(10):
        try:
            print( Mic_tuning.direction)
            doa.append(Mic_tuning.direction)
            time.sleep(1)
        except KeyboardInterrupt:
            break

print(doa)

# Example usage
wav_file = 'output.wav'
spl_db = calculate_spl_from_wav(wav_file)
print("Sound Pressure Level (SPL): {:.2f} dB".format(spl_db))