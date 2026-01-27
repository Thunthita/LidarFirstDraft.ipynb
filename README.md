All of the code is following the gluing method of Photon counting and analog signal from Prototype Lidar developed in NARIT
                         
Photon Counting Processing

raw photon counting
→ photon per bin conversion
→ background correction
→ afterpulse correction
→ deadtime counts
→ deadtime correction
→ corrected photon signal


Analog signal processing

raw analog signal
→ background correction
→ linear scaling (k, b)                  ; Photon_corrected ≈ k × Analog_corrected + b
→ analog signal scaled for gluing        ; Analog_scaled = k × Analog + b, where: k = gain scaling factor  b = offset correction


Channel Gluing (Merge Logic)

corrected photon signal  (near range)
+ scaled analog signal   (far range)
→ merged counts per bin


Range² Correction

merged counts per bin
→ multiply by range²
→ final lidar backscatter profile
