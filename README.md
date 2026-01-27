All of the code is following the gluing method of Photon counting and analog signal from Prototype Lidar developed in NARIT
                         ANALOG CHANNEL
┌──────────────────────┐
│    Analog Signal     │
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Background Correction│
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│  k-scale & b-offset  │  ← calibration from overlap region
└─────────┬────────────┘
          │
          ▼
┌──────────────────────┐
│ Analog Scaled for    │
│ Gluing               │
└─────────┬────────────┘
          │
          │
          │                PHOTON COUNTING CHANNEL
          │        ┌───────────────────────────────┐
          │        │      Photon Counting Raw       │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Photon per bin conversion     │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Background Correction         │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Afterpulse Correction         │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Deadtime Counts               │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Deadtime Correction            │
          │        └───────────┬───────────────────┘
          │                    │
          │                    ▼
          │        ┌───────────────────────────────┐
          │        │ Photon Counts (Corrected)     │
          │        └───────────┬───────────────────┘
          │                    │
          └──────────────┬─────┘
                         ▼
                ┌──────────────────────┐
                │ Merge Counts per bin │  ← glue region logic
                └─────────┬────────────┘
                          ▼
                ┌──────────────────────┐
                │   Range² Correction  │
                └──────────────────────┘
