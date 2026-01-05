# System Architecture: Pre-Crash Intelligence System

## Overview

This document describes the deployable system architecture for the Pre-Crash Intelligence System for Two-Wheelers, designed for Indian traffic conditions.

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TWO-WHEELER SYSTEM                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  IMU Sensor  │───▶│  Processor   │───▶│   Alert System       │  │
│  │  (100 Hz)    │    │  (Edge/Phone)│    │   (Haptic/Audio/LED) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                    │                      │               │
│         └────────────────────┼──────────────────────┘               │
│                              │                                      │
│                              ▼                                      │
│                    ┌──────────────────┐                            │
│                    │  Bluetooth/WiFi  │                            │
│                    │   (Optional)     │                            │
│                    └────────┬─────────┘                            │
│                              │                                      │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │   Cloud Server   │
                    │   (Analytics)    │
                    └──────────────────┘
```

---

## 2. Deployment Options

### Option A: Smartphone-Based System

**Hardware Requirements:**
- Smartphone with 6-axis IMU (accelerometer + gyroscope)
- Phone mount on motorcycle handlebar
- Bluetooth earphone/speaker (optional)

**Software Stack:**
```
┌───────────────────────────────────────┐
│         Android/iOS App               │
├───────────────────────────────────────┤
│  UI Layer (React Native/Flutter)      │
├───────────────────────────────────────┤
│  Prediction Engine (TFLite/CoreML)    │
├───────────────────────────────────────┤
│  Sensor Manager (100 Hz sampling)     │
├───────────────────────────────────────┤
│  Alert Manager (Haptic/Audio)         │
└───────────────────────────────────────┘
```

**Advantages:**
- No additional hardware cost
- Easy software updates
- Wide user base reach
- GPS integration included

**Limitations:**
- Battery drain (~5%/hour)
- Phone positioning variability
- Background task restrictions

---

### Option B: Dedicated Edge Device

**Hardware Requirements:**

| Component | Recommended | Budget |
|-----------|-------------|--------|
| Processor | Raspberry Pi Zero 2W | ESP32 |
| IMU | MPU9250 (9-axis) | MPU6050 (6-axis) |
| Alert | Piezo buzzer + LED | Buzzer only |
| Power | 5V from motorcycle | USB powerbank |
| Case | 3D printed waterproof | Generic enclosure |

**Estimated Cost:** ₹1,500 - ₹3,000

**System Diagram:**
```
┌────────────────────────────────────────────────────────────┐
│                    EDGE DEVICE                             │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  ┌─────────────┐        ┌─────────────────────────────┐   │
│  │   MPU6050   │──I2C──▶│    Raspberry Pi Zero 2W     │   │
│  │   (IMU)     │        │    ┌───────────────────┐    │   │
│  └─────────────┘        │    │  Python Runtime   │    │   │
│                         │    │  ┌─────────────┐  │    │   │
│  ┌─────────────┐        │    │  │ RF Model    │  │    │   │
│  │  12V→5V    │───USB──▶│    │  │ (~15MB)     │  │    │   │
│  │ Converter  │         │    │  └─────────────┘  │    │   │
│  └─────────────┘        │    └───────────────────┘    │   │
│                         └─────────────────────────────┘   │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                                            │
│  ┌─────────────┐        ┌─────────────┐                   │
│  │  Buzzer     │◀──GPIO─│  LED (RGB)  │                   │
│  │  (Alert)    │        │  (Status)   │                   │
│  └─────────────┘        └─────────────┘                   │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**Advantages:**
- Dedicated processing
- Consistent sensor placement
- No phone dependency
- Lower latency

**Limitations:**
- Additional hardware cost
- Installation complexity
- Waterproofing required

---

## 3. Real-Time Processing Pipeline

### 3.1 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     REAL-TIME PROCESSING                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐ │
│    │  Sensor  │────▶│  Buffer  │────▶│ Feature  │────▶│  Model   │ │
│    │  Read    │     │  (1s)    │     │ Extract  │     │ Predict  │ │
│    │  10ms    │     │  100pts  │     │   5ms    │     │   5ms    │ │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘ │
│                                                             │       │
│                                                             ▼       │
│    ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐ │
│    │  Alert   │◀────│  Risk    │◀────│ Temporal │◀────│  Score   │ │
│    │ Trigger  │     │  Level   │     │ Smooth   │     │  Output  │ │
│    │   0ms    │     │   0ms    │     │   0ms    │     │  [0,1]   │ │
│    └──────────┘     └──────────┘     └──────────┘     └──────────┘ │
│                                                                     │
│    Total Latency: ~20ms (well within 1-3s warning window)          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Timing Requirements

| Stage | Max Latency | Actual |
|-------|-------------|--------|
| Sensor Reading | 10ms | 10ms |
| Buffer Management | 1ms | <1ms |
| Feature Extraction | 10ms | 5ms |
| Model Inference | 10ms | 5ms |
| Alert Generation | 1ms | <1ms |
| **Total** | **32ms** | **~20ms** |

---

## 4. Alert System Design

### 4.1 Multi-Modal Alerts

```
Risk Level    │ Haptic          │ Audio           │ Visual
──────────────┼─────────────────┼─────────────────┼─────────────
SAFE          │ None            │ None            │ Green LED
LOW_RISK      │ None            │ None            │ Green LED
MEDIUM_RISK   │ Gentle pulse    │ None            │ Yellow LED
HIGH_RISK     │ Strong pulse    │ Short beep      │ Orange LED  
CRITICAL      │ Continuous      │ Alarm           │ Red Flash
```

### 4.2 Alert Priorities

1. **Haptic First:** Rider feels vibration immediately
2. **Audio Second:** Beep confirms through helmet speaker
3. **Visual Third:** LED visible in peripheral vision

### 4.3 Cooldown & Debouncing

```python
ALERT_COOLDOWN = 3.0  # seconds between same-level alerts
PREDICTION_SMOOTHING = 10  # average over last 10 predictions
FALSE_POSITIVE_THRESHOLD = 3  # require 3 consecutive high-risk
```

---

## 5. Power Management

### 5.1 Smartphone Power Budget

| Component | Power Draw |
|-----------|------------|
| IMU Sampling | ~50mW |
| Processing | ~200mW |
| Display (off) | 0mW |
| Alerts | ~10mW |
| **Total** | ~260mW (~5%/hr on 4000mAh) |

### 5.2 Edge Device Power

| Component | Power Draw |
|-----------|------------|
| Pi Zero 2W | ~300mW |
| MPU6050 | ~10mW |
| Buzzer/LED | ~50mW (active) |
| **Total** | ~360mW (from 12V motorcycle) |

---

## 6. Indian Traffic Optimizations

### 6.1 Scenario-Specific Handling

| Scenario | Detection Strategy |
|----------|-------------------|
| **Lane Splitting** | Lowered lateral threshold to tolerate frequent weaving |
| **Traffic Signals** | Hard braking allowed at stops (context-aware) |
| **Potholes** | Vertical spikes differentiated from crash by duration |
| **Speed Breakers** | Predictable patterns excluded from alerts |
| **Mixed Traffic** | Multi-vehicle proximity tolerance |

### 6.2 Adaptive Thresholds

```python
# Base thresholds (can be adjusted per road type)
THRESHOLDS = {
    'highway': {
        'hard_braking': -4.5,  # Higher tolerance
        'lateral_instability': 3.5,
    },
    'city': {
        'hard_braking': -3.5,  # Lower tolerance
        'lateral_instability': 2.5,
    },
    'village': {
        'hard_braking': -3.0,  # Rougher roads expected
        'lateral_instability': 4.0,
    }
}
```

---

## 7. Future Enhancements

### 7.1 Phase 2 Features
- GPS speed integration
- Road type classification
- Rider behavior profiling
- Weather condition adaptation

### 7.2 Phase 3 Features
- Camera-based obstacle detection
- V2V communication
- Cloud-based fleet analytics
- Insurance integration

---

## 8. Implementation Checklist

### Smartphone App Development
- [ ] IMU sensor access (Android SensorManager / iOS CoreMotion)
- [ ] Background processing service
- [ ] TFLite/CoreML model integration
- [ ] Haptic feedback API
- [ ] Audio alert system
- [ ] Battery optimization
- [ ] UI/UX design

### Edge Device Development
- [ ] Raspberry Pi setup with Python
- [ ] I2C communication with MPU6050
- [ ] Model deployment (joblib/pickle)
- [ ] GPIO control for alerts
- [ ] Power circuit design
- [ ] Waterproof enclosure
- [ ] Mounting hardware
