# 03_predict.py - Complete Input Specifications

## 📋 Table of Contents
1. [Interactive Mode Inputs](#interactive-mode)
2. [Batch Mode Inputs](#batch-mode)
3. [Response Curve Mode Inputs](#response-curve-mode)
4. [Valid Values Reference](#valid-values)
5. [Examples](#examples)

---

## 🎯 Interactive Mode

**Command:**
```bash
python 03_predict.py
# OR
python 03_predict.py --mode interactive
```

### Required Inputs (Prompted Interactively)

| Input Prompt | Parameter Name | Data Type | Valid Range/Values | Example |
|--------------|----------------|-----------|-------------------|---------|
| "Inhibitor name:" | `inhibitor_name` | string | Any text | `Curry leaf extract` |
| "Concentration (mg/L):" | `inhibitor_conc_mg_L` | float | 0 - 5000 | `1000` |
| "Temperature (°C):" | `temperature_C` | float | 0 - 100 | `25` |
| "H2SO4 molarity (M):" | `acid_molarity_M` | float | 0.1 - 2.0 | `0.5` |
| "Immersion time (hours):" | `immersion_time_h` | float | 0.5 - 200 | `6` |
| "Steel grade:" | `steel_grade` | string | See valid values | `ASTM A36` |
| "Method:" | `method` | string | See valid values | `Weight loss` |

### Auto-Set Parameters
- `acid`: Automatically set to `"H2SO4"` (system is trained for sulfuric acid only)

---

## 📊 Batch Mode

**Command:**
```bash
python 03_predict.py --mode batch --input your_file.csv --output results.csv
```

### Command Line Arguments

| Argument | Required? | Description | Example |
|----------|-----------|-------------|---------|
| `--mode batch` | ✅ Required | Activates batch mode | `--mode batch` |
| `--input FILE.csv` | ✅ Required | Path to input CSV | `--input experiments.csv` |
| `--output FILE.csv` | ❌ Optional | Path to save results | `--output predictions.csv` |

### CSV File Format

**Required Columns (must be in your CSV):**

| Column Name | Data Type | Description | Example Values |
|-------------|-----------|-------------|----------------|
| `inhibitor_name` | string | Name of corrosion inhibitor | `Curry leaf extract`, `Peanut shell extract` |
| `inhibitor_conc_mg_L` | float | Concentration in mg/L | `500`, `1000`, `2000` |
| `temperature_C` | float | Temperature in Celsius | `25`, `30`, `40` |
| `acid_molarity_M` | float | H2SO4 molarity | `0.5`, `1.0` |
| `immersion_time_h` | float | Immersion time in hours | `6`, `12`, `24` |
| `acid` | string | Type of acid (must be "H2SO4") | `H2SO4` |
| `steel_grade` | string | Type of steel | `ASTM A36`, `mild steel` |
| `method` | string | Experimental method | `Weight loss`, `EIS` |

**Example CSV File:**
```csv
inhibitor_name,inhibitor_conc_mg_L,temperature_C,acid_molarity_M,immersion_time_h,acid,steel_grade,method
Curry leaf extract,1000,25,0.5,6,H2SO4,ASTM A36,Weight loss
Peanut shell extract,500,30,0.5,12,H2SO4,mild steel,Weight loss
Spinach leaf extract,1500,25,0.5,24,H2SO4,carbon steel,Weight loss
```

### Output Format

The script adds these columns to your input CSV:

| Column Name | Description |
|-------------|-------------|
| `predicted_IE_pct` | Predicted inhibition efficiency (0-100%) |
| `uncertainty_lower` | Lower bound of prediction (IE - 15%) |
| `uncertainty_upper` | Upper bound of prediction (IE + 15%) |

---

## 📈 Response Curve Mode

**Command:**
```bash
python 03_predict.py --mode response_curve --inhibitor "Inhibitor Name" --output curve.png
```

### Command Line Arguments

| Argument | Required? | Description | Example |
|----------|-----------|-------------|---------|
| `--mode response_curve` | ✅ Required | Activates response curve mode | `--mode response_curve` |
| `--inhibitor "NAME"` | ✅ Required | Inhibitor name (use quotes if contains spaces) | `--inhibitor "Curry leaf extract"` |
| `--output FILE.png` | ❌ Optional | Output image filename | `--output my_curve.png` |

### Default Conditions Used

When generating response curves, these standard conditions are used:

| Parameter | Default Value | Can Change? |
|-----------|---------------|-------------|
| `acid` | `H2SO4` | ❌ No |
| `acid_molarity_M` | `0.5` | ✅ Yes (modify in script) |
| `temperature_C` | `25.0` | ✅ Yes (modify in script) |
| `immersion_time_h` | `6.0` | ✅ Yes (modify in script) |
| `steel_grade` | `ASTM A36` | ✅ Yes (modify in script) |
| `method` | `Weight loss` | ✅ Yes (modify in script) |
| `inhibitor_conc_mg_L` | 1 to 3000 (50 points, log scale) | ✅ Yes (modify in script) |

---

## ✅ Valid Values Reference

### 1. `inhibitor_name`
**Type:** String (case-sensitive)

**Examples from training data:**
- `Curry leaf extract`
- `Peanut shell extract`
- `Spinach leaf extract`
- `Reed leaves extract (RLE)`
- `Asteriscus graveolens essential oil (AG oil)`
- `Pulicaria mauritanica essential oil (PM oil)`
- `Warionia saharae essential oil (WS oil)`
- `Pistachio oil`
- `Origanum compactum extract (OCE)`
- `Chromolaena odorata ethanol leaf extract (LECO)`
- `Cordia sebestena leaf extract`
- `Prunus persica leaf extract`
- `PG extract` (Piper guineense)
- `Helichrysum italicum essential oil`

**Note:** You can use any name, but model performs best with inhibitors similar to those in training data.

---

### 2. `inhibitor_conc_mg_L`
**Type:** Float (positive number)

**Valid Range:** 0 - 5000 mg/L
**Recommended Range:** 100 - 3000 mg/L (where model has most training data)

**Common Values:**
- Low: `100`, `200`, `300`
- Medium: `500`, `1000`, `1500`
- High: `2000`, `2500`, `3000`

**Special Notes:**
- `0` = Blank/control sample (no inhibitor)
- Values > 3000 mg/L are **extrapolation** (less reliable)

---

### 3. `temperature_C`
**Type:** Float

**Valid Range:** 0 - 100°C
**Common Range:** 20 - 70°C

**Typical Values:**
- Room temperature: `25`, `25.0`
- Mild heating: `30`, `35`, `40`
- Elevated: `50`, `60`, `70`

**Special Values from Data:**
- `29.85` (303K converted)
- `39.85` (313K converted)
- `49.85` (323K converted)

**Note:** Higher temperatures often decrease IE for physisorption mechanisms

---

### 4. `acid_molarity_M`
**Type:** Float

**Valid Range:** 0.1 - 2.0 M
**Most Common:** `0.5` M (standard test concentration)

**Common Values:**
- `0.1` - Very dilute
- `0.5` - **Standard** (most training data)
- `1.0` - Concentrated
- `2.0` - Highly concentrated

**Note:** Model trained mostly on 0.5M H2SO4

---

### 5. `immersion_time_h`
**Type:** Float

**Valid Range:** 0.5 - 200 hours
**Common Range:** 1 - 72 hours

**Typical Values:**
- Short-term: `1`, `3`, `5`, `6`
- Medium-term: `12`, `24`, `48`
- Long-term: `72`, `96`, `120`, `168`

**Most Common:** `6` hours (standard test duration)

---

### 6. `acid`
**Type:** String

**Valid Values:** `H2SO4` **ONLY**

**Invalid:** `HCl`, `HNO3`, `H3PO4`, etc.

**Note:** Model is specifically trained for sulfuric acid environment. Using other acids will give unreliable results.

---

### 7. `steel_grade`
**Type:** String (case-insensitive)

**Valid Values:**
- `ASTM A36` - American standard structural steel
- `mild steel` - General mild steel
- `carbon steel` - General carbon steel
- `Q235` - Chinese standard carbon steel
- `Q235 carbon steel`
- `AISI 1018 carbon steel`

**Recommended:** Use one of:
- `ASTM A36` (most common in training)
- `mild steel`
- `carbon steel`

**Note:** Similar steel grades are treated as equivalent by the model

---

### 8. `method`
**Type:** String (case-sensitive)

**Valid Values:**
- `Weight loss` - Most common
- `Weight Loss (WL)` - Alternative notation
- `WL` - Short form
- `EIS` - Electrochemical Impedance Spectroscopy
- `PDP` - Potentiodynamic Polarization
- `Electrochemical` - General electrochemical
- `LPR` - Linear Polarization Resistance

**Recommended:** Use `Weight loss` (most training data)

**Note:** Method has minor impact on predictions compared to other features

---

## 📝 Examples

### Example 1: Interactive Mode
```bash
$ python 03_predict.py

Enter experimental conditions:
  Inhibitor name: Curry leaf extract
  Concentration (mg/L): 1500
  Temperature (°C) [default: 25]: 30
  H2SO4 molarity (M) [default: 0.5]: 0.5
  Immersion time (hours) [default: 6]: 12
  Steel grade [default: ASTM A36]: mild steel
  Method [default: Weight loss]: Weight loss

>>> Predicted IE: 72.3% <<<
(Typical uncertainty: ±15%)
```

---

### Example 2: Batch Mode CSV

**File: `my_experiments.csv`**
```csv
inhibitor_name,inhibitor_conc_mg_L,temperature_C,acid_molarity_M,immersion_time_h,acid,steel_grade,method
Curry leaf extract,500,25,0.5,6,H2SO4,ASTM A36,Weight loss
Curry leaf extract,1000,25,0.5,6,H2SO4,ASTM A36,Weight loss
Curry leaf extract,1500,25,0.5,6,H2SO4,ASTM A36,Weight loss
Curry leaf extract,2000,25,0.5,6,H2SO4,ASTM A36,Weight loss
Peanut shell extract,500,30,0.5,12,H2SO4,mild steel,Weight loss
Peanut shell extract,1000,30,0.5,12,H2SO4,mild steel,Weight loss
New inhibitor X,1000,25,0.5,6,H2SO4,carbon steel,Weight loss
```

**Command:**
```bash
python 03_predict.py --mode batch --input my_experiments.csv --output predictions.csv
```

**Output: `predictions.csv`**
```csv
inhibitor_name,inhibitor_conc_mg_L,temperature_C,...,predicted_IE_pct,uncertainty_lower,uncertainty_upper
Curry leaf extract,500,25,...,65.2,50.2,80.2
Curry leaf extract,1000,25,...,72.8,57.8,87.8
Curry leaf extract,1500,25,...,78.1,63.1,93.1
...
```

---

### Example 3: Response Curve

**Command:**
```bash
python 03_predict.py --mode response_curve --inhibitor "Curry leaf extract" --output curry_curve.png
```

**Output:**
- PNG image showing IE vs concentration (1-3000 mg/L)
- Console prints data table
- Curve saved to `curry_curve.png`

---

### Example 4: Testing New Inhibitor

**Scenario:** You want to test a brand new inhibitor not in the training data

**Input (Interactive):**
```
Inhibitor name: My Novel Plant Extract
Concentration (mg/L): 1000
Temperature (°C): 25
H2SO4 molarity (M): 0.5
Immersion time (hours): 6
Steel grade: ASTM A36
Method: Weight loss
```

**Note:** Model will make prediction based on similar inhibitors in training data. Reliability depends on chemical similarity.

---

## ⚠️ Important Notes

### What to Avoid:

❌ **Different Acids**
```csv
acid
HCl          # ← WRONG! Model trained on H2SO4 only
H2SO4        # ← CORRECT
```

❌ **Extreme Concentrations**
```csv
inhibitor_conc_mg_L
10000        # ← Too high (extrapolation)
5            # ← Too low (may work but uncertain)
1000         # ← GOOD (within training range)
```

❌ **Missing Required Columns**
```csv
# Missing 'method' column - will cause error!
inhibitor_name,inhibitor_conc_mg_L,temperature_C,acid_molarity_M,immersion_time_h,acid,steel_grade
```

---

## 🎯 Quick Reference Card

### Minimum Valid Input (Batch CSV):
```csv
inhibitor_name,inhibitor_conc_mg_L,temperature_C,acid_molarity_M,immersion_time_h,acid,steel_grade,method
Test Inhibitor,1000,25,0.5,6,H2SO4,ASTM A36,Weight loss
```

### Recommended Standard Conditions:
- **Concentration:** 1000 mg/L
- **Temperature:** 25°C
- **Acid Molarity:** 0.5 M H2SO4
- **Time:** 6 hours
- **Steel:** ASTM A36 or mild steel
- **Method:** Weight loss

### Most Reliable Predictions When:
✅ Concentration: 100-3000 mg/L
✅ Temperature: 25-70°C
✅ Acid: 0.5M H2SO4
✅ Inhibitor: Similar to training data (plant extracts, natural oils)

---

## 📞 Need Help?

**Common Issues:**

1. **"Missing required feature" error**
   - Check all column names match exactly (case-sensitive!)
   - Ensure no typos: `temperature_C` not `temperature_c`

2. **"could not convert string to float" error**
   - Check numeric columns only contain numbers
   - No text in `inhibitor_conc_mg_L`, `temperature_C`, etc.

3. **Unrealistic predictions (IE > 100% or < 0%)**
   - Should be clipped automatically to [0, 100]
   - If not, check your input values are reasonable

4. **Very different predictions than expected**
   - Verify you're using H2SO4 (not HCl or others)
   - Check concentration is in mg/L (not g/L or ppm)
   - Ensure temperature is in Celsius (not Kelvin or Fahrenheit)

---

**Last Updated:** January 3, 2026
**Compatible with:** 03_predict.py v1.0
