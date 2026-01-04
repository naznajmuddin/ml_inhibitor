# Research Sources: Green Corrosion Inhibitors & ML Prediction
*Compiled for: H2SO4 environment, concentration-response modeling, ML prediction accuracy*

---

## 📚 Table of Contents
1. [Comprehensive Reviews on Green Corrosion Inhibitors](#reviews)
2. [Machine Learning for Corrosion Prediction](#ml-methods)
3. [Concentration-Response & Adsorption Modeling](#concentration)
4. [Key Recommendations for Your Research](#recommendations)

---

## 1. Comprehensive Reviews on Green Corrosion Inhibitors {#reviews}

### ⭐ PRIMARY REVIEWS - Start Here

**1. "Plant leaf extracts as green corrosion inhibitors of steel in acidic and seawater environments"**
- **Source:** Environmental Science and Pollution Research (2025)
- **URL:** https://link.springer.com/article/10.1007/s11356-025-37116-6
- **Why it's valuable:** Very recent review covering H2SO4 environments, includes synergistic effects (like maple leaves + iodide ions)
- **Key finding:** Wang et al. (2023) demonstrated synergistic effects can significantly boost IE%

**2. "The green plant-based corrosion inhibitors—a sustainable strategy"**
- **Source:** Surface Science and Technology (2025)
- **URL:** https://link.springer.com/article/10.1007/s44251-025-00084-7
- **Why it's valuable:** Lists 65 plant extracts specifically in H2SO4 medium
- **Key findings:** 
  - Best H2SO4 performers: *Sida cordifolia* (99.0%), *Adhatoda vasica* (98.8%)
  - Efficiency range: 42% (*Eriobotrya japonica*) to 99%
  - All tested on mild steel

**3. "Plant extracts as green corrosion inhibitors for different kinds of steel: A review"**
- **Source:** PMC / Journal Article (2024)
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC11304013/
- **Key highlights:**
  - *Equisetum hyemale* extract: 85% efficiency at 1000 ppm in 1M H2SO4
  - Temperature effects analyzed (efficiency decreases with temp in most cases)
  - Provides insight into concentration optimization (typically 500-3000 ppm range)

**4. "Green corrosion inhibitors derived from plant extracts and drugs for mild steel in acid media"**
- **Source:** ScienceDirect (2024)
- **URL:** https://www.sciencedirect.com/science/article/pii/S2666845924001843
- **Focus:** Comprehensive mechanisms and structural requirements

**5. "Green Corrosion Inhibitors Based on Plant Extracts: A Technological and Scientific Prospection"**
- **Source:** Applied Sciences / MDPI (2023)
- **URL:** https://www.mdpi.com/2076-3417/13/13/7482
- **Why it's valuable:** 
  - Analyzed 335 articles + 42 patents
  - Lychee peel extract: 95.7% IE at 3.0 g/L (0.5M H2SO4, 298K)
  - Shows industrial viability of plant extracts

**6. "A review of plant extracts as green corrosion inhibitors for CO2 corrosion"**
- **Source:** npj Materials Degradation (2022)
- **URL:** https://www.nature.com/articles/s41529-021-00201-5
- **Why it's valuable:** Extends to both HCl and H2SO4 studies, excellent methodological overview

---

## 2. Machine Learning for Corrosion Prediction {#ml-methods}

### ⭐ MOST RELEVANT FOR YOUR WORK

**1. "Machine Learning-Driven Prediction of Corrosion Inhibitor Efficiency: Emerging Algorithms"**
- **Source:** Arabian Journal for Science and Engineering (2025) ⭐ **Very Recent**
- **URL:** https://link.springer.com/article/10.1007/s13369-025-10386-5
- **Critical for your research:**
  - Gradient Boosting Regressor performed best (same family as your HistGradientBoosting!)
  - Achieved R² > 0.90 with proper feature engineering
  - Discusses virtual sample generation to augment small datasets
  - **Key insight:** Akrom et al. (2023) showed GBR outperformed KNN and SVM

**2. "Chemoinformatics for corrosion science: Data-driven modeling"**
- **Source:** Molecular Informatics - Wiley (2024)
- **URL:** https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202400082
- **CRITICAL INSIGHTS for your model:**
  - **Problem with IE%:** It's concentration-dependent, making direct comparisons problematic
  - **Recommendation:** Either fix concentration OR include it as a feature (which you're doing ✓)
  - **Saturation issue:** At high IE% (>90%), model sensitivity decreases
  - **Alternative targets:** Consider modeling ln(Kads) instead of IE% for better predictions

**3. "Data-driven corrosion inhibition efficiency prediction incorporating 2D-3D molecular graphs"**
- **Source:** TU Delft / Corrosion Science (2023)
- **URL:** https://repository.tudelft.nl/record/uuid:b08fd89d-8e9c-48ef-9959-f9cdbbd10b6a
- **Why it matters:** 
  - Dataset of 1,241 data points from 184 papers (similar scale to yours)
  - Successfully incorporated inhibitor concentration as a feature
  - Demonstrates cross-category prediction viability

**4. "Reviewing machine learning of corrosion prediction in a data-oriented perspective"**
- **Source:** npj Materials Degradation (2022)
- **URL:** https://www.nature.com/articles/s41529-022-00218-4
- **Key takeaway:** Diversifying variable types (material + environment + concentration) significantly improves performance

**5. "Laying the experimental foundation for corrosion inhibitor discovery through machine learning"**
- **Source:** npj Materials Degradation (2024)
- **URL:** https://www.nature.com/articles/s41529-024-00435-z
- **Valuable for:** Understanding what makes a high-quality training dataset

**6. "A machine learning approach to predict the efficiency of corrosion inhibition by natural products"**
- **Source:** Physica Scripta (2024)
- **URL:** https://ui.adsabs.harvard.edu/abs/2024PhyS...99c6006A/abstract
- **Methods tested:** Random Forest, Gradient Boosting, KNN
- **Your validation:** GB performed excellently with natural organic inhibitors

---

## 3. Concentration-Response & Adsorption Modeling {#concentration}

### Understanding the Underlying Mechanisms

**1. "On the use of the Langmuir and other adsorption isotherms in corrosion inhibition"**
- **Source:** ScienceDirect (2023)
- **URL:** https://www.sciencedirect.com/science/article/pii/S0010938X23001543
- **Why critical:** 
  - Explains why Langmuir isotherm slope should equal 1
  - Provides theoretical basis for adsorption equilibrium constant
  - Helps validate your concentration-response curves

**2. "Adsorption Isotherm Modeling in Corrosion Inhibition Studies"**
- **Source:** IntechOpen (2024)
- **URL:** https://www.intechopen.com/chapters/1185987
- **Key models covered:**
  - Langmuir (most common for plant extracts)
  - Freundlich (for heterogeneous surfaces)
  - Temkin (for chemisorption)
  - Flory-Huggins (multi-site adsorption)

**3. "On the evaluation of metal-corrosion inhibitor interactions by adsorption isotherms"**
- **Source:** ScienceDirect (2023)
- **URL:** https://www.sciencedirect.com/science/article/abs/pii/S0022286023007391
- **Important:** Unit consistency for Kads - concentration units must be reciprocal

**4. Specific Plant Extract Studies with Concentration Data:**

   a. **Justicia brandegeeana in H2SO4**
   - Source: Taylor & Francis (2024)
   - URL: https://www.tandfonline.com/doi/full/10.1080/17518253.2024.2320254
   - **IE:** 92.9-94.1% at different temperatures
   - **Concentration tested:** 75-400 ppm range
   - **Medium:** 1.0 mol/L H2SO4
   
   b. **Malvaviscus arboreus in H2SO4**
   - **IE:** 97.5% at 500 ppm
   - **Medium:** 1.0 mol/L H2SO4
   - **Time:** Stable at 3, 24, and 48 hours

   c. **Brassica rapa in H2SO4**
   - **IE:** 94.3% (PP method), 92.5% (EIS method)
   - **Medium:** Q235 steel in H2SO4

---

## 4. Key Recommendations for Your Research {#recommendations}

### 🎯 Based on Literature Analysis

#### **A. Data Quality & Preprocessing**

1. **Address the Concentration-IE Paradox:**
   - Literature confirms concentration MUST be included as a feature (you're doing this ✓)
   - Consider log-transform of concentration (many isotherms are log-linear)
   - Watch for saturation effects above 90% IE

2. **Feature Engineering Suggestions:**
   - Add derived feature: `log(inhibitor_conc_mg_L)` or `log10(inhibitor_conc_mg_L)`
   - Consider interaction terms: `temperature_C * inhibitor_conc_mg_L`
   - Standardize across different acid molarities using normalization

3. **Handle Temperature Effects:**
   - Literature shows IE typically *decreases* with temperature for physisorption
   - If your data shows opposite, may indicate chemisorption (different mechanism)
   - Include temperature interaction terms

#### **B. Model Improvements**

1. **Address Small Dataset Challenges:**
   - **Virtual sample generation** (mentioned in multiple ML papers)
   - Use of Kernel Density Estimation (KDE) to augment training
   - Bootstrap aggregation for uncertainty quantification

2. **Uncertainty Quantification:**
   - Your current ±18% model uncertainty is reasonable for this field
   - Consider quantile regression for prediction intervals
   - Use ensemble methods to capture epistemic uncertainty

3. **Alternative Target Variables:**
   Based on the chemoinformatics review, consider predicting:
   - `ln(Kads)` - adsorption equilibrium constant
   - `ΔG_ads` - Gibbs free energy of adsorption
   - Both are more theoretically grounded than IE% alone

#### **C. Validation & Interpretation**

1. **Cross-Validation Strategy:**
   - Your GroupKFold by paper_id is excellent practice ✓
   - Consider leaving out entire inhibitor families for external validation
   - Test on concentration ranges outside training data (with appropriate warnings)

2. **Feature Importance Analysis:**
   - Permutation importance is more reliable than built-in feature importance ✓
   - Your approach is aligned with best practices in the ML literature

3. **Physical Constraints:**
   - Enforce IE% ∈ [0, 100] (you're doing this ✓)
   - Consider monotonicity constraints: IE should generally increase with concentration
   - Check predictions against known adsorption isotherms

#### **D. Expanding Your Dataset**

**High-Priority Additions from Literature:**

1. **Well-Characterized Inhibitors in H2SO4:**
   - Lychee peel extract (95.7% @ 3000 mg/L)
   - Malvaviscus arboreus (97.5% @ 500 mg/L)
   - Equisetum hyemale (85% @ 1000 mg/L)
   - Justicia brandegeeana (92-94% @ various conc.)

2. **Data Sources to Mine:**
   - The Surface Science & Technology review lists 65 H2SO4 studies
   - PMC database has excellent structured data
   - Focus on weight loss and electrochemical methods for consistency

#### **E. Reporting & Publication**

1. **Model Transparency:**
   - Report hyperparameters (max_depth=3, min_samples_leaf=10) ✓
   - Include cross-validation scores
   - Discuss extrapolation limits clearly

2. **Comparison Benchmarks:**
   - Compare against simple baselines (mean predictor, linear regression)
   - Reference Akrom et al.'s GBR performance (R² > 0.90)
   - Show improvement over concentration-only models

3. **Practical Applicability:**
   - Provide confidence intervals for predictions
   - Highlight optimal concentration ranges
   - Discuss cost-effectiveness (plant extracts vs. synthetic)

---

## 📊 Quick Reference: Performance Benchmarks from Literature

| Study | Algorithm | R² | Dataset Size | Notes |
|-------|-----------|-----|--------------|-------|
| Akrom 2023 | Gradient Boosting | >0.90 | Natural products | Best performer |
| Ma 2023 | 2D-3D Graphs + NN | High | 1,241 samples | Multi-concentration |
| Herowati 2024 | Random Forest | 0.99* | Pyrimidines | *With virtual samples |
| Your Model | HistGradientBoosting | TBD | ~190 samples | H2SO4 + mild steel |

*Note: R² of 0.99 used virtual sample augmentation - not directly comparable to real-data-only models*

---

## 🔬 Methodology Alignment Checklist

✅ **You're Already Doing:**
- Including concentration as feature
- Using group-based CV (by paper)
- Gradient boosting regressor family
- Permutation feature importance
- Clipping predictions to [0,100]
- Publication-quality visualizations

💡 **Consider Adding:**
- Log-transformed concentration
- Virtual sample generation (KDE)
- Alternative targets (ln(Kads), ΔG_ads)
- Monotonicity constraints
- Quantile regression for uncertainty
- Temperature-concentration interactions

---

## 📖 Additional Reading (Theoretical Foundation)

1. **Langmuir Adsorption Theory:** Wikipedia article provides excellent mathematical foundation
2. **QSPR for Corrosion:** Multiple papers show quantum chemical descriptors can improve predictions
3. **Green Chemistry Principles:** Plant extracts align with 12 principles of green chemistry

---

## 🎓 Citation Note

When citing these sources, prioritize:
1. **Most recent reviews** (2024-2025) for current state-of-art
2. **High-impact journals** (npj Materials Degradation, Corrosion Science)
3. **Method-specific papers** (for your ML approach, cite Akrom 2023 and Ma 2023)
4. **Your specific inhibitors** (if found in literature)

---

**Last Updated:** January 2026  
**Compiled by:** Claude AI Research Assistant  
**Focus Area:** Green corrosion inhibitors in H2SO4, ML prediction, concentration-response modeling
