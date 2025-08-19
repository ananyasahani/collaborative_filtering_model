# Collaborative Filtering Model for Supplier Recommendations

This project implements a **Collaborative Filtering (CF)** recommendation system to help street food vendors discover reliable suppliers.  
The system uses vendor ratings on multiple supplier quality parameters to predict missing ratings and recommend suppliers.

This was built as part of the **Tutedude Web Development Hackathon**, where the challenge was to design scalable, intelligent tools for vendor–supplier interactions.

---

## 📌 Table of Contents
- [Overview of Collaborative Filtering](#-overview-of-collaborative-filtering)  
- [How the Model Works](#️-how-the-model-works)  
- [Data Structure](#-data-structure)  
- [Mock Data Example](#-mock-data-example)  
- [Prediction Process Step-by-Step](#-prediction-process-step-by-step)  
- [Code Structure](#-code-structure)  
- [Limitations and Future Improvements](#-limitations-and-future-improvements)  
- [How to Run](#️-how-to-run)  

---

## 🔎 Overview of Collaborative Filtering

Collaborative Filtering (CF) is a widely used recommendation technique that predicts a user’s preferences based on the preferences of other similar users.

- **User-Based CF:** Finds vendors with similar preferences and uses their ratings to predict missing values for the target vendor.  
- **Item-Based CF:** Focuses on similarities between items (suppliers), which could be added as an extension.  

### Why CF?
- Vendors don’t rate every supplier → Data is **sparse**.  
- CF fills in these gaps by leveraging rating patterns from similar vendors.  
- It produces **personalized supplier recommendations** (different vendors get different rankings).  

**Similarity Metric:** We use **cosine similarity**, which measures how aligned two vendors’ rating patterns are, regardless of magnitude.  

---

## ⚙️ How the Model Works

1. **Build a User-Item Matrix**  
   Vendors (rows) × Suppliers (columns), where cell values are ratings (1–5). Missing entries are left as `NaN`.

2. **Find Similar Vendors**  
   Use KNN with cosine similarity to find the top `k` similar vendors to a given vendor.

3. **Predict Missing Ratings**  
   For suppliers not rated by the target vendor:  
   - Look at ratings from similar vendors.  
   - Compute a weighted average using similarity scores.  
   - If no neighbors have rated it, fallback to a global or vendor average.  

4. **Recommend Suppliers**  
   Predicted ratings are sorted in descending order, so vendors see their **top recommended suppliers first**.  

---

## 📊 Data Structure

The input dataset has one row per `(vendor, supplier)` interaction:

| vendor_id | supplier_id | rating | freshness_rating | rejection_rate | on_time_delivery | fulfillment_accuracy | value_for_money | customer_support |
|-----------|-------------|--------|------------------|----------------|------------------|----------------------|-----------------|-----------------|
| V1        | S1          | 4.10   | 4.24             | 4.12           | 4.74             | 4.59                 | 3.53            | 3.35            |
| V1        | S2          | 3.88   | 3.95             | 4.20           | 4.18             | 3.73                 | 3.31            | 3.91            |
| V1        | S3          | 3.70   | 4.08             | 3.99           | 3.36             | 4.37                 | 3.18            | 3.25            |
| V1        | S4          | 3.87   | 4.02             | 3.91           | 3.66             | 4.36                 | 3.49            | 3.81            |

- **vendor_id:** Unique ID of a vendor (e.g., V1).  
- **supplier_id:** Unique ID of a supplier (e.g., S1).  
- **rating:** Average rating (1–5), computed as the mean of the individual parameter ratings.  
- **Other columns:** Detailed breakdown of the supplier’s performance (`freshness_rating`, `on_time_delivery`, etc.).  

This data is then pivoted into a **user–item matrix**:

| vendor_id | S1   | S2   | S3   | S4   | S5   |
|-----------|------|------|------|------|------|
| V1        | 4.10 | 3.88 | 3.70 | 3.87 | NaN  |
| V2        | 4.50 | NaN  | NaN  | 4.20 | 3.90 |
| V3        | NaN  | 4.30 | 3.80 | NaN  | 4.00 |

---

## 🔮 Prediction Process Step-by-Step

**Example:** Predict missing ratings for `V1`

1. Check which suppliers `V1` has not rated (e.g., `S5`).  
2. Find the **top-3 most similar vendors** to `V1` using cosine similarity.  
3. Collect their ratings for `S5`.  
4. Compute weighted average:

\[
\text{predicted rating} = \frac{\sum(\text{similarity} \times \text{rating})}{\sum(\text{similarity})}
\]

5. If no similar vendor has rated `S5`, fallback to global/vendor average.  

This produces a **predicted rating vector** for all missing suppliers of `V1`.

---

## 🧩 Code Structure

- **Data Loading/Generation:** Loads JSON mock data (or generates random sparse data).  
- **User–Item Matrix Creation:** Pivot vendors vs suppliers.  
- **Model Training:** Fit KNN on the matrix using cosine similarity.  
- **Prediction Function:** Predict missing ratings for a given vendor.  
- **Export:** Save both mock ratings and predicted recommendations as JSON.  

### Files
- `mockdata.json` → Input dataset.  
- `vendor_ratings_mock.json` → Vendor-supplier ratings.  
- `predicted_ratings_v1.json` → Predictions for vendor V1.  

---

## ⚠️ Limitations and Future Improvements

- **Cold Start Problem:** New vendors/suppliers with no ratings get poor predictions.  
- **Zero-Fill Bias:** Filling NaNs with `0` for cosine similarity may distort similarity; masked similarity would be better.  
- **Scalability:** KNN is brute-force; large datasets would benefit from approximate nearest neighbor (ANN) search.  

### Extensions
- Item-based CF (recommend suppliers similar to previously liked ones).  
- Hybrid model combining vendor ratings + supplier metadata (e.g., product categories, region).  
- Incorporating deep learning (e.g., matrix factorization or neural CF).  

---

## ▶️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/collaborative-filtering-suppliers.git
   cd collaborative-filtering-suppliers
