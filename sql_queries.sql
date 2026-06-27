
-- ────────────────────────────────────────────
-- 1. BASIC OVERVIEW
-- ────────────────────────────────────────────

-- Total customers and churn rate
SELECT
    COUNT(*)                                    AS total_customers,
    SUM(churnflag)                              AS total_churned,
    ROUND(AVG(churnflag) * 100, 2)             AS churn_rate_pct,
    ROUND(AVG(monthlycharges), 2)              AS avg_monthly_charge,
    ROUND(AVG(tenure), 1)                      AS avg_tenure_months
FROM customers;


-- Column-level null check (run per column you care about)
SELECT
    COUNT(*) - COUNT(totalcharges)  AS null_totalcharges,
    COUNT(*) - COUNT(tenure)        AS null_tenure,
    COUNT(*) - COUNT(monthlycharges) AS null_monthly
FROM customers;


-- ────────────────────────────────────────────
-- 2. CHURN BY KEY DIMENSIONS
-- ────────────────────────────────────────────

-- By contract type (most important feature!)
SELECT
    contract,
    COUNT(*)                            AS customers,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct,
    ROUND(AVG(monthlycharges), 2)      AS avg_monthly_charge
FROM customers
GROUP BY contract
ORDER BY churn_rate_pct DESC;


-- By internet service type
SELECT
    internetservice,
    COUNT(*)                            AS customers,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
GROUP BY internetservice
ORDER BY churn_rate_pct DESC;


-- By payment method
SELECT
    paymentmethod,
    COUNT(*)                            AS customers,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
GROUP BY paymentmethod
ORDER BY churn_rate_pct DESC;


-- By senior citizen status
SELECT
    CASE WHEN seniorcitizen = 1 THEN 'Senior' ELSE 'Non-Senior' END AS segment,
    COUNT(*)                            AS customers,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
GROUP BY seniorcitizen;


-- ────────────────────────────────────────────
-- 3. TENURE ANALYSIS
-- ────────────────────────────────────────────

-- Churn by tenure bucket
SELECT
    CASE
        WHEN tenure <= 6   THEN '0-6 months'
        WHEN tenure <= 12  THEN '7-12 months'
        WHEN tenure <= 24  THEN '13-24 months'
        WHEN tenure <= 48  THEN '25-48 months'
        ELSE               '49+ months'
    END                                 AS tenure_group,
    COUNT(*)                            AS customers,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct,
    ROUND(AVG(monthlycharges), 2)      AS avg_monthly_charges
FROM customers
GROUP BY tenure_group
ORDER BY churn_rate_pct DESC;


-- Average tenure for churned vs retained
SELECT
    churn,
    ROUND(AVG(tenure), 1)              AS avg_tenure_months,
    ROUND(AVG(monthlycharges), 2)      AS avg_monthly_charges,
    ROUND(AVG(totalcharges), 2)        AS avg_total_spent
FROM customers
GROUP BY churn;


-- ────────────────────────────────────────────
-- 4. HIGH-RISK SEGMENTS
-- ────────────────────────────────────────────

-- The most dangerous combination: Fiber + Month-to-month
SELECT
    COUNT(*)                            AS segment_size,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct,
    ROUND(AVG(monthlycharges), 2)      AS avg_monthly_charges,
    ROUND(AVG(tenure), 1)              AS avg_tenure_months
FROM customers
WHERE internetservice = 'Fiber optic'
  AND contract = 'Month-to-month';


-- Senior citizens on month-to-month plans
SELECT
    COUNT(*)                            AS segment_size,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
WHERE seniorcitizen = 1
  AND contract = 'Month-to-month';


-- New customers (< 6 months) with no tech support
SELECT
    COUNT(*)                            AS segment_size,
    SUM(churnflag)                      AS churned,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
WHERE tenure <= 6
  AND techsupport = 'No'
  AND internetservice != 'No';


-- ────────────────────────────────────────────
-- 5. BUSINESS IMPACT (REVENUE AT RISK)
-- ────────────────────────────────────────────

-- Monthly revenue at risk from churned customers
SELECT
    churn,
    COUNT(*)                               AS customers,
    ROUND(SUM(monthlycharges), 0)          AS total_monthly_revenue,
    ROUND(AVG(monthlycharges), 2)          AS avg_monthly_charge,
    ROUND(SUM(totalcharges), 0)            AS total_lifetime_revenue
FROM customers
GROUP BY churn;


-- Top 10 highest-value churned customers
SELECT
    customerid,
    tenure,
    monthlycharges,
    totalcharges,
    contract,
    internetservice
FROM customers
WHERE churn = 'Yes'
ORDER BY totalcharges DESC
LIMIT 10;


-- ────────────────────────────────────────────
-- 6. ADD-ON SERVICES IMPACT
-- ────────────────────────────────────────────

-- Does having tech support reduce churn?
SELECT
    techsupport,
    internetservice,
    COUNT(*)                            AS customers,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
WHERE internetservice != 'No'
GROUP BY techsupport, internetservice
ORDER BY churn_rate_pct DESC;


-- Effect of online security on churn
SELECT
    onlinesecurity,
    COUNT(*)                            AS customers,
    ROUND(AVG(churnflag) * 100, 1)     AS churn_rate_pct
FROM customers
WHERE internetservice != 'No'
GROUP BY onlinesecurity
ORDER BY churn_rate_pct DESC;


-- ────────────────────────────────────────────
-- 7. USEFUL FOR INTERVIEW TALKING POINTS
-- ────────────────────────────────────────────

-- Summary dashboard (single query, great for README)
SELECT
    (SELECT COUNT(*) FROM customers)                                          AS total_customers,
    (SELECT SUM(churnflag) FROM customers)                                    AS total_churned,
    (SELECT ROUND(AVG(churnflag)*100,1) FROM customers)                      AS overall_churn_pct,
    (SELECT ROUND(AVG(churnflag)*100,1) FROM customers
     WHERE contract='Month-to-month')                                         AS mtm_churn_pct,
    (SELECT ROUND(AVG(churnflag)*100,1) FROM customers
     WHERE contract='Two year')                                               AS twoyear_churn_pct,
    (SELECT ROUND(SUM(monthlycharges),0) FROM customers
     WHERE churn='Yes')                                                       AS monthly_revenue_at_risk;